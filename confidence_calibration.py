import logging
import os
import pickle
from typing import Dict, Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class ConfidenceCalibrator:
    """
    Calibrates confidence scores using isotonic regression to align with true match accuracy.
    Specifically designed for schema matching confidence calibration.
    """
    
    def __init__(self):
        self.isotonic_regressor = IsotonicRegression(y_min=0.0, y_max=1.0, increasing="auto", out_of_bounds='clip')
        self.is_fitted = False
        self.calibration_data = {
            'raw_confidences': [],
            'match_correctness': [],
            'categories': [],
        }

    def collect_training_data(self, predictions: dict[str, tuple[Optional[str], float, Optional[str]]], expected_mapping: dict[str, str]):
        """
        Collect raw confidences and their corresponding match correctness.
        
        Args:
            predictions: Dictionary of predictions
            ground_truth: Dictionary of ground truth values
        """
        for target_col, expected_source in expected_mapping.items():
            if target_col not in predictions:
                logger.error(f"Target column {target_col} not in predictions; skipping.")
                continue
            predicted_source, raw_confidence, reasoning = predictions[target_col]
            is_correct = (predicted_source == expected_source)
            self.calibration_data['raw_confidences'].append(raw_confidence)
            self.calibration_data['match_correctness'].append(int(is_correct))
            self.calibration_data['categories'].append(self.categorize_prediction(predicted_source, expected_source))

    def categorize_prediction(self, predicted_source: Optional[str], expected_source: Optional[str]) -> int:
        if expected_source is not None and predicted_source is not None and predicted_source == expected_source:
            return 0
        elif expected_source is not None and predicted_source is not None and predicted_source != expected_source:
            return 1
        elif expected_source is not None and predicted_source is None:
            return 2
        elif expected_source is None and predicted_source is not None:
            return 3
        elif expected_source is None and predicted_source is None:
            return 4
        else:
            raise ValueError("Unexpected case in categorize_prediction")


    def fit(self, min_samples=100):
        """
        Fit isotonic regression on collected data with train/validation split.
        
        Args:
            min_samples: Minimum number of samples required for fitting
            validation_split: Fraction of data to use for validation (0.0 to 1.0)
        """
        if len(self.calibration_data['raw_confidences']) < min_samples:
            print(f"⚠️ Insufficient data for calibration: {len(self.calibration_data['raw_confidences'])} < {min_samples}")
        
        X = np.array(self.calibration_data['raw_confidences'])
        y = np.array(self.calibration_data['match_correctness'])
        
        # Validate confidence values
        if not np.all((X >= 0) & (X <= 1)):
            print("⚠️ Warning: Some raw confidences are outside [0,1] range. Clipping values.")
            X = np.clip(X, 0.0, 1.0)

        X_train, y_train = X, y
        
        # Sort by confidence for isotonic regression
        sorted_indices = np.argsort(X_train)
        X_sorted = X_train[sorted_indices]
        y_sorted = y_train[sorted_indices]

        self.isotonic_regressor.fit(X_sorted, y_sorted)
        self.is_fitted = True
    
    def calibrate_confidence(self, raw_confidence: float) -> float:
        """
        Apply isotonic regression to calibrate a single confidence score.
        
        Args:
            raw_confidence: Raw confidence score [0, 1]
            
        Returns:
            Calibrated confidence score representing true match probability
        """
        return self.calibrate_confidences(np.array([raw_confidence]))[0]

    def calibrate_confidences(self, raw_confidences: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            print("⚠️ Calibrator not fitted. Returning raw confidences.")
            return raw_confidences

        return self.isotonic_regressor.transform(raw_confidences)
    
    def calibrate_predictions(self, predictions: Dict) -> Dict:
        """
        Apply calibration to all predictions in a mapping dict.
        
        Args:
            predictions: Dict[target_col: (source_col, raw_confidence, reasoning)]

        Returns:
            Dict[target_col: (source_col, calibrated_confidence, reasoning)]
        """
        calibrated = {}
        for target_col, prediction_tuple in predictions.items():
            source_col, raw_conf, reasoning = prediction_tuple
            calibrated_conf = self.calibrate_confidence(raw_conf)
            calibrated[target_col] = (source_col, calibrated_conf, reasoning)
        return calibrated

    def make_buckets(self, raw_confs: np.ndarray, correctness: np.ndarray, n_buckets=20):
        """
        Create confidence buckets for calibration analysis.
        
        Args:
            n_buckets: Number of buckets to create
        Returns:
            List of buckets (Xs, Ys)
        """
        n_buckets = min(n_buckets, len(raw_confs))

        # Use quantiles to get equal-size buckets
        quantiles = np.linspace(0, 1, n_buckets + 1)
        bucket_boundaries = np.quantile(raw_confs, quantiles)
        bucket_boundaries[0] = 0.0
        bucket_boundaries[-1] = 2.0

        # Remove duplicate boundaries (may occur with many identical values)
        bucket_boundaries = np.unique(bucket_boundaries)
        buckets = []
        for i in range(len(bucket_boundaries) - 1):
            lower = bucket_boundaries[i]
            upper = bucket_boundaries[i + 1]
            # Get the Xs and Ys for the current bucket
            in_bin = (raw_confs >= lower) & (raw_confs < upper)
            Xs = raw_confs[in_bin]
            Ys = correctness[in_bin]
            buckets.append((Xs, Ys))

        return buckets

    def plot_calibration_curve(self, save_path: str ):
        """
        Plot reliability diagram showing calibration quality.
        
        Args:
            save_path: Where to save the plot
        """
        if not self.is_fitted:
            print("⚠️ Cannot create buckets - calibrator not fitted")
            return []

        raw_confs = np.array(self.calibration_data['raw_confidences'])
        correctness = np.array(self.calibration_data['match_correctness'])
        if len(raw_confs) == 0:
            print("⚠️ No data to create buckets")
            return []

        buckets = self.make_buckets(raw_confs, correctness)

        plt.figure(figsize=(18, 5))

        # Plot 1: Raw confidence reliability
        plt.subplot(1, 3, 1)
        raw_bin_centers = []
        raw_bin_accuracies = []
        raw_bin_counts = []
        
        for bucket in buckets:
            Xs, Ys = bucket
            if len(Xs) == 0:
                continue
            bin_center = Xs.mean()
            bin_accuracy = Ys.mean()
            bin_count = len(Xs)
            raw_bin_centers.append(bin_center)
            raw_bin_accuracies.append(bin_accuracy)
            raw_bin_counts.append(bin_count)

        plt.plot(raw_bin_centers, raw_bin_accuracies, 'ro-', label='Raw Confidence', markersize=8)
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.7)
        plt.xlabel('Raw Confidence')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        # Add sample count annotations
        for center, accuracy, count in zip(raw_bin_centers, raw_bin_accuracies, raw_bin_counts):
            plt.annotate(f'n={count}', (center, accuracy), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8, alpha=0.7)

        # Plot 2: Calibration function
        plt.subplot(1, 3, 2)
        x_vals = np.linspace(0, 1, 100)
        y_vals = self.isotonic_regressor.transform(x_vals)
        plt.plot(x_vals, y_vals, 'g-', label='Isotonic Regression', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Identity', alpha=0.7)
        plt.xlabel('Raw Confidence')
        plt.ylabel('Calibrated Confidence')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 3: Calibrated confidence reliability
        plt.subplot(1, 3, 3)

        calibrated_confs = self.calibrate_confidences(raw_confs)

        calibrated_buckets = self.make_buckets(calibrated_confs, correctness)

        calibrated_bin_centers = []
        calibrated_bin_accuracies = []
        calibrated_bin_counts = []

        for bucket in calibrated_buckets:
            Xs, Ys = bucket
            if len(Xs) == 0:
                continue
            bin_center = Xs.mean()
            bin_accuracy = Ys.mean()
            bin_count = len(Xs)
            calibrated_bin_centers.append(bin_center)
            calibrated_bin_accuracies.append(bin_accuracy)
            calibrated_bin_counts.append(bin_count)

        plt.plot(calibrated_bin_centers, calibrated_bin_accuracies, 'bo-', label='Calibrated Confidence', markersize=8)
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.7)
        plt.xlabel('Calibrated Confidence')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        
        # Add sample count annotations
        for center, accuracy, count in zip(calibrated_bin_centers, calibrated_bin_accuracies, calibrated_bin_counts):
            plt.annotate(f'n={count}', (center, accuracy), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_confidence_histogram(self, save_path: str ):
        """
        Plot histogram of raw and calibrated confidence scores.
        
        Args:
            save_path: Where to save the plot
        """
        if not self.is_fitted:
            print("⚠️ Cannot plot histogram - calibrator not fitted")
            return

        categories = np.array(self.calibration_data['categories'])
        raw_confs = np.array(self.calibration_data['raw_confidences'])
        calibrated_confs = self.calibrate_confidences(raw_confs)

        labels = ['(E: A, P: A)', '(E: A, P: B)', '(E: A, P: N)', '(E: N, P: A)', '(E: N, P: N)']
        colors = ['#00ff00', '#ff0000', '#800080', '#ffa500', '#0000ff']

        plt.figure(figsize=(12, 5))

        x = [raw_confs[categories == i] for i in range(5)]

        plt.subplot(1, 2, 1)
        plt.hist(x, bins=20, range=(0, 1), alpha=0.5, label=labels, color=colors, density=True, stacked=True)
        plt.xlabel('Raw Confidence')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)

        x = [calibrated_confs[categories == i] for i in range(5)]

        plt.subplot(1, 2, 2)
        plt.hist(x, bins=20, range=(0, 1), alpha=0.5, label=labels, color=colors, density=True, stacked=True)
        plt.xlabel('Calibrated Confidence')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def save(self, filepath: str):
        """Save calibrator to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'isotonic_regressor': self.isotonic_regressor,
                'is_fitted': self.is_fitted,
                'calibration_data': self.calibration_data,
            }, f)
    
    @classmethod
    def load(cls, filepath: str):
        """Load calibrator from file"""
        instance = cls()
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        instance.isotonic_regressor = data['isotonic_regressor']
        instance.is_fitted = data['is_fitted']
        instance.calibration_data = data['calibration_data']
        return instance

    def generate_latex_calibration_plot(self, save_path: str):
        """
        Generate LaTeX code for calibration plot using pgfplots
        """
        if not self.is_fitted:
            print("⚠️ Cannot generate LaTeX plot - calibrator not fitted")
            return None
        
        # Calculate data for reliability diagram
        raw_confs = np.array(self.calibration_data['raw_confidences'])
        correctness = np.array(self.calibration_data['match_correctness'])

        # Generate LaTeX code
        latex_code = """\\begin{figure}[htbp]
\\centering
\\begin{tikzpicture}
\\begin{axis}[
    width=0.8\\textwidth,
    height=0.6\\textwidth,
    xlabel={Mean Predicted Confidence},
    ylabel={Accuracy},
    title={Confidence Calibration},
    legend pos=south east,
    grid=major,
    xmin=0, xmax=1,
    ymin=0, ymax=1,
]

% Perfect calibration line
\\addplot[black, dashed, thick] coordinates {(0,0) (1,1)};
\\addlegendentry{Perfect Calibration}

% Raw confidence
\\addplot[red, mark=o, thick] coordinates {
"""

        buckets = self.make_buckets(raw_confs, correctness)

        # Add raw confidence data points
        for (Xs, Ys) in buckets:
            if len(Xs) == 0:
                continue
            conf = Xs.mean()
            acc = Ys.mean()
            latex_code += f"    ({conf:.3f},{acc:.3f})\n"
        
        latex_code += """};
\\addlegendentry{Raw Confidence}

% Calibrated confidence
\\addplot[blue, mark=square, thick] coordinates {
"""

        calibrated_confs = self.calibrate_confidences(raw_confs)

        calibrated_buckets = self.make_buckets(calibrated_confs, correctness)

        # Add calibrated confidence data points
        for (Xs, Ys) in calibrated_buckets:
            if len(Xs) == 0:
                continue
            conf = Xs.mean()
            acc = Ys.mean()
            latex_code += f"    ({conf:.3f},{acc:.3f})\n"
        
        latex_code += """};
\\addlegendentry{Calibrated Confidence}

\\end{axis}
\\end{tikzpicture}
\\caption{Reliability diagram showing confidence calibration before and after isotonic regression. Points closer to the diagonal indicate better calibration.}
\\label{fig:confidence_calibration}
\\end{figure}
"""

        # Save LaTeX code to file
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            f.write(latex_code)

        return latex_code