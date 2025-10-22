import numpy as np
from sklearn.isotonic import IsotonicRegression
import pickle
import os
from typing import Dict, Optional, Tuple, List, Union
import matplotlib.pyplot as plt

class ConfidenceCalibrator:
    """
    Calibrates confidence scores using isotonic regression to align with true match accuracy.
    Specifically designed for schema matching confidence calibration.
    """
    
    def __init__(self):
        self.isotonic_regressor = IsotonicRegression(out_of_bounds='clip')
        self.is_fitted = False
        self.calibration_data = {'raw_confidences': [], 'match_correctness': []}

    def collect_training_data(self, all_predictions, all_ground_truth):
        """
        Collect raw confidences and their corresponding match correctness.
        
        Args:
            all_predictions: List of predictions_by_approach dictionaries
            all_ground_truth: List of ground truth dictionaries
        """
        
        print("####", type(all_predictions), len(all_predictions) if hasattr(all_predictions, '__len__') else 'no len')

        
        # Handle the case where all_predictions is a list of predictions_by_approach dictionaries
        if isinstance(all_predictions, list):
            for predictions_by_approach, ground_truth in zip(all_predictions, all_ground_truth):
                # Skip if ground_truth is None
                if ground_truth is None:
                    print(f"⚠️ Skipping predictions due to missing ground truth")
                    continue
                    
                # predictions_by_approach is a dict where each value contains the predictions
                # We'll collect data from all approaches
                for approach_name, predictions in predictions_by_approach.items():
                    if isinstance(predictions, dict):
                        for target_col, prediction_data in predictions.items():
                            if isinstance(prediction_data, tuple) and len(prediction_data) >= 2:
                                try:
                                    predicted_source, raw_confidence = prediction_data[0], prediction_data[1]
                                except (IndexError, TypeError) as e:
                                    print(f"⚠️ Skipping invalid prediction data for {target_col}: {e}")
                                    continue
                                
                                if target_col in ground_truth:
                                    is_correct = (predicted_source == ground_truth[target_col])
                                    self.calibration_data['raw_confidences'].append(raw_confidence)
                                    self.calibration_data['match_correctness'].append(int(is_correct))

        else:
            # Handle the original case where predictions is a single dictionary
            predictions = all_predictions
            ground_truth = all_ground_truth
            # Skip if ground_truth is None
            if ground_truth is None:
                print(f"⚠️ Skipping predictions due to missing ground truth")
                return
                
            for target_col, (predicted_source, raw_confidence, reasoning) in predictions.items():
                if target_col in ground_truth:
                    is_correct = (predicted_source == ground_truth[target_col])
                    self.calibration_data['raw_confidences'].append(raw_confidence)
                    self.calibration_data['match_correctness'].append(int(is_correct))
    
    
    def fit(self, min_samples=100):
        """
        Fit isotonic regression on collected data.
        
        Args:
            min_samples: Minimum number of samples required for fitting
        """
        if len(self.calibration_data['raw_confidences']) < min_samples:
            print(f"⚠️ Insufficient data for calibration: {len(self.calibration_data['raw_confidences'])} < {min_samples}")
            return False
        
        X = np.array(self.calibration_data['raw_confidences'])
        y = np.array(self.calibration_data['match_correctness'])
        
        # Validate confidence values
        if not np.all((X >= 0) & (X <= 1)):
            print("⚠️ Warning: Some raw confidences are outside [0,1] range. Clipping values.")
            X = np.clip(X, 0.0, 1.0)
        
        # Sort by confidence for isotonic regression
        sorted_indices = np.argsort(X)
        X_sorted = X[sorted_indices]
        y_sorted = y[sorted_indices]
        
        self.isotonic_regressor.fit(X_sorted, y_sorted)
        self.is_fitted = True
            
        print(f"✅ Confidence calibrator fitted on {len(X)} samples")
        return True
    
    def calibrate_confidence(self, raw_confidence: float) -> float:
        """
        Apply isotonic regression to calibrate a single confidence score.
        
        Args:
            raw_confidence: Raw confidence score [0, 1]
            
        Returns:
            Calibrated confidence score representing true match probability
        """
        if not self.is_fitted:
            print("⚠️ Calibrator not fitted. Returning raw confidence.")
            return raw_confidence
        
        # Ensure input is in valid range
        raw_confidence = np.clip(raw_confidence, 0.0, 1.0)
        calibrated = self.isotonic_regressor.transform([raw_confidence])[0]
        
        # Ensure output is in valid range
        return np.clip(calibrated, 0.0, 1.0)
    
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
            # Handle both 2-tuple and 3-tuple formats for backward compatibility
            if len(prediction_tuple) == 2:
                source_col, raw_conf = prediction_tuple
                reasoning = None
            elif len(prediction_tuple) == 3:
                source_col, raw_conf, reasoning = prediction_tuple
            else:
                continue  # Skip invalid formats
                
            calibrated_conf = self.calibrate_confidence(raw_conf)
            
            if reasoning is not None:
                calibrated[target_col] = (source_col, calibrated_conf, reasoning)
            else:
                calibrated[target_col] = (source_col, calibrated_conf)
        return calibrated
    
    def plot_calibration_curve(self, save_path: str = None):
        """
        Plot reliability diagram showing calibration quality.
        """
        if not self.is_fitted:
            print("⚠️ Cannot plot - calibrator not fitted")
            return
        
        # Create reliability diagram
        raw_confs = np.array(self.calibration_data['raw_confidences'])
        correctness = np.array(self.calibration_data['match_correctness'])
        
        # Bin the data
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_centers = []
        bin_accuracies = []
        calibrated_centers = []
        
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Raw confidence reliability
        plt.subplot(1, 2, 1)
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (raw_confs >= bin_lower) & (raw_confs < bin_upper)
            if in_bin.sum() > 0:
                bin_center = (bin_lower + bin_upper) / 2
                bin_accuracy = correctness[in_bin].mean()
                bin_centers.append(bin_center)
                bin_accuracies.append(bin_accuracy)

        plt.plot(bin_centers, bin_accuracies, 'ro-', label='Raw Confidence')
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        plt.xlabel('Mean Predicted Confidence')
        plt.ylabel('Accuracy')
        plt.title('Raw Confidence Calibration')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Calibrated confidence reliability
        plt.subplot(1, 2, 2)
        calibrated_confs = np.array([self.calibrate_confidence(c) for c in raw_confs])
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (calibrated_confs >= bin_lower) & (calibrated_confs < bin_upper)
            if in_bin.sum() > 0:
                bin_center = (bin_lower + bin_upper) / 2
                bin_accuracy = correctness[in_bin].mean()
                calibrated_centers.append(bin_center)
        
        # Recompute for calibrated
        calibrated_bin_centers = []
        calibrated_bin_accuracies = []
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (calibrated_confs >= bin_lower) & (calibrated_confs < bin_upper)
            if in_bin.sum() > 0:
                bin_center = (bin_lower + bin_upper) / 2
                bin_accuracy = correctness[in_bin].mean()
                calibrated_bin_centers.append(bin_center)
                calibrated_bin_accuracies.append(bin_accuracy)
        
        plt.plot(calibrated_bin_centers, calibrated_bin_accuracies, 'bo-', label='Calibrated Confidence')
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        plt.xlabel('Mean Predicted Confidence')
        plt.ylabel('Accuracy')
        plt.title('Calibrated Confidence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Calibration curve saved to {save_path}")
        
        return plt.gcf()
    
    def save(self, filepath: str):
        """Save calibrator to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'isotonic_regressor': self.isotonic_regressor,
                'is_fitted': self.is_fitted,
                'calibration_data': self.calibration_data
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
    
    def get_calibration_stats(self):
        """Get statistics about the calibration quality"""
        if not self.is_fitted:
            return None
        
        raw_confs = np.array(self.calibration_data['raw_confidences'])
        correctness = np.array(self.calibration_data['match_correctness'])
        calibrated_confs = np.array([self.calibrate_confidence(c) for c in raw_confs])
        
        # Calculate Expected Calibration Error (ECE)
        def calculate_ece(confidences, correctness, n_bins=10):
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = correctness[in_bin].mean()
                    avg_confidence_in_bin = confidences[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            return ece
        
        raw_ece = calculate_ece(raw_confs, correctness)
        calibrated_ece = calculate_ece(calibrated_confs, correctness)
        
        return {
            'raw_ece': raw_ece,
            'calibrated_ece': calibrated_ece,
            'improvement': raw_ece - calibrated_ece,
            'n_samples': len(raw_confs)
        }
    
    def generate_latex_calibration_plot(self):
        """
        Generate LaTeX code for calibration plot using pgfplots
        """
        if not self.is_fitted:
            print("⚠️ Cannot generate LaTeX plot - calibrator not fitted")
            return None
        
        # Calculate data for reliability diagram
        raw_confs = np.array(self.calibration_data['raw_confidences'])
        correctness = np.array(self.calibration_data['match_correctness'])
        calibrated_confs = np.array([self.calibrate_confidence(c) for c in raw_confs])
        
        # Bin the data
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        # Calculate raw confidence reliability
        raw_data_points = []
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (raw_confs >= bin_lower) & (raw_confs < bin_upper)
            if in_bin.sum() > 0:
                bin_center = (bin_lower + bin_upper) / 2
                bin_accuracy = correctness[in_bin].mean()
                raw_data_points.append((bin_center, bin_accuracy))
        
        # Calculate calibrated confidence reliability
        calibrated_data_points = []
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (calibrated_confs >= bin_lower) & (calibrated_confs < bin_upper)
            if in_bin.sum() > 0:
                bin_center = (bin_lower + bin_upper) / 2
                bin_accuracy = correctness[in_bin].mean()
                calibrated_data_points.append((bin_center, bin_accuracy))
        
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
        
        # Add raw confidence data points
        for conf, acc in raw_data_points:
            latex_code += f"    ({conf:.3f},{acc:.3f})\n"
        
        latex_code += """};
\\addlegendentry{Raw Confidence}

% Calibrated confidence
\\addplot[blue, mark=square, thick] coordinates {
"""
        
        # Add calibrated confidence data points
        for conf, acc in calibrated_data_points:
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
        os.makedirs("./output", exist_ok=True)
        with open("./output/confidence_calibration_plot.tex", "w") as f:
            f.write(latex_code)
        
        print("✅ LaTeX calibration plot code saved to ./output/confidence_calibration_plot.tex")
        return latex_code