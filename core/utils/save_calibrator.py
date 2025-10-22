from typing import Any

import os
import matplotlib.pyplot as plt

from confidence_calibration import ConfidenceCalibrator

def train_and_save_calibrator(confidence_calibrator: ConfidenceCalibrator, name: str) -> None:
    """Train and save the confidence calibrator"""
    print(f"\n🎯 Training Confidence {name.capitalize()} Calibrator...")
    if confidence_calibrator and confidence_calibrator.fit():
        # Save the trained calibrator
        
        os.makedirs("./models/confidence_calibrators", exist_ok=True)
        calibrator_path = f"./models/confidence_calibrators/COMA.pkl"
        # calibrator_path = f"./models/confidence_calibrators/{name}.pkl"
        confidence_calibrator.save(calibrator_path)

        # Generate calibration plots
        os.makedirs("./output/confidence_calibration_curve", exist_ok=True)
        fig = confidence_calibrator.plot_calibration_curve(f"./output/confidence_calibration_curve/{name}.png")
        if fig:
            plt.close(fig)  # Close to free memory
        
        # Generate LaTeX calibration plot code
        latex_code = confidence_calibrator.generate_latex_calibration_plot()
        
        # Print calibration statistics
        stats = confidence_calibrator.get_calibration_stats()
        if stats:
            print(f"📊 Confidence Calibration Results:")
            print(f"   Raw ECE: {stats['raw_ece']:.4f}")
            print(f"   Calibrated ECE: {stats['calibrated_ece']:.4f}")
            print(f"   Improvement: {stats['improvement']:.4f}")
            print(f"   Training samples: {stats['n_samples']}")