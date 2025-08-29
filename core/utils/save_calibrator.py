from typing import Any

import os
import matplotlib.pyplot as plt


def train_and_save_calibrator(gpt_calibrator: Any):
    """Train and save the GPT calibrator"""
    print("\n🎯 Training GPT Isotonic Calibrator...")
    if gpt_calibrator and gpt_calibrator.fit():
        # Save the trained calibrator
        os.makedirs("./models", exist_ok=True)
        calibrator_path = "./models/gpt_isotonic_calibrator.pkl"
        gpt_calibrator.save(calibrator_path)
        
        # Generate calibration plots
        os.makedirs("./output", exist_ok=True)
        fig = gpt_calibrator.plot_calibration_curve("./output/gpt_calibration_curve.png")
        if fig:
            plt.close(fig)  # Close to free memory
        
        # Generate LaTeX calibration plot code
        latex_code = gpt_calibrator.generate_latex_calibration_plot()
        
        # Print calibration statistics
        stats = gpt_calibrator.get_calibration_stats()
        if stats:
            print(f"📊 GPT Calibration Results:")
            print(f"   Raw ECE: {stats['raw_ece']:.4f}")
            print(f"   Calibrated ECE: {stats['calibrated_ece']:.4f}")
            print(f"   Improvement: {stats['improvement']:.4f}")
            print(f"   Training samples: {stats['n_samples']}")