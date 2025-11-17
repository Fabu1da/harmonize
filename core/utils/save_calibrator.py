import os
import matplotlib.pyplot as plt

from confidence_calibration import ConfidenceCalibrator

def save_calibrator(confidence_calibrator: ConfidenceCalibrator, name: str) -> None:
    """Train and save the confidence calibrator"""

    os.makedirs("./models/confidence_calibrators", exist_ok=True)
    calibrator_path = f"./models/confidence_calibrators/{name}.pkl"
    confidence_calibrator.save(calibrator_path)

    # Generate calibration plots
    os.makedirs("./output/confidence_calibration_curve", exist_ok=True)
    confidence_calibrator.plot_calibration_curve(f"./output/confidence_calibration_curve/{name}.png")
    os.makedirs("./output/confidence_histograms", exist_ok=True)
    confidence_calibrator.plot_confidence_histogram(f"./output/confidence_histograms/{name}.png")

    confidence_calibrator.generate_latex_calibration_plot(f"./output/confidence_calibration_plot/{name}.tex")
