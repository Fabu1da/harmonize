import dotenv

dotenv.load_dotenv(override=True)

import json

# from config import APPROACH_NAMES
from reports.analyze_errors import compute_confusion, find_failure, generate_latex_confusion, generate_latex_failure
from config import APPROACHES, APPROACH_NAMES
from reports.report import calibrated_impact_summary_table, category_comparison_table, overall_report, per_dataset_accuracy_report, top_methods_ranking
from reports.statistical_analysis import extract_accuracies
from reports.statistical_analysis import anova_test, bootstrap_ci, generate_latex_table, generate_latex_table, paired_t_tests
from reports.generate_performance_table import performance_table
from reports.agreement import create_agreement_report
from scripts.metrics import get_metrics




if __name__ == "__main__":
        
    scores_path = "assets/predicted/scores.json"
    predicted_dir = "assets/predicted"
    expected_dir = "assets/expected"

    
    # Load processed data from a JSON file
    with open("./assets/predicted/scores.json", "r") as f:
        print("Loading data from scores.json")
        data = json.load(f)
        
    # Generate the overall report
    overall_report(data)
    
    # Generate the per-dataset report
    per_dataset_accuracy_report(data)

    # Generate the category comparison table
    category_comparison_table(data)
    
    # Generate the calibration impact summary table
    calibrated_impact_summary_table(data)

    # Generate the top methods ranking
    top_methods_ranking(data)

    # Create agreement report
    create_agreement_report(data)

    # get_metrics()
    
    with open(scores_path, "r") as f:
        scores_data = json.load(f)

    # Compute confusion
    confusion_data = {}
    for approach, approach_name in APPROACHES:
        if approach in scores_data:
            confusion_data[approach] = compute_confusion(approach_name, scores_data, predicted_dir, expected_dir)

    # Find failure analysis
    failure = []
    for approach_name in APPROACH_NAMES:
        failure.extend(find_failure(approach_name, scores_data, predicted_dir, expected_dir, num_analysis=1))

    # Generate confusion matrix LaTeX reports
    generate_latex_confusion(confusion_data)

    # Generate failure analysis LaTeX
    generate_latex_failure(failure)


    print("LaTeX files saved to reports/")

    calibrated_approach_names = ["calibrated/" + name for name in APPROACH_NAMES]
    approach_names = APPROACH_NAMES + calibrated_approach_names

    acc_data = extract_accuracies(data, approach_names)

    # Perform tests
    t_results = paired_t_tests(acc_data)
    anova_result = anova_test(acc_data)
    ci_results = {m: bootstrap_ci(acc_data, m) for m in calibrated_approach_names if m in acc_data}
    # Generate LaTeX
    generate_latex_table(t_results, anova_result, ci_results)

    performance_table()

    print("Statistical tests LaTeX saved to assets/reports/statistical_analysis.tex")