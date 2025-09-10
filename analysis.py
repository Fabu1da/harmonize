# show distribution of agreement counts
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def show_distribution(agreements):
    """Show distribution of agreement counts"""
    
    # Sum up counts across all dictionaries
    #3/3, 2/3, 1/3, 0/3
    totals = {'all_correct_count': 0, 'two_correct_count': 0, 'one_correct_count': 0, 'none_correct_count': 0}
    for agreement in agreements:
        for key in totals:
            totals[key] += agreement.get(key, 0)

    # Plot with custom labels
    labels = ['3/3', '2/3', '1/3', '0/3']
    values = [totals['all_correct_count'], totals['two_correct_count'], 
              totals['one_correct_count'], totals['none_correct_count']]

    # Generate LaTeX code for the chart
    latex_code = """\\begin{figure}[htbp]
\\centering
\\begin{tikzpicture}
\\begin{axis}[
    ybar,
    xlabel={Agreement Category},
    ylabel={Count},
    title={Distribution of Agreement},
    symbolic x coords={3/3,2/3,1/3,0/3},
    xtick=data,
    ymin=0,
    width=10cm,
    height=6cm
]
\\addplot coordinates {"""

    for label, value in zip(labels, values):
        latex_code += f"\n    ({label},{value})"
    
    latex_code += """
};
\\end{axis}
\\end{tikzpicture}
\\caption{Distribution of Agreement}
\\label{fig:agreement_distribution}
\\end{figure}"""

    # Save LaTeX code to file
    with open('./output/agreement_distribution.tex', 'w') as f:
        f.write(latex_code)
    
    print("LaTeX code saved to agreement_distribution.tex")
    print(latex_code)