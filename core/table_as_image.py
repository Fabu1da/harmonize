import os
import logging
import pandas as pd
import matplotlib.pyplot as plt

def export_table_as_image(data, headers, filename):
    """Export table data as a high-quality image file."""
    df = pd.DataFrame(data, columns=headers)

    fig, ax = plt.subplots(figsize=(len(headers) * 2, len(data) * 0.6 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns, 
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    os.makedirs("output", exist_ok=True)
    filepath = os.path.join("output", filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    logging.info(f"Table exported to {filepath}")
    