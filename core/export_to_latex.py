import os
import logging

logging.basicConfig(level=logging.INFO)

def export_table_as_latex(data, headers, filename, caption="", label=""):
    """Export table data as LaTeX table format with proper escaping."""
    os.makedirs("output", exist_ok=True)
    filepath = os.path.join("output", filename)
    
    def escape_latex(text):
        """Escape special LaTeX characters and remove ANSI codes."""
        import re
        text = str(text)
        # Remove ANSI color codes
        text = re.sub(r'\033\[[0-9;]*m', '', text)
        # Escape LaTeX special characters
        latex_special_chars = {
            '&': '\\&', '%': '\\%', '$': '\\$', '#': '\\#',
            '_': '\\_', '{': '\\{', '}': '\\}', '~': '\\textasciitilde{}',
            '^': '\\textasciicircum{}'
        }
        for char, escape in latex_special_chars.items():
            text = text.replace(char, escape)
        return text
    
    with open(filepath, 'w') as f:
        num_cols = len(headers)
        col_spec = 'l' * num_cols
        
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write(f"\\begin{{tabular}}{{{col_spec}}}\n")
        f.write("\\toprule\n")
        
        # Write headers
        escaped_headers = [escape_latex(h) for h in headers]
        f.write(" & ".join(escaped_headers) + " \\\\\n")
        f.write("\\midrule\n")
        
        # Write data rows
        for row in data:
            escaped_row = [escape_latex(cell) for cell in row]
            f.write(" & ".join(escaped_row) + " \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        
        if caption:
            f.write(f"\\caption{{{caption}}}\n")
        if label:
            f.write(f"\\label{{{label}}}\n")
        
        f.write("\\end{table}\n")
    
    logging.info(f"LaTeX table exported to {filepath}")
    return filepath





def export_table_as_latex_landscape(data, headers, filename, caption="", label=""):
    """Export wide table as LaTeX landscape table with smaller font"""
    os.makedirs("output", exist_ok=True)
    filepath = os.path.join("output", filename)
    
    with open(filepath, 'w') as f:
        # Landscape table for wide tables
        f.write("\\begin{landscape}\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\small\n")  # Smaller font for wide tables
        
        # Dynamic column specification based on content width
        num_cols = len(headers)
        if num_cols > 8:
            col_spec = 'p{1.5cm}' * num_cols  # Fixed width columns for very wide tables
        else:
            col_spec = 'l' * num_cols
        
        f.write(f"\\begin{{tabular}}{{{col_spec}}}\n")
        f.write("\\toprule\n")
        
        # Write headers with line breaks for long headers
        formatted_headers = []
        for header in headers:
            # Break long headers
            if len(header) > 10:
                header = header.replace(' ', '\\\\ ')
            formatted_headers.append(header)
        
        header_row = " & ".join(formatted_headers) + " \\\\\n"
        f.write(header_row)
        f.write("\\midrule\n")
        
        # Write data rows
        for row in data:
            cleaned_row = []
            for cell in row:
                cell_str = str(cell)
                # Escape LaTeX special characters
                cell_str = cell_str.replace('&', '\\&')
                cell_str = cell_str.replace('%', '\\%')
                cell_str = cell_str.replace('$', '\\$')
                cell_str = cell_str.replace('#', '\\#')
                cell_str = cell_str.replace('_', '\\_')
                cell_str = cell_str.replace('{', '\\{')
                cell_str = cell_str.replace('}', '\\}')
                # Remove ANSI color codes and emoji
                import re
                cell_str = re.sub(r'\033\[[0-9;]*m', '', cell_str)
                cell_str = re.sub(r'[✅❌]', '', cell_str)  # Remove checkmarks
                cleaned_row.append(cell_str)
            
            data_row = " & ".join(cleaned_row) + " \\\\\n"
            f.write(data_row)
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        
        if caption:
            f.write(f"\\caption{{{caption}}}\n")
        if label:
            f.write(f"\\label{{{label}}}\n")
        
        f.write("\\end{table}\n")
        f.write("\\end{landscape}\n")
    
    print(f"✅ LaTeX landscape table saved to {filepath}")
    return filepath