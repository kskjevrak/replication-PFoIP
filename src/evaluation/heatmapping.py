import re
import numpy as np
import os

def add_heatmap_to_latex_table(latex_table, exclude_from_scale=None):
    """
    Add per-column heatmap coloring with more prominent colors.
    
    Parameters:
    -----------
    exclude_from_scale : list
        Model names to exclude from min/max calculation (but still color)
        e.g., ['Naïve', 'ARMAX'] to exclude from normalization
    """
    
    if exclude_from_scale is None:
        exclude_from_scale = ['Naïve']  # Default exclude naive
    
    def get_color_for_value(value, min_val, max_val):
        """Convert value to smooth green-yellow-red scale (lower is better = greener)"""
        if max_val == min_val:
            return "white"
        
        normalized = (value - min_val) / (max_val - min_val)
        
        # Smooth green-yellow-red color scheme
        if normalized <= 0.1:
            return "green!80"      # Strong green for best
        elif normalized <= 0.25:
            return "green!55"      # Medium green
        elif normalized <= 0.4:
            return "green!35"      # Light green
        elif normalized <= 0.55:
            return "yellow!40"     # Light yellow
        elif normalized <= 0.7:
            return "yellow!60"     # Medium yellow
        elif normalized <= 0.85:
            return "red!25!yellow!25"  # Yellow-red transition
        else:
            return "red!45"        # Softer red for worst
    
    lines = latex_table.split('\n')
    
    # Find data rows
    data_rows = []
    skip_patterns = [r'\\toprule', r'\\midrule', r'\\bottomrule', r'\\addlinespace', 
                    r'\\caption', r'\\label', r'\\begin', r'\\end', r'\\centering']
    
    for i, line in enumerate(lines):
        if ('&' in line and 
            not any(re.search(pattern, line) for pattern in skip_patterns) and
            not line.strip().startswith('Model &')):
            
            cells = [cell.strip() for cell in line.split('&')]
            
            # Get model name (clean it up)
            model_name = cells[0].replace('\\quad', '').strip()

            # Extract numeric values (skip first cell)
            numeric_values = []
            for cell in cells[1:]:
                match = re.search(r'\d+\.?\d*', cell)
                if match and '-' not in cell:  # Skip cells with just dashes
                    try:
                        numeric_values.append(float(match.group()))
                    except ValueError:
                        numeric_values.append(None)
                else:
                    numeric_values.append(None)
            
            if any(v is not None for v in numeric_values):
                data_rows.append((i, line, cells, numeric_values, model_name))

    if not data_rows:
        return "% Required packages: \\usepackage{xcolor,colortbl}\n" + latex_table
    
    # Calculate min/max for each column, EXCLUDING specified models
    num_cols = len(data_rows[0][3])
    col_ranges = []
    col_mins = []

    for col_idx in range(num_cols):
        col_values = []
        for _, _, _, numeric_vals, model_name in data_rows:
            # Skip excluded models when calculating scale
            if not any(excl.lower() in model_name.lower() for excl in exclude_from_scale):
                if col_idx < len(numeric_vals) and numeric_vals[col_idx] is not None:
                    col_values.append(numeric_vals[col_idx])
        
        if col_values:
            col_ranges.append((min(col_values), max(col_values)))
            col_mins.append(min(col_values))
            print(f"Column {col_idx}: range {min(col_values):.3f} - {max(col_values):.3f}")
        else:
            col_ranges.append((0, 1))
            col_mins.append(0)
        
    # Apply colors to ALL rows (including excluded ones)
    modified_lines = {}
    
    for row_idx, original_line, cells, numeric_vals, model_name in data_rows:
        new_cells = [cells[0]]  # Keep model name as-is
        
        # Color numeric cells
        for col_idx, (cell, numeric_val) in enumerate(zip(cells[1:], numeric_vals)):
            if numeric_val is not None and col_idx < len(col_ranges):
                min_val, max_val = col_ranges[col_idx]
                color = get_color_for_value(numeric_val, min_val, max_val)
                
                # Check if this is the best (minimum) value in the column
                is_best = (col_idx < len(col_mins) and 
                        abs(numeric_val - col_mins[col_idx]) < 1e-6)
                
                # Add color and bold if best
                if is_best:
                    colored_cell = re.sub(
                        r'(\d+\.?\d*)', 
                        rf'\\cellcolor{{{color}}}\\textbf{{\1}}', 
                        cell, 
                        count=1
                    )
                else:
                    colored_cell = re.sub(
                        r'(\d+\.?\d*)', 
                        rf'\\cellcolor{{{color}}}\1', 
                        cell, 
                        count=1
                    )
                new_cells.append(colored_cell)
            else:
                new_cells.append(cell)
        
        modified_lines[row_idx] = ' & '.join(new_cells)
    
    # Reconstruct table
    result_lines = []
    for i, line in enumerate(lines):
        if i in modified_lines:
            result_lines.append(modified_lines[i])
        else:
            result_lines.append(line)
    
    return "% Required packages: \\usepackage{xcolor,colortbl}\n" + '\n'.join(result_lines)

def fix_table_spacing(table_content, table_type):
    """
    Fix table spacing - remove extra whitespace from appendix tables
    
    Parameters:
    -----------
    table_content : str
        LaTeX table content
    table_type : str
        Type of table ('main' or zone name)
    
    Returns:
    --------
    str: Modified table content
    """
    if table_type != 'main':
        # For appendix tables, replace tabular* with regular tabular to remove extra spacing
        # Replace the stretched tabular environment with normal spacing
        table_content = re.sub(
            r'\\begin{tabular\*}{\\textwidth}{@{\\extracolsep{\\fill}}([^}]+)@{}}',
            r'\\begin{tabular}{\1}',
            table_content
        )
        table_content = table_content.replace('\\end{tabular*}', '\\end{tabular}')
    
    return table_content

def extract_tables_from_latex(content):
    """
    Extract individual tables from LaTeX content.
    
    Returns:
    --------
    list: List of (table_content, table_type) tuples where table_type is 'main' or zone name
    """
    tables = []
    
    # Pattern to match table environments
    table_pattern = r'\\begin{table\*?}.*?\\end{table\*?}'
    
    # Find all table matches
    for match in re.finditer(table_pattern, content, re.DOTALL):
        table_content = match.group(0)
        
        # Determine table type from label or caption
        if 'label{tab:forecast_accuracy_main}' in table_content:
            table_type = 'main'
        elif 'label{tab:forecast_accuracy_no1}' in table_content:
            table_type = 'no1'
        elif 'label{tab:forecast_accuracy_no2}' in table_content:
            table_type = 'no2'
        elif 'label{tab:forecast_accuracy_no3}' in table_content:
            table_type = 'no3'
        elif 'label{tab:forecast_accuracy_no4}' in table_content:
            table_type = 'no4'
        elif 'label{tab:forecast_accuracy_no5}' in table_content:
            table_type = 'no5'
        else:
            table_type = 'unknown'
        
        tables.append((table_content, table_type))
    
    return tables

def process_latex_file(input_file='forecast_tables.tex', 
                      output_file='forecast_tables_colored.tex',
                      exclude_from_scale=None):
    """
    Read LaTeX tables from input file, apply heatmap coloring, and save to output file.
    
    Parameters:
    -----------
    input_file : str
        Path to input LaTeX file
    output_file : str  
        Path to output LaTeX file
    exclude_from_scale : list
        Model names to exclude from min/max calculation
    """
    
    if exclude_from_scale is None:
        exclude_from_scale = ['Naïve', 'ARMAX']
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return False
    
    # Read input file
    print(f"Reading from {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return False
    
    # Extract tables
    tables = extract_tables_from_latex(content)
    print(f"Found {len(tables)} tables to process")
    
    if not tables:
        print("No tables found in input file.")
        return False
    
    # Process each table
    processed_content_parts = []
    
    # Add header comments and packages
    processed_content_parts.append("% Required packages: \\usepackage{xcolor,colortbl}")
    processed_content_parts.append("% Generated by heatmap processor")
    processed_content_parts.append("")
    
    # Find content before first table
    first_table_start = content.find('\\begin{table')
    if first_table_start > 0:
        pre_content = content[:first_table_start].strip()
        if pre_content:
            processed_content_parts.append(pre_content)
            processed_content_parts.append("")
    
    for i, (table_content, table_type) in enumerate(tables):
        print(f"Processing {table_type} table...")
        
        # Fix table spacing for appendix tables
        table_content = fix_table_spacing(table_content, table_type)
        
        # For main table, duplicate it as no1 table data since they're the same
        if table_type == 'main':
            # Process main table
            colored_table = add_heatmap_to_latex_table(table_content, exclude_from_scale)
            # Remove the package declaration from individual tables since we add it at the top
            colored_table = colored_table.replace("% Required packages: \\usepackage{xcolor,colortbl}\n", "")
            processed_content_parts.append("% Main table for paper body")
            processed_content_parts.append(colored_table)
        elif table_type == 'no1':
            # Use same coloring as main table since data is the same
            colored_table = add_heatmap_to_latex_table(table_content, exclude_from_scale)
            colored_table = colored_table.replace("% Required packages: \\usepackage{xcolor,colortbl}\n", "")
            processed_content_parts.append("% Appendix tables")
            processed_content_parts.append(f"% Table for {table_type}")
            processed_content_parts.append(colored_table)
        else:
            # Process other zone tables
            colored_table = add_heatmap_to_latex_table(table_content, exclude_from_scale)
            colored_table = colored_table.replace("% Required packages: \\usepackage{xcolor,colortbl}\n", "")
            processed_content_parts.append(f"% Table for {table_type}")
            processed_content_parts.append(colored_table)
        
        processed_content_parts.append("")  # Add spacing between tables
    
    # Combine all processed content
    final_content = '\n'.join(processed_content_parts)
    
    # Write output file
    print(f"Writing to {output_file}...")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(final_content)
        print(f"Successfully created {output_file}")
        return True
    except Exception as e:
        print(f"Error writing file: {e}")
        return False

def main():
    """
    Main function to process LaTeX tables with heatmap coloring.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Apply heatmap coloring to LaTeX forecast tables')
    parser.add_argument('--input', '-i', default='forecast_tables.tex', 
                        help='Input LaTeX file (default: forecast_tables.tex)')
    parser.add_argument('--output', '-o', default='forecast_tables_colored.tex',
                        help='Output LaTeX file (default: forecast_tables_colored.tex)')
    parser.add_argument('--exclude', nargs='+', default=['Na\\"{i}ve'],
                        help='Model names to exclude from scaling (default: Naïve ARMAX)')
    
    args = parser.parse_args()
    
    # Process the file
    success = process_latex_file(
        input_file=args.input,
        output_file=args.output, 
        exclude_from_scale=args.exclude
    )
    
    if success:
        print("Heatmap processing completed successfully!")
    else:
        print("Heatmap processing failed.")

if __name__ == "__main__":
    main()