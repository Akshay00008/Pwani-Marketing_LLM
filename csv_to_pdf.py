import csv
import pandas as pd

def convert_csv_to_text(csv_path, txt_path, delimiter=',', format_type='simple'):
    """
    Convert a CSV file to a formatted text file.
    
    Parameters:
    csv_path (str): Path to the input CSV file
    txt_path (str): Path where the text file will be saved
    delimiter (str): CSV delimiter character
    format_type (str): 'simple' for basic format, 'table' for ASCII table format
    """
    # Read CSV file using pandas for better handling of different formats
    df = pd.read_csv(csv_path)
    
    with open(txt_path, 'w', encoding='utf-8') as txt_file:
        if format_type == 'simple':
            # Write headers
            headers = df.columns.tolist()
            txt_file.write(' | '.join(str(header) for header in headers) + '\n')
            txt_file.write('-' * (sum(len(str(header)) for header in headers) + 
                                 3 * (len(headers) - 1)) + '\n')
            
            # Write data rows
            for _, row in df.iterrows():
                txt_file.write(' | '.join(str(value) for value in row) + '\n')
                
        elif format_type == 'table':
            # Calculate column widths
            col_widths = [max(len(str(df[col].max())), len(col)) + 2 
                         for col in df.columns]
            
            # Create separator line
            separator = '+' + '+'.join('-' * width for width in col_widths) + '+\n'
            
            # Write headers
            txt_file.write(separator)
            header_line = '|'
            for header, width in zip(df.columns, col_widths):
                header_line += str(header).center(width) + '|'
            txt_file.write(header_line + '\n')
            txt_file.write(separator)
            
            # Write data rows
            for _, row in df.iterrows():
                data_line = '|'
                for value, width in zip(row, col_widths):
                    data_line += str(value).center(width) + '|'
                txt_file.write(data_line + '\n')
            
            txt_file.write(separator)

# Example usage
if __name__ == "__main__":
    # Simple format example
    convert_csv_to_text("pwani_content.csv", "output_simple.txt", format_type='simple')
    
    # Table format example
    convert_csv_to_text("pwani_content.csv", "output_table.txt", format_type='table')