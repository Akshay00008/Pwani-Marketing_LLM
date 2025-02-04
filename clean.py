import re
import string

def clean_text_file(input_path, output_path):
    """
    Clean a text file by removing unwanted characters and formatting issues.
    
    Args:
        input_path (str): Path to the input text file
        output_path (str): Path where the cleaned file will be saved
    """
    try:
        # Read the input file
        with open(input_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Cleaning operations
        cleaned_text = text.strip()  # Remove leading/trailing whitespace
        
        # Remove multiple consecutive spaces
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        
        # Remove multiple consecutive newlines
        cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
        
        # Remove special characters but keep basic punctuation
        allowed_chars = string.ascii_letters + string.digits + string.punctuation + ' \n'
        cleaned_text = ''.join(char for char in cleaned_text if char in allowed_chars)
        
        # Write the cleaned text to output file
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(cleaned_text)
            
        print(f"File cleaned successfully. Output saved to: {output_path}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def remove_empty_lines(text):
    """Remove all empty lines from text."""
    return '\n'.join(line for line in text.splitlines() if line.strip())

def normalize_whitespace(text):
    """Normalize whitespace in text."""
    return ' '.join(text.split())

# Example usage
if __name__ == "__main__":
    # Clean a single file
    clean_text_file("/Users/vishalroy/Downloads/ContentGenApp/output_simple.txt", "cleaned_output.txt")
    
    # Clean multiple files in a batch
    import glob
    for file_path in glob.glob("*.txt"):
        output_path = f"cleaned_{file_path}"
        clean_text_file(file_path, output_path)