import pandas as pd
import os
import sys

def explore_csv(file_path, output_file):
    """
    Reads a CSV file with various encodings and generates an exploration report.
    """
    print(f"Exploring {file_path}...")
    
    # Try different encodings
    encodings = ['utf-8', 'utf-8-sig', 'big5', 'gbk']
    df = None
    used_encoding = None
    
    for encoding in encodings:
        try:
            print(f"Trying encoding: {encoding}...")
            df = pd.read_csv(file_path, encoding=encoding)
            used_encoding = encoding
            print(f"Successfully read with encoding: {encoding}")
            break
        except Exception as e:
            print(f"Failed with encoding {encoding}: {e}")
            
    if df is None:
        print("Could not read the file with any of the attempted encodings.")
        return

    # Generate Report
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Exploration Report for {file_path}\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Successfully read with encoding: {used_encoding}\n\n")
        
        f.write("First 5 rows:\n")
        f.write(df.head().to_string())
        f.write("\n\n")
        
        f.write("Column Info:\n")
        # Capture info() output
        import io
        buffer = io.StringIO()
        df.info(buf=buffer)
        f.write(buffer.getvalue())
        f.write("\n\n")
        
        f.write("Basic Statistics (Describe):\n")
        f.write(df.describe(include='all').to_string())
        f.write("\n\n")
        
        f.write("Missing Values:\n")
        f.write(df.isnull().sum().to_string())
        f.write("\n\n")

    print(f"Exploration report saved to {output_file}")

if __name__ == "__main__":
    # Define paths
    # Assuming the script is in code/ and data is in ../data/0128/data.csv
    # But using absolute paths based on user input for safety or relative to script
    
    # Base directory assumed to be one level up from code/ or just hardcode relative path based on structure
    # Structure:
    # Work/TTAS/
    #   code/
    #   data/0128/data.csv
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, 'data', '0128', 'data.csv')
    output_path = os.path.join(script_dir, 'exploration_summary.txt')
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        # Fallback to check if running from project root
        data_path_alt = os.path.join('data', '0128', 'data.csv')
        if os.path.exists(data_path_alt):
             data_path = data_path_alt
        else:
             sys.exit(1)
             
    explore_csv(data_path, output_path)
