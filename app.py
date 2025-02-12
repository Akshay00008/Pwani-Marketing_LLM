import os
import sys

# Add src directory to Python path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.append(src_path)

# Import the Streamlit UI
from ui import main

# Run the application
if __name__ == "__main__":
    main()