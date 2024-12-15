import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from typing import Tuple, Dict, Any

# ... (keep all the previous imports and class definitions)

def main(csv_file: str):
    try:
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # Create output filename
        output_file = os.path.splitext(csv_file)[0] + '.plot'
        
        analyzer = DataAnalyzer(csv_file)
        analysis_result = analyzer.analyze_data()
        readme_content = analyzer.create_readme(analysis_result)
        story = analyzer.generate_story(analysis_result)
        
        # Save all output to the .plot file
        with open(output_file, 'w') as f:
            f.write(readme_content)
            f.write("\n\n## Data Story\n")
            f.write(story)
        
        # Save visualizations
        for viz_name, viz_path in analysis_result.visualizations.items():
            new_viz_path = os.path.splitext(output_file)[0] + '_' + viz_name + '.png'
            os.rename(viz_path, new_viz_path)
            logging.info(f"Saved visualization: {new_viz_path}")
        
        logging.info(f"Analysis complete! Results saved in '{output_file}'")
        logging.info(f"Visualizations saved with prefix '{os.path.splitext(output_file)[0]}'")
    except Exception as e:
        logging.error(f"An error occurred during analysis: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run autolysis.py <csv_file>")
        sys.exit(1)
    main(sys.argv[1])

print("Modified autolysis script executed successfully!")
