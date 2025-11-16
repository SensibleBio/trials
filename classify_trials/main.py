"""
main.py - Simple script for classifying mRNA clinical trials from a specific file.

This script:
1. Loads mRNA clinical trials from the file mrna_trials_500.json
2. Classifies trials into predefined categories using GPT-3.5 Turbo
3. Generates a summary of classifications

Usage:
    python main.py
"""

import json
import os
import logging # Added

# Initialize logger for main.py
logger = logging.getLogger(__name__) # Added
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # Added

def main():
    """Main function to process, classify trials, and export to Google Sheets."""
    # Fixed input and output files
    input_file = "pulled_data/20251019_203003/merged_trials.json"
    output_file = "pulled_data/20251019_203003/merged_trials_classified.json" # Local JSON output

    logger.info(f"Loading trials from {input_file}...") # Changed to logger

    # Check if input file exists
    if not os.path.exists(input_file):
        logger.error(f"Input file {input_file} not found. Please ensure it exists.") # Changed to logger
        return

    # Load trials from file
    try:
        with open(input_file, "r") as f:
            trials = json.load(f)
        logger.info(f"Loaded {len(trials)} trials from {input_file}") # Changed to logger
    except json.JSONDecodeError:
        logger.error(f"{input_file} is not a valid JSON file.") # Changed to logger
        return
    except Exception as e:
        logger.error(f"An error occurred while loading {input_file}: {e}")
        return

    if not trials:
        logger.warning("No trials found in the input file. Exiting.")
        return

    # Classify trials and export to a new Google Sheet
    logger.info(f"Starting classification for {len(trials)} trials and exporting to Google Sheets...")
    
    # This will classify, save to output_file (local JSON), and export to a new Google Sheet.
    # The new sheet title is specified here.
    # classify_and_export_trials handles logging of sheet URLs internally.
    results = classify_and_export_trials(
        trials,
        output_file=output_file,
        export_to_sheets=True,
        new_spreadsheet_title="mRNA Trial Classifications Demo (Main)" 
        # To use an existing sheet_id, you would pass:
        # spreadsheet_id="YOUR_EXISTING_SHEET_ID" 
        # and remove new_spreadsheet_title or set it to None.
    )

    if results:
        # Generate and print summary to console (classify_and_export_trials handles sheet summary)
        logger.info("Generating console classification summary...") # Changed to logger
        categories = generate_classification_summary(results)
        print_classification_summary(categories) # This function already uses logger internally
        logger.info(f"Local classification results also saved to: {output_file}") # Added
    else:
        logger.error("Classification process did not return any results.")

    logger.info("Process completed.") # Changed to logger

if __name__ == "__main__":
    main()