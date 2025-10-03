import json
import logging
import sys
import datetime
from google_sheets_client import GoogleSheetsClient
from config import GOOGLE_SHEETS_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_trial_data(trial):
    """Validate that a trial has the required fields."""
    required_fields = ['nct_id', 'brief_title', 'official_title', 'brief_summary']
    return all(field in trial for field in required_fields)

def format_trial_for_export(trial):
    """Format a trial's data for export to Google Sheets."""
    # Extract key fields, with fallbacks for missing data
    formatted_trial = {
        'NCT ID': trial.get('nct_id', ''),
        'Brief Title': trial.get('brief_title', ''),
        'Overall Status': trial.get('status', ''),
        'Phase': ', '.join(trial.get('phases', [])),
        'Sponsor': trial.get('sponsor', ''),
        'Interventions': json.dumps(trial.get('interventions', {})) if isinstance(trial.get('interventions'), dict) else str(trial.get('interventions', '')),
        'Official Title': trial.get('official_title', ''),
        'Brief Summary': trial.get('brief_summary', ''),
        'Detailed Description': trial.get('detailed_description', ''),
        'Conditions': ', '.join(trial.get('conditions', [])),
        'Start Date': trial.get('start_date', ''),
        'Completion Date': trial.get('completion_date', ''),
        'Last Update': trial.get('last_update', '')
    }
    return formatted_trial

def export_merged_trials(input_file: str, spreadsheet_id: str, verbose: bool = False, limit: int | None = None):
    """
    Export merged trials data to Google Sheets.
    
    Args:
        input_file: Path to the JSON file containing merged trials
        spreadsheet_id: ID of the Google Spreadsheet to export to
        verbose: If True, export all fields found in the data. If False, export concise set.
        limit: Optional limit on the number of trials to export.
    """
    # Initialize the exporter
    exporter = GoogleSheetsClient()
    
    if not exporter.service:
        logger.error("Failed to initialize Google Sheets exporter. Check credentials.")
        return False
    
    # Load the trials data
    try:
        with open(input_file, 'r') as f:
            trials_data = json.load(f)
        logger.info(f"Loaded {len(trials_data)} trials from {input_file}")
        
        processed_trials = []
        formatted_trials = []  # Initialize formatted_trials list
        
        if verbose:
            # Collect all unique keys
            all_keys = set()
            count = 0
            for trial in trials_data:
                if limit is not None and count >= limit:
                    break
                all_keys.update(trial.keys())
                processed_trials.append(trial)
                count += 1
                
            all_keys = sorted(all_keys)
            logger.info(f"Verbose export: exporting fields: {all_keys}")
            
            # Create formatted trials with all fields
            for trial in processed_trials:
                formatted_trial = {}
                for key in all_keys:
                    value = trial.get(key, '')
                    # Handle lists by joining with commas
                    if isinstance(value, list):
                        value = ', '.join(str(v) for v in value)
                    # Handle dictionaries by converting to JSON string
                    elif isinstance(value, dict):
                        try:
                            value = json.dumps(value)
                        except TypeError:
                            value = str(value)
                    formatted_trial[key] = str(value)
                formatted_trials.append(formatted_trial)
            
            # Use all keys as headers
            headers = all_keys
        else:
            # Non-verbose mode: use existing format
            count = 0
            for trial in trials_data:
                if limit is not None and count >= limit:
                    break
                if validate_trial_data(trial):
                    formatted_trial = format_trial_for_export(trial)
                    formatted_trials.append(formatted_trial)
                else:
                    logger.warning(f"Skipping invalid trial: {trial.get('nct_id', 'NO NCT ID')}")
                count += 1

            headers = list(format_trial_for_export({}).keys())
        
        if not formatted_trials:
            logger.error("No valid trials to export after validation")
            return False
            
        logger.info(f"Formatted {len(formatted_trials)} valid trials for export")
        if formatted_trials:
            logger.info("Sample trial data:")
            logger.info(json.dumps(formatted_trials[0], indent=2))
            
    except Exception as e:
        logger.error(f"Error loading or processing trials data: {e}")
        return False
    
    # Generate a timestamped worksheet name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    worksheet_name = f"{timestamp}_RawData" + ("_Verbose" if verbose else "")
    logger.info(f"Preparing worksheet '{worksheet_name}'...")
    
    # Create a new worksheet
    if not exporter.create_worksheet(spreadsheet_id, worksheet_name):
        logger.error(f"Failed to create worksheet '{worksheet_name}'")
        return False
    
    # Export to Google Sheets
    success = exporter.export_trial_data(formatted_trials, spreadsheet_id, worksheet_name=worksheet_name)
    
    if success:
        logger.info(f"Successfully exported {len(formatted_trials)} trials to Google Sheets")
        logger.info(f"Sheet URL: https://docs.google.com/spreadsheets/d/{spreadsheet_id}")
    else:
        logger.error("Failed to export trials to Google Sheets")
    
    return success

if __name__ == "__main__":
    # Use the spreadsheet ID from config
    spreadsheet_id = GOOGLE_SHEETS_CONFIG.get('default_spreadsheet_id')
    if not spreadsheet_id:
        logger.error("No spreadsheet ID found in config")
        exit(1)
    
    # Path to the merged trials file
    input_file = "pulled_data/first_run/final_merged_trials.json"
    
    # Parse command-line arguments
    verbose = False
    limit = None # Set limit to None by default
    if len(sys.argv) > 1:
        if sys.argv[1] == '--verbose':
            verbose = True
        # Check for limit argument
        if len(sys.argv) > 2 and sys.argv[2] == '--limit':
             try:
                 limit = int(sys.argv[3])
             except (ValueError, IndexError):
                 logger.error("Invalid or missing limit value. Usage: python export_merged_trials.py [--verbose] [--limit <number>]")
                 exit(1)
        # Handle --limit without --verbose
        elif sys.argv[1] == '--limit':
             try:
                 limit = int(sys.argv[2])
             except (ValueError, IndexError):
                 logger.error("Invalid or missing limit value. Usage: python export_merged_trials.py [--verbose] [--limit <number>]")
                 exit(1)
    
    # Export the data
    export_merged_trials(input_file, spreadsheet_id, verbose=verbose, limit=limit) 