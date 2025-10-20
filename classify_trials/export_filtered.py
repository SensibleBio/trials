from .google_sheets_client import GoogleSheetsClient
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize the client
    client = GoogleSheetsClient()
    
    if not client.service:
        logger.error("Failed to initialize Google Sheets Client.")
        return
    
    # Read the filtered JSON file
    json_file = "20250602_223022_RawData_filtered.json"
    logger.info(f"Reading filtered data from {json_file}...")
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        # Get metadata and trials
        metadata = data["_metadata"]
        trials = data["trials"]
        
        if not trials:
            logger.error("No trial data found in JSON.")
            return
        
        # Set up worksheet
        spreadsheet_id = "1HpyjKdt_n929Pmca0rAyDt1mrw2YDExjoO9CGgIuCso"
        worksheet_name = "20250602_223022_RawData_filtered"
        
        # Check if worksheet exists
        method_name = "list_sheets"
        api_call = lambda: client.service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
        response = client._make_api_request_with_retry(api_call, method_name)
        sheet_titles = [sheet.get('properties', {}).get('title') for sheet in response['sheets']] if response and 'sheets' in response else []
        if worksheet_name not in sheet_titles:
            logger.info(f"Worksheet '{worksheet_name}' does not exist. Creating it...")
            if not client.create_worksheet(spreadsheet_id, worksheet_name):
                logger.error(f"Failed to create worksheet '{worksheet_name}'.")
                return
        
        # Use the standard export method
        logger.info(f"Exporting {len(trials)} records to worksheet '{worksheet_name}'...")
        success = client.export_trial_data(trials, spreadsheet_id, worksheet_name)
        if success:
            logger.info(f"Successfully exported filtered data to worksheet '{worksheet_name}'")
        else:
            logger.error(f"Failed to export filtered data to worksheet '{worksheet_name}'")
            
    except FileNotFoundError:
        logger.error(f"File {json_file} not found.")
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in {json_file}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main() 