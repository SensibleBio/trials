from google_sheets_client import GoogleSheetsClient
import logging
import pandas as pd
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize the client
    client = GoogleSheetsClient()
    
    if not client.service:
        logger.error("Failed to initialize Google Sheets Client. Check credentials.json.")
        return
    
    # Get the spreadsheet ID from config
    spreadsheet_id = "1HpyjKdt_n929Pmca0rAyDt1mrw2YDExjoO9CGgIuCso"
    worksheet_name = "20250602_223022_RawData"
    
    try:
        # Read the data
        logger.info(f"Reading data from worksheet '{worksheet_name}'...")
        data = client.read_sheet_data(spreadsheet_id, worksheet_name)
        
        if data is None:
            logger.error("Failed to read data from the worksheet.")
            return
            
        if not data:
            logger.warning("No data found in the worksheet.")
            return
            
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(data)
        
        # Filter out records where verity is "FP"
        logger.info("Filtering out records where verity is 'FP'...")
        df_filtered = df[df['verity'] != 'FP']
        filtered_data = df_filtered.to_dict('records')
        
        # Create category counts from filtered data
        category_counts = df_filtered['category'].value_counts().to_dict()
        
        # Create the final JSON structure
        output_data = {
            "_metadata": {
                "classification_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "input_file": worksheet_name,
                "total_trials": len(filtered_data),
                "category_counts": category_counts,
                "filtered_out": len(data) - len(filtered_data)
            },
            "trials": filtered_data
        }
        
        # Save to JSON file
        output_filename = f"{worksheet_name}_filtered.json"
        with open(output_filename, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        logger.info(f"Successfully saved filtered data to {output_filename}")
        logger.info(f"Total trials after filtering: {len(filtered_data)}")
        logger.info(f"Filtered out {len(data) - len(filtered_data)} records")
        logger.info("Category counts after filtering:")
        for category, count in category_counts.items():
            logger.info(f"  {category}: {count}")
            
        # Create new worksheet with filtered data
        new_worksheet_name = f"{worksheet_name}_filtered"
        logger.info(f"Creating new worksheet '{new_worksheet_name}' with filtered data...")
        
        # First, create the new worksheet
        if not client.create_worksheet(spreadsheet_id, new_worksheet_name):
            logger.error("Failed to create new worksheet.")
            return
            
        # Add metadata rows
        metadata_rows = [
            ["Filtered Data Summary"],
            ["Classification Timestamp:", output_data["_metadata"]["classification_timestamp"]],
            ["Original Input File:", output_data["_metadata"]["input_file"]],
            ["Total Trials:", str(output_data["_metadata"]["total_trials"])],
            ["Filtered Out:", str(output_data["_metadata"]["filtered_out"])],
            [],
            ["Category Counts:"],
            ["Category", "Count"]
        ]
        
        # Add category counts
        for category, count in category_counts.items():
            metadata_rows.append([category, str(count)])
            
        metadata_rows.append([])  # Empty row for spacing
        
        # Add headers
        headers = list(filtered_data[0].keys()) if filtered_data else []
        metadata_rows.append(headers)
        
        # Add data rows
        data_rows = []
        for trial in filtered_data:
            row = [str(trial.get(header, '')) for header in headers]
            data_rows.append(row)
            
        # Combine all rows
        all_rows = metadata_rows + data_rows
        
        # Write to the new worksheet
        if client.export_trial_data(all_rows, spreadsheet_id, new_worksheet_name):
            logger.info(f"Successfully exported filtered data to worksheet '{new_worksheet_name}'")
        else:
            logger.error(f"Failed to export filtered data to worksheet '{new_worksheet_name}'")
        
    except Exception as e:
        logger.error(f"An error occurred while reading the worksheet: {e}")

if __name__ == "__main__":
    main() 