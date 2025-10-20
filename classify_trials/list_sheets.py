from .google_sheets_client import GoogleSheetsClient
import logging

# Configure logging to see exporter messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    client = GoogleSheetsClient()
    spreadsheet_id = "1HpyjKdt_n929Pmca0rAyDt1mrw2YDExjoO9CGgIuCso"
    if not client.service:
        logger.error("Failed to initialize Google Sheets Client.")
        return
    method_name = "list_sheets"
    api_call = lambda: client.service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    response = client._make_api_request_with_retry(api_call, method_name)
    if response and 'sheets' in response:
        sheet_titles = [sheet.get('properties', {}).get('title') for sheet in response['sheets']]
        logger.info(f"Worksheets in spreadsheet: {sheet_titles}")
    else:
        logger.error("Failed to retrieve worksheet list.")

if __name__ == "__main__":
    main() 