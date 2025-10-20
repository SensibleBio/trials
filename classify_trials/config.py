"""
This file contains configuration settings for accessing Google Sheets API.

Attributes:
    GOOGLE_SHEETS_CONFIG (dict): A dictionary containing configuration parameters
        for Google Sheets API integration.
        - "scopes" (list): A list of authorization scopes required for API access.
        - "credentials_file" (str): The name of the JSON file containing API credentials.
                                      This file needs to be created by the user.
        - "default_spreadsheet_id" (str | None): The default Google Spreadsheet ID to use.
                                                 Can be None if not set.
        - "batch_size" (int): The number of rows/columns to process in a single API batch request.
        - "retry_attempts" (int): The number of times to retry a failed API request.
"""

GOOGLE_SHEETS_CONFIG = {
    "scopes": ["https://www.googleapis.com/auth/spreadsheets"],
    "credentials_file": "credentials.json",  # This file will need to be created by the user
    "default_spreadsheet_id": "1HpyjKdt_n929Pmca0rAyDt1mrw2YDExjoO9CGgIuCso",
    "batch_size": 100,
    "retry_attempts": 3
}
