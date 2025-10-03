"""
This module provides the GoogleSheetsClient class for interacting with Google Sheets API.
It allows both reading from and writing to Google Spreadsheets.
"""

import os
import time
import json
import logging
import datetime
import random # Added for retry backoff
from typing import List, Dict, Optional, Callable, Any

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
import google.auth.exceptions
from googleapiclient.errors import HttpError

from config import GOOGLE_SHEETS_CONFIG

logger = logging.getLogger(__name__)

class GoogleSheetsClient:
    """
    A class to handle both reading from and writing to Google Sheets.
    It includes retry logic with exponential backoff for API requests.
    
    The class provides two main types of functionality:
    1. Reading operations:
       - read_sheet_data: Read data from a worksheet
    
    2. Writing operations:
       - export_classification_results: Export classification data
       - export_trial_data: Export trial data
       - create_summary_sheet: Create a summary sheet
       - create_worksheet: Create a new worksheet
       - create_new_spreadsheet: Create a new spreadsheet
       - clear_worksheet: Clear worksheet contents
    """

    def __init__(self, credentials_path: str = GOOGLE_SHEETS_CONFIG["credentials_file"]):
        """
        Initializes the GoogleSheetsClient.

        Args:
            credentials_path (str): Path to the service account credentials JSON file.
                                    Defaults to the path specified in GOOGLE_SHEETS_CONFIG.
        """
        self.credentials = None
        self.service = None
        try:
            self.credentials = Credentials.from_service_account_file(
                credentials_path,
                scopes=GOOGLE_SHEETS_CONFIG["scopes"]
            )
            self.service = build('sheets', 'v4', credentials=self.credentials)
            logger.info("Successfully authenticated with Google Sheets API.")
        except FileNotFoundError:
            logger.error(f"Credentials file not found at {credentials_path}. "
                         "Please ensure the file exists and the path is correct.")
        except google.auth.exceptions.GoogleAuthError as e:
            logger.error(f"Google authentication failed: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during authentication: {e}")

    def _make_api_request_with_retry(self, api_call: Callable[[], Any], method_name: str) -> Optional[Any]:
        """
        Executes an API call with retry logic and exponential backoff.

        Args:
            api_call (Callable[[], Any]): The function that makes the API call (e.g., lambda: self.service.spreadsheets().get(...).execute()).
            method_name (str): The name of the public method calling this helper, for logging.

        Returns:
            Optional[Any]: The API response if successful, None otherwise.
        """
        if not self.service:
            logger.error(f"Google Sheets API service is not initialized. Cannot make API call for {method_name}.")
            return None

        for attempt in range(GOOGLE_SHEETS_CONFIG.get("retry_attempts", 3)):
            try:
                return api_call()
            except HttpError as e:
                logger.error(f"API error on attempt {attempt + 1} for {method_name}: {e.resp.status} - {e.content}")
                if e.resp.status == 429: # Rate limiting
                    sleep_time = (2 ** attempt) + random.random()
                    logger.info(f"Rate limit exceeded for {method_name}. Retrying in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                elif 500 <= e.resp.status < 600: # Server-side errors
                    sleep_time = (2 ** attempt) + random.random()
                    logger.info(f"Server error ({e.resp.status}) for {method_name}. Retrying in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                else: # Other client-side HttpErrors
                    logger.error(f"Non-retriable HttpError for {method_name}: {e}. Not retrying.")
                    return None # Or re-raise e if preferred
            except Exception as e:
                logger.error(f"An unexpected error occurred on attempt {attempt + 1} for {method_name}: {e}")
                # Depending on policy, might want to retry some unexpected errors too
                # For now, fail on first unexpected error after HttpError specific handling.
                return None # Or re-raise e
        
        logger.error(f"All {GOOGLE_SHEETS_CONFIG.get('retry_attempts', 3)} retry attempts failed for {method_name}.")
        return None

    def _get_sheet_id(self, spreadsheet_id: str, worksheet_name: str) -> Optional[int]:
        """
        Retrieves the ID of a worksheet given its name.

        Args:
            spreadsheet_id (str): The ID of the Google Spreadsheet.
            worksheet_name (str): The name/title of the worksheet.

        Returns:
            Optional[int]: The sheet ID if found, else None.
        """
        method_name = "_get_sheet_id"
        api_call = lambda: self.service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
        response = self._make_api_request_with_retry(api_call, method_name)

        if response:
            for sheet in response.get('sheets', []):
                if sheet.get('properties', {}).get('title') == worksheet_name:
                    return sheet.get('properties', {}).get('sheetId')
            logger.warning(f"Sheet '{worksheet_name}' not found in spreadsheet '{spreadsheet_id}'.")
        else:
            logger.error(f"Failed to get spreadsheet details for '{spreadsheet_id}' to find sheet '{worksheet_name}'.")
        return None

    def _apply_formatting_requests(self, spreadsheet_id: str, requests: List[Dict]) -> bool:
        """
        Applies a list of formatting requests to a spreadsheet.

        Args:
            spreadsheet_id (str): The ID of the Google Spreadsheet.
            requests (List[Dict]): A list of Google Sheets API Request objects.

        Returns:
            bool: True if formatting was successful, False otherwise.
        """
        method_name = "_apply_formatting_requests"
        if not requests:
            logger.info(f"No formatting requests to apply for spreadsheet '{spreadsheet_id}'.")
            return True 

        body = {'requests': requests}
        api_call = lambda: self.service.spreadsheets().batchUpdate(spreadsheetId=spreadsheet_id, body=body).execute()
        response = self._make_api_request_with_retry(api_call, method_name)

        if response:
            logger.info(f"Successfully applied {len(requests)} formatting requests to spreadsheet '{spreadsheet_id}'.")
            return True
        else:
            logger.error(f"Failed to apply formatting requests to spreadsheet '{spreadsheet_id}'.")
            return False

    def create_new_spreadsheet(self, title: str) -> Optional[str]:
        """
        Creates a new Google Spreadsheet with retry logic.

        Args:
            title (str): The title for the new spreadsheet.

        Returns:
            Optional[str]: The ID of the newly created spreadsheet, or None if creation failed.
        """
        spreadsheet_body = {
            'properties': {
                'title': title
            }
        }
        
        api_call = lambda: self.service.spreadsheets().create(body=spreadsheet_body, fields='spreadsheetId').execute()
        response = self._make_api_request_with_retry(api_call, "create_new_spreadsheet")

        if response:
            spreadsheet_id = response.get('spreadsheetId')
            logger.info(f"Successfully created new spreadsheet with ID: {spreadsheet_id} and Title: {title}")
            return spreadsheet_id
        else:
            logger.error(f"Failed to create spreadsheet '{title}' after retries.")
            return None

    def create_worksheet(self, spreadsheet_id: str, worksheet_name: str) -> bool:
        """
        Creates a new worksheet (tab) in an existing Google Spreadsheet.

        Args:
            spreadsheet_id (str): The ID of the target Google Spreadsheet.
            worksheet_name (str): The title for the new worksheet.

        Returns:
            bool: True if worksheet creation was successful, False otherwise.
        """
        method_name = "create_worksheet"
        requests = [{
            'addSheet': {
                'properties': {
                    'title': worksheet_name
                }
            }
        }]

        body = {'requests': requests}
        api_call = lambda: self.service.spreadsheets().batchUpdate(spreadsheetId=spreadsheet_id, body=body).execute()
        response = self._make_api_request_with_retry(api_call, method_name)

        if response:
            logger.info(f"Successfully created worksheet '{worksheet_name}' in spreadsheet '{spreadsheet_id}'.")
            return True
        else:
            logger.error(f"Failed to create worksheet '{worksheet_name}' in spreadsheet '{spreadsheet_id}'.")
            return False

    def export_classification_results(self, results: List[Dict], spreadsheet_id: str, worksheet_name: str = "Classifications") -> bool:
        """
        Exports classification results to a specified worksheet with retry logic and batching.

        Args:
            results (List[Dict]): A list of dictionaries, where each dictionary
                                  represents a classification result.
                                  Expected keys: 'nct_id', 'brief_title', 'category', 'classification'.
            spreadsheet_id (str): The ID of the Google Spreadsheet.
            worksheet_name (str): The name of the worksheet to export data to.
                                  Defaults to "Classifications".

        Returns:
            bool: True if export was successful, False otherwise.
        """
        if not results:
            logger.warning(f"No classification results provided for spreadsheet '{spreadsheet_id}', worksheet '{worksheet_name}'. Skipping export.")
            return True
        
        if not isinstance(results, list) or not all(isinstance(item, dict) for item in results):
            logger.error("Invalid format for 'results'. Expected a list of dictionaries.")
            return False
            
        if results: # Basic validation of first item
            first_item = results[0]
            if 'nct_id' not in first_item or 'classification' not in first_item:
                logger.warning("First item in 'results' may be malformed (missing 'nct_id' or 'classification'). Proceeding with export.")

        headers = ["NCT ID", "Brief Title", "Category", "Confidence", "Rationale", "Key Indicators", "Export Date"]
        
        # Prepare metadata rows
        export_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata_rows_content = [
            ["Export Timestamp:", export_time_str],
            ["Total Results:", len(results)],
            [] # Empty row for spacing
        ]
        
        data_rows = []
        # Note: export_timestamp for each row is now part of the main data preparation
        # The overall export_time_str is for the metadata section.
        
        for result in results:
            classification_text = result.get('classification', '')
            rationale = "N/A"
            key_indicators = "N/A"
            confidence = "N/A"
            row_export_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") # Per-row timestamp

            for line in classification_text.splitlines():
                if line.startswith("RATIONALE:"):
                    rationale = line.replace("RATIONALE:", "").strip()
                elif line.startswith("KEY INDICATORS:"):
                    key_indicators = line.replace("KEY INDICATORS:", "").strip()
                elif line.startswith("CONFIDENCE LEVEL:"):
                    confidence = line.replace("CONFIDENCE LEVEL:", "").strip()
            
            row_data = [
                result.get('nct_id', 'N/A'),
                result.get('brief_title', 'N/A'),
                result.get('category', 'N/A'),
                confidence,
                rationale,
                key_indicators,
                row_export_timestamp # Using per-row timestamp here
            ]
            data_rows.append(row_data)

        # The actual header row index will be len(metadata_rows_content)
        header_row_index = len(metadata_rows_content) # This is the 0-based index where headers will be
        
        batch_size = GOOGLE_SHEETS_CONFIG.get("batch_size", 100) # Default batch size if not in config
        all_rows_to_append = metadata_rows_content + [headers] + data_rows
        
        num_batches = (len(all_rows_to_append) + batch_size -1) // batch_size # Ceiling division
        if not all_rows_to_append: # Should be caught by earlier check, but as a safeguard
            logger.info("No data to append (metadata + headers + data rows is empty).")
            return True

        overall_success = True
        for i in range(num_batches):
            batch_start_index = i * batch_size
            batch_end_index = min((i + 1) * batch_size, len(all_rows_to_append))
            current_batch_values = all_rows_to_append[batch_start_index:batch_end_index]

            if not current_batch_values: # Should not happen if num_batches is calculated correctly
                continue

            logger.info(f"Exporting batch {i+1} of {num_batches} ({len(current_batch_values)} rows) to worksheet '{worksheet_name}' in spreadsheet '{spreadsheet_id}'.")

            body = {
                'range': worksheet_name, # Append will find the next empty rows
                'values': current_batch_values
            }
            api_call = lambda: self.service.spreadsheets().values().append(
                spreadsheetId=spreadsheet_id,
                range=worksheet_name, 
                valueInputOption='USER_ENTERED',
                insertDataOption='INSERT_ROWS',
                body=body
            ).execute()

            response = self._make_api_request_with_retry(api_call, f"export_classification_results (batch {i+1}/{num_batches})")

            if not response:
                logger.error(f"Failed to export batch {i+1} of {num_batches} for classification results to spreadsheet '{spreadsheet_id}'.")
                overall_success = False
                break # Stop if a batch fails
        
        if overall_success:
            logger.info(f"Successfully exported all {num_batches} batches of classification results to spreadsheet ID '{spreadsheet_id}', worksheet '{worksheet_name}'.")
            sheet_id = self._get_sheet_id(spreadsheet_id, worksheet_name)
            if sheet_id is not None:
                formatting_requests_list = [
                    {'updateSheetProperties': {'properties': {'sheetId': sheet_id, 'gridProperties': {'frozenRowCount': header_row_index + 1}}, 'fields': 'gridProperties.frozenRowCount'}},
                    {'repeatCell': {'range': {'sheetId': sheet_id, 'startRowIndex': header_row_index, 'endRowIndex': header_row_index + 1}, 'cell': {'userEnteredFormat': {'textFormat': {'bold': True}}}, 'fields': 'userEnteredFormat.textFormat.bold'}}
                ]
                for col_idx in range(len(headers)): # Use col_idx to avoid confusion with batch index i
                    formatting_requests_list.append({'autoResizeDimensions': {'dimensions': {'sheetId': sheet_id, 'dimension': 'COLUMNS', 'startIndex': col_idx, 'endIndex': col_idx+1}}})
                
                # Specific formatting for metadata rows (e.g., bold labels in first column)
                # -1 to skip empty spacer row from metadata_rows_content for formatting
                for meta_row_idx in range(len(metadata_rows_content) -1): 
                     formatting_requests_list.append({'repeatCell': {'range': {'sheetId': sheet_id, 'startRowIndex': meta_row_idx, 'endRowIndex': meta_row_idx+1, 'startColumnIndex': 0, 'endColumnIndex': 1}, 'cell': {'userEnteredFormat': {'textFormat': {'bold': True}}}, 'fields': 'userEnteredFormat.textFormat.bold'}})

                self._apply_formatting_requests(spreadsheet_id, formatting_requests_list)
            else:
                logger.warning(f"Could not find sheetId for '{worksheet_name}' in spreadsheet '{spreadsheet_id}' to apply formatting.")
            return True
        else:
            logger.error(f"Failed to export all classification results to spreadsheet '{spreadsheet_id}' due to batch failure.")
            return False

    def export_trial_data(self, trials: List[Dict], spreadsheet_id: str, worksheet_name: str = "Raw Data") -> bool:
        """
        Exports raw trial data to a specified worksheet with retry logic and batching.

        Args:
            trials (List[Dict]): A list of dictionaries, where each dictionary
                                 represents a trial data point. Expected keys include
                                 'NCT ID', 'Brief Title', 'Official Title', 'Sponsor',
                                 'Phase'/'Phases', 'Conditions', 'Interventions', 'Summary'.
            spreadsheet_id (str): The ID of the Google Spreadsheet.
            worksheet_name (str): The name of the worksheet to export data to.
                                  Defaults to "Raw Data".

        Returns:
            bool: True if export was successful, False otherwise.
        """
        if not trials:
            logger.warning(f"No trial data provided for spreadsheet '{spreadsheet_id}', worksheet '{worksheet_name}'. Skipping export.")
            return True

        if not isinstance(trials, list) or not all(isinstance(item, dict) for item in trials):
            logger.error("Invalid format for 'trials'. Expected a list of dictionaries.")
            return False

        # Assuming the first dictionary in the list contains all relevant keys for headers
        # Or we could explicitly pass headers. For now, infer from first trial.
        if not trials:
            logger.error("Cannot infer headers from empty trials list.")
            return False

        headers = list(trials[0].keys())
        data_rows = []

        for trial in trials:
            row = [str(trial.get(header, '')) for header in headers]
            data_rows.append(row)

        batch_size = GOOGLE_SHEETS_CONFIG.get("batch_size", 100)
        all_rows_to_append = [headers] + data_rows # Add headers as the first row

        num_batches = (len(all_rows_to_append) + batch_size -1) // batch_size
        if not all_rows_to_append:
             logger.info("No data to append (headers + data rows is empty).") # Should be caught earlier but as a safeguard
             return True

        overall_success = True
        for i in range(num_batches):
            batch_start_index = i * batch_size
            batch_end_index = min((i + 1) * batch_size, len(all_rows_to_append))
            current_batch_values = all_rows_to_append[batch_start_index:batch_end_index]

            if not current_batch_values:
                continue
            
            logger.info(f"Exporting batch {i+1} of {num_batches} ({len(current_batch_values)} rows) to worksheet '{worksheet_name}' in spreadsheet '{spreadsheet_id}'.")

            body = {
                'range': worksheet_name, # Append will find the next empty rows
                'values': current_batch_values
            }
            api_call = lambda: self.service.spreadsheets().values().append(
                spreadsheetId=spreadsheet_id,
                range=worksheet_name, 
                valueInputOption='USER_ENTERED',
                insertDataOption='INSERT_ROWS',
                body=body
            ).execute()
            
            response = self._make_api_request_with_retry(api_call, f"export_trial_data (batch {i+1}/{num_batches})")

            if not response:
                logger.error(f"Failed to export batch {i+1} of {num_batches} for trial data to spreadsheet '{spreadsheet_id}'.")
                overall_success = False
                break
        
        if overall_success:
            logger.info(f"Successfully exported all {num_batches} batches of trial data to spreadsheet ID '{spreadsheet_id}', worksheet '{worksheet_name}'.")
            sheet_id = self._get_sheet_id(spreadsheet_id, worksheet_name)
            if sheet_id is not None:
                header_row_idx = 0 # Standard header at the top
                formatting_requests_list = [
                    {'updateSheetProperties': {'properties': {'sheetId': sheet_id, 'gridProperties': {'frozenRowCount': header_row_idx + 1}}, 'fields': 'gridProperties.frozenRowCount'}},
                    {'repeatCell': {'range': {'sheetId': sheet_id, 'startRowIndex': header_row_idx, 'endRowIndex': header_row_idx + 1}, 'cell': {'userEnteredFormat': {'textFormat': {'bold': True}}}, 'fields': 'userEnteredFormat.textFormat.bold'}}
                ]
                for col_idx in range(len(headers)):
                    formatting_requests_list.append({'autoResizeDimensions': {'dimensions': {'sheetId': sheet_id, 'dimension': 'COLUMNS', 'startIndex': col_idx, 'endIndex': col_idx+1}}})
                self._apply_formatting_requests(spreadsheet_id, formatting_requests_list)
            else:
                logger.warning(f"Could not find sheetId for '{worksheet_name}' in spreadsheet '{spreadsheet_id}' to apply formatting.")
            return True
        else:
            logger.error(f"Failed to export all trial data to spreadsheet '{spreadsheet_id}' due to batch failure.")
            return False

    def create_summary_sheet(self, results: List[Dict], spreadsheet_id: str, worksheet_name: str = "Summary") -> bool:
        """
        Creates a summary sheet from classification results with retry logic.

        Args:
            results (List[Dict]): A list of dictionaries, where each dictionary
                                  represents a classification result. Expected keys:
                                  'category', 'classification' (string containing "CONFIDENCE LEVEL: ...").
            spreadsheet_id (str): The ID of the Google Spreadsheet.
            worksheet_name (str): The name of the worksheet to create the summary in.
                                  Defaults to "Summary".

        Returns:
            bool: True if summary sheet creation was successful, False otherwise.
        """
        headers = ["Category", "Count", "Percentage", "High Confidence", "Medium Confidence", "Low Confidence"]
        summary_data = {} # To store counts for each category
        total_classified_trials = 0

        for result in results:
            category = result.get('category')
            if not category or category == "Not eligible": # Assuming "Not eligible" should not be in summary
                continue
            
            total_classified_trials += 1
            classification_text = result.get('classification', '')
            confidence = "Unknown" # Default if not found or not matching

            for line in classification_text.splitlines():
                if line.startswith("CONFIDENCE LEVEL:"):
                    parsed_confidence = line.replace("CONFIDENCE LEVEL:", "").strip().lower()
                    if parsed_confidence == "high":
                        confidence = "High"
                    elif parsed_confidence == "medium":
                        confidence = "Medium"
                    elif parsed_confidence == "low":
                        confidence = "Low"
                    break # Found confidence, no need to parse further lines for it

            if category not in summary_data:
                summary_data[category] = {
                    "Count": 0,
                    "High Confidence": 0,
                    "Medium Confidence": 0,
                    "Low Confidence": 0,
                    "Unknown Confidence": 0 # For internal counting if needed
                }
            
            summary_data[category]["Count"] += 1
            if confidence == "High":
                summary_data[category]["High Confidence"] += 1
            elif confidence == "Medium":
                summary_data[category]["Medium Confidence"] += 1
            elif confidence == "Low":
                summary_data[category]["Low Confidence"] += 1
            else: # Includes "Unknown"
                summary_data[category]["Unknown Confidence"] += 1

        data_rows = []
        if total_classified_trials == 0: # Avoid division by zero
            logger.warning("No classified trials found to create a summary. Skipping summary sheet creation.")
            # Still might want to create an empty sheet with headers, or return True/False based on requirements.
            # For now, let's proceed to create it with headers and potentially empty data.
            pass


        for category, counts in summary_data.items():
            percentage = (counts["Count"] / total_classified_trials * 100) if total_classified_trials > 0 else 0
            row = [
                category,
                counts["Count"],
                f"{percentage:.2f}%",
                counts["High Confidence"],
                counts["Medium Confidence"],
                counts["Low Confidence"]
            ]
            # If Unknown Confidence counts are significant, one might want to add a column for them.
            # For now, it's implicitly handled by not being in High/Medium/Low.
            data_rows.append(row)
        
        # Sort by count descending for better readability
        data_rows.sort(key=lambda x: x[1], reverse=True)

        values_to_append = [headers] + data_rows

        body = {
            'range': worksheet_name,
            'valueInputOption': 'USER_ENTERED',
            'insertDataOption': 'INSERT_ROWS',
            'values': values_to_append
        }

        api_call = lambda: self.service.spreadsheets().values().append(
            spreadsheetId=spreadsheet_id,
            range=worksheet_name,
            valueInputOption='USER_ENTERED',
            insertDataOption='INSERT_ROWS',
            body=body
        ).execute()

        response = self._make_api_request_with_retry(api_call, "create_summary_sheet")

        if response:
            logger.info(f"Successfully created summary sheet in spreadsheet ID '{spreadsheet_id}', worksheet '{worksheet_name}'.")
            sheet_id = self._get_sheet_id(spreadsheet_id, worksheet_name)
            if sheet_id is not None:
                header_row_index = 0 # Standard header at the top
                formatting_requests = [
                    {'updateSheetProperties': {'properties': {'sheetId': sheet_id, 'gridProperties': {'frozenRowCount': header_row_index + 1}}, 'fields': 'gridProperties.frozenRowCount'}},
                    {'repeatCell': {'range': {'sheetId': sheet_id, 'startRowIndex': header_row_index, 'endRowIndex': header_row_index + 1}, 'cell': {'userEnteredFormat': {'textFormat': {'bold': True}}}, 'fields': 'userEnteredFormat.textFormat.bold'}}
                ]
                for i in range(len(headers)): # headers variable is defined within this method's scope
                    formatting_requests.append({'autoResizeDimensions': {'dimensions': {'sheetId': sheet_id, 'dimension': 'COLUMNS', 'startIndex': i, 'endIndex': i+1}}})
                self._apply_formatting_requests(spreadsheet_id, formatting_requests)
            else:
                 logger.warning(f"Could not find sheetId for '{worksheet_name}' in spreadsheet '{spreadsheet_id}' to apply formatting.")
            return True
        else:
            logger.error(f"Failed to create summary sheet in spreadsheet '{spreadsheet_id}' after retries.")
            return False

    def clear_worksheet(self, spreadsheet_id: str, worksheet_name: str) -> bool:
        """
        Clears all contents of a worksheet by name.

        Args:
            spreadsheet_id (str): The ID of the Google Spreadsheet.
            worksheet_name (str): The name/title of the worksheet to clear.

        Returns:
            bool: True if the worksheet was cleared successfully, False otherwise.
        """
        method_name = "clear_worksheet"
        range_to_clear = worksheet_name  # Clear the whole sheet
        api_call = lambda: self.service.spreadsheets().values().clear(
            spreadsheetId=spreadsheet_id,
            range=range_to_clear,
            body={}
        ).execute()
        response = self._make_api_request_with_retry(api_call, method_name)
        if response:
            logger.info(f"Successfully cleared worksheet '{worksheet_name}' in spreadsheet '{spreadsheet_id}'.")
            return True
        else:
            logger.error(f"Failed to clear worksheet '{worksheet_name}' in spreadsheet '{spreadsheet_id}'.")
            return False

    def read_sheet_data(self, spreadsheet_id: str, worksheet_name: str, range_name: Optional[str] = None) -> Optional[List[Dict]]:
        """
        Reads data from a Google Sheet and returns it as a list of dictionaries.
        Each dictionary represents a row, with column headers as keys.

        Args:
            spreadsheet_id (str): The ID of the Google Spreadsheet.
            worksheet_name (str): The name of the worksheet to read from.
            range_name (Optional[str]): The range to read (e.g., 'A1:D10'). If None, reads the entire sheet.

        Returns:
            Optional[List[Dict]]: A list of dictionaries containing the sheet data, where each dictionary
                                 represents a row with column headers as keys. Returns None if the read fails.
        """
        method_name = "read_sheet_data"
        
        # Construct the range string
        range_str = f"{worksheet_name}"
        if range_name:
            range_str += f"!{range_name}"
        
        # Make the API call to get the values
        api_call = lambda: self.service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range=range_str
        ).execute()
        
        response = self._make_api_request_with_retry(api_call, method_name)
        
        if not response or 'values' not in response:
            logger.error(f"Failed to read data from spreadsheet '{spreadsheet_id}', worksheet '{worksheet_name}'")
            return None
            
        values = response.get('values', [])
        if not values:
            logger.info(f"No data found in spreadsheet '{spreadsheet_id}', worksheet '{worksheet_name}'")
            return []
            
        # Get headers from the first row
        headers = values[0]
        
        # Convert remaining rows to dictionaries
        result = []
        for row in values[1:]:
            # Pad row with empty strings if it's shorter than headers
            padded_row = row + [''] * (len(headers) - len(row))
            # Create dictionary with headers as keys
            row_dict = dict(zip(headers, padded_row))
            result.append(row_dict)
            
        logger.info(f"Successfully read {len(result)} rows from spreadsheet '{spreadsheet_id}', worksheet '{worksheet_name}'")
        return result
