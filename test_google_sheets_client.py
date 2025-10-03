"""
Tests for the GoogleSheetsClient class.
"""

import unittest
from unittest.mock import patch, MagicMock, PropertyMock, call
import datetime
import json # For HttpError content if needed

# Import exceptions and the class to be tested
import google.auth.exceptions
from googleapiclient.errors import HttpError 
from google_sheets_client import GoogleSheetsClient
from config import GOOGLE_SHEETS_CONFIG # Will be patched

# Helper to create HttpError instances
def create_http_error(status_code: int, reason: str = "Error"):
    resp = MagicMock()
    resp.status = status_code
    resp.reason = reason
    # HttpError expects content to be bytes
    return HttpError(resp=resp, content=bytes(reason, 'utf-8'))

class TestGoogleSheetsClient(unittest.TestCase):
    """
    Test cases for the GoogleSheetsClient class.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = GOOGLE_SHEETS_CONFIG.copy()
        # Patch GOOGLE_SHEETS_CONFIG for all tests in this class
        self.mock_config_patch = patch('google_sheets_client.GOOGLE_SHEETS_CONFIG', new_callable=dict)
        self.mock_config = self.mock_config_patch.start()
        self.mock_config.update({
            "scopes": ["https://www.googleapis.com/auth/spreadsheets"],
            "credentials_file": "dummy_creds.json",
            "default_spreadsheet_id": "default_test_id",
            "batch_size": 100, # Can be overridden in specific tests
            "retry_attempts": 3   # Can be overridden
        })

        # It's often easier to work with a fresh instance per test or per group of tests
        # self.exporter = GoogleSheetsClient(credentials_path="dummy_creds.json")
        # However, for __init__ tests, we create it inside the test method

    def tearDown(self):
        """Clean up after each test method."""
        self.mock_config_patch.stop()

    @patch('google_sheets_client.Credentials.from_service_account_file')
    @patch('google_sheets_client.build')
    def test_init_successful(self, mock_build, mock_from_service_account_file):
        """Test successful initialization of GoogleSheetsClient."""
        mock_creds = MagicMock()
        mock_from_service_account_file.return_value = mock_creds
        mock_service = MagicMock()
        mock_build.return_value = mock_service

        client = GoogleSheetsClient(credentials_path="dummy_creds.json")

        mock_from_service_account_file.assert_called_once_with(
            "dummy_creds.json",
            scopes=self.mock_config["scopes"]
        )
        mock_build.assert_called_once_with('sheets', 'v4', credentials=mock_creds)
        self.assertEqual(client.credentials, mock_creds)
        self.assertEqual(client.service, mock_service)

    @patch('google_sheets_client.Credentials.from_service_account_file')
    def test_init_failure_credentials_file_not_found(self, mock_from_service_account_file):
        """Test initialization failure when credentials file is not found."""
        mock_from_service_account_file.side_effect = FileNotFoundError("File not found")
        
        with patch('google_sheets_client.logger.error') as mock_logger_error:
            client = GoogleSheetsClient(credentials_path="non_existent.json")
            self.assertIsNone(client.service)
            self.assertIsNone(client.credentials)
            mock_logger_error.assert_called_with("Credentials file not found at non_existent.json. Please ensure the file exists and the path is correct.")

    @patch('google_sheets_client.Credentials.from_service_account_file')
    def test_init_failure_google_auth_error(self, mock_from_service_account_file):
        """Test initialization failure due to GoogleAuthError."""
        mock_from_service_account_file.side_effect = google.auth.exceptions.GoogleAuthError("Auth error")

        with patch('google_sheets_client.logger.error') as mock_logger_error:
            client = GoogleSheetsClient(credentials_path="dummy_creds.json")
            self.assertIsNone(client.service)
            self.assertIsNone(client.credentials) # Credentials might be set before exception, or not, depends on impl.
                                                  # Current impl sets self.credentials before self.service
                                                  # Let's assume if service is None, creds might also be an issue or None
            mock_logger_error.assert_called_with("Google authentication failed: Auth error")

    # --- Tests for create_new_spreadsheet ---
    @patch('google_sheets_client.Credentials.from_service_account_file') # To allow init
    @patch('google_sheets_client.build') # To allow init
    def test_create_new_spreadsheet_success(self, mock_build, mock_creds_init):
        """Test successful creation of a new spreadsheet."""
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        
        client = GoogleSheetsClient() # Uses dummy_creds.json from setUp mock_config
        client.service = mock_service # Ensure service is set

        expected_response = {'spreadsheetId': 'test_spreadsheet_id_123'}
        mock_service.spreadsheets().create().execute.return_value = expected_response
        
        # Mock _make_api_request_with_retry to directly return the successful call's result
        with patch.object(client, '_make_api_request_with_retry', side_effect=lambda func, name: func()) as mock_retry_helper:
            spreadsheet_id = client.create_new_spreadsheet(title="Test Sheet")
            self.assertEqual(spreadsheet_id, 'test_spreadsheet_id_123')
            mock_service.spreadsheets().create.assert_called_once_with(
                body={'properties': {'title': "Test Sheet"}},
                fields='spreadsheetId'
            )
            mock_retry_helper.assert_called_once()


    @patch('google_sheets_client.Credentials.from_service_account_file')
    @patch('google_sheets_client.build')
    def test_create_new_spreadsheet_api_error(self, mock_build, mock_creds_init):
        """Test spreadsheet creation failure due to an API error."""
        mock_service = MagicMock()
        mock_build.return_value = mock_service

        client = GoogleSheetsClient()
        client.service = mock_service

        # Simulate an HttpError from the API call
        # The _make_api_request_with_retry should handle this and return None
        mock_service.spreadsheets().create().execute.side_effect = create_http_error(500, "Server Error")

        with patch('google_sheets_client.logger.error') as mock_logger_error:
            # Patch _make_api_request_with_retry to reflect its actual behavior of returning None on error
            with patch.object(client, '_make_api_request_with_retry', return_value=None) as mock_retry_helper_returns_none:
                spreadsheet_id = client.create_new_spreadsheet(title="Error Sheet")
                self.assertIsNone(spreadsheet_id)
                # We expect _make_api_request_with_retry to be called, and it will log the error internally
                mock_retry_helper_returns_none.assert_called_once()
                # Check that the calling method also logs an error
                mock_logger_error.assert_any_call("Failed to create spreadsheet 'Error Sheet' after retries.")


    @patch('google_sheets_client.Credentials.from_service_account_file')
    @patch('google_sheets_client.build')
    def test_create_new_spreadsheet_service_none(self, mock_build, mock_creds_init):
        """Test create_new_spreadsheet when self.service is None."""
        # This simulates auth failure during __init__
        mock_build.return_value = None # Ensure service is None initially
        
        client = GoogleSheetsClient()
        client.service = None # Explicitly ensure service is None

        # _make_api_request_with_retry should handle service being None first
        with patch('google_sheets_client.logger.error') as mock_logger_error:
             with patch.object(client, '_make_api_request_with_retry', return_value=None) as mock_retry_helper:
                spreadsheet_id = client.create_new_spreadsheet(title="No Service Sheet")
                self.assertIsNone(spreadsheet_id)
                mock_retry_helper.assert_called_once() # Retry helper is called, sees service is None
                # The logger call comes from _make_api_request_with_retry
                # And then from create_new_spreadsheet itself
                mock_logger_error.assert_any_call("Google Sheets API service is not initialized. Cannot make API call for create_new_spreadsheet.")
                mock_logger_error.assert_any_call("Failed to create spreadsheet 'No Service Sheet' after retries.")

    # --- Tests for export_classification_results ---
    @patch('google_sheets_client.datetime') # To mock datetime.now()
    @patch.object(GoogleSheetsClient, '_get_sheet_id', return_value=12345) # Mock get_sheet_id
    @patch.object(GoogleSheetsClient, '_apply_formatting_requests', return_value=True) # Mock formatting
    @patch('google_sheets_client.Credentials.from_service_account_file') # To allow init
    @patch('google_sheets_client.build') # To allow init
    def test_export_classification_results_success_single_batch(
        self, mock_build, mock_creds_init, mock_apply_formatting, mock_get_sheet_id, mock_datetime
    ):
        """Test successful export of classification results in a single batch."""
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        client = GoogleSheetsClient()
        client.service = mock_service

        # Mock datetime.now()
        mock_now = MagicMock()
        mock_now.strftime.return_value = "2023-01-01 12:00:00"
        mock_datetime.datetime.now.return_value = mock_now
        
        sample_results = [
            {'nct_id': 'NCT001', 'brief_title': 'Title 1', 'category': 'Cancer', 'classification': 'CONFIDENCE LEVEL: High\nRATIONALE: Test Rationale 1\nKEY INDICATORS: KI1'},
            {'nct_id': 'NCT002', 'brief_title': 'Title 2', 'category': 'Vaccine', 'classification': 'CONFIDENCE LEVEL: Medium\nRATIONALE: Test Rationale 2'}, # Missing KI
        ]

        expected_metadata_rows = [
            ["Export Timestamp:", "2023-01-01 12:00:00"],
            ["Total Results:", 2],
            []
        ]
        expected_header_row = ["NCT ID", "Brief Title", "Category", "Confidence", "Rationale", "Key Indicators", "Export Date"]
        expected_data_rows = [
            ['NCT001', 'Title 1', 'Cancer', 'High', 'Test Rationale 1', 'KI1', "2023-01-01 12:00:00"],
            ['NCT002', 'Title 2', 'Vaccine', 'Medium', 'Test Rationale 2', 'N/A', "2023-01-01 12:00:00"],
        ]
        expected_values_to_append = expected_metadata_rows + [expected_header_row] + expected_data_rows

        # Mock the append call response
        mock_service.spreadsheets().values().append().execute.return_value = {'updates': {'updatedRange': 'Sheet1!A1'}}

        with patch.object(client, '_make_api_request_with_retry', side_effect=lambda func, name: func()) as mock_retry_helper:
            success = client.export_classification_results(sample_results, "test_spreadsheet_id", "ClassificationsSheet")
            self.assertTrue(success)
            
            # Check that append was called correctly
            # The body is constructed inside the method, so we check the final call to execute()
            actual_call_args = mock_service.spreadsheets().values().append().execute.call_args
            self.assertIsNotNone(actual_call_args) # Ensure it was called

            # Verify call to _make_api_request_with_retry for the append operation
            # The lambda func passed to _make_api_request_with_retry is what we want to inspect indirectly
            # by checking the .execute() call that it would trigger
            
            # Check the 'values' in the body of the append call
            # This requires access to the 'body' argument of the 'append' method call
            # The structure is service.spreadsheets().values().append(spreadsheetId=..., range=..., ..., body=...).execute()
            # We need to look at the args of append() itself.
            append_args, append_kwargs = mock_service.spreadsheets().values().append.call_args
            self.assertIn('body', append_kwargs)
            self.assertEqual(append_kwargs['body']['values'], expected_values_to_append)
            self.assertEqual(append_kwargs['spreadsheetId'], "test_spreadsheet_id")
            self.assertEqual(append_kwargs['range'], "ClassificationsSheet") # also specified in body, but good to check here
            self.assertEqual(append_kwargs['valueInputOption'], "USER_ENTERED")
            self.assertEqual(append_kwargs['insertDataOption'], "INSERT_ROWS")

            mock_get_sheet_id.assert_called_once_with("test_spreadsheet_id", "ClassificationsSheet")
            mock_apply_formatting.assert_called_once()
            # More detailed check for formatting requests can be added if necessary

    @patch('google_sheets_client.datetime')
    @patch.object(GoogleSheetsClient, '_get_sheet_id', return_value=12345)
    @patch.object(GoogleSheetsClient, '_apply_formatting_requests', return_value=True)
    @patch('google_sheets_client.Credentials.from_service_account_file')
    @patch('google_sheets_client.build')
    def test_export_classification_results_batching(
        self, mock_build, mock_creds_init, mock_apply_formatting, mock_get_sheet_id, mock_datetime
    ):
        """Test export of classification results with batching."""
        # Override batch_size for this test
        self.mock_config['batch_size'] = 3 # metadata(3) + header(1) + 2 data rows = 6. This will cause batching.
                                         # Let's set batch_size to 4 to be clearer: metadata+header = 4. So data splits.
        self.mock_config['batch_size'] = 4


        mock_service = MagicMock()
        mock_build.return_value = mock_service
        client = GoogleSheetsClient()
        client.service = mock_service
        
        mock_now = MagicMock()
        mock_now.strftime.return_value = "2023-01-01 12:00:00"
        mock_datetime.datetime.now.return_value = mock_now

        sample_results = [
            {'nct_id': 'NCT001', 'brief_title': 'T1', 'category': 'C1', 'classification': 'CONFIDENCE LEVEL: High'},
            {'nct_id': 'NCT002', 'brief_title': 'T2', 'category': 'C2', 'classification': 'CONFIDENCE LEVEL: Med'},
            {'nct_id': 'NCT003', 'brief_title': 'T3', 'category': 'C3', 'classification': 'CONFIDENCE LEVEL: Low'},
        ]
        
        # Expected calls to append().execute()
        # Batch 1: Meta (3) + Header (1) = 4 rows.
        # Batch 2: Data row 1 (NCT001)
        # Batch 3: Data row 2 (NCT002)
        # Batch 4: Data row 3 (NCT003)
        # This seems too granular if batch_size is 4.
        # all_rows_to_append = metadata_rows_content (3) + [headers] (1) + data_rows (3) = 7 rows
        # Batch 1: metadata (3) + header (1) = 4 rows from all_rows_to_append[0:4]
        # Batch 2: data_rows[0], data_rows[1], data_rows[2] = 3 rows from all_rows_to_append[4:7]

        mock_service.spreadsheets().values().append().execute.return_value = {'updates': {'updatedRange': 'Sheet1!A1'}}

        # We need _make_api_request_with_retry to be called for each batch
        # Use a list to store the bodies of each call to append
        actual_append_bodies = []
        def capture_append_body(*args, **kwargs):
            actual_append_bodies.append(kwargs['body']['values'])
            return {'updates': {'updatedRange': 'Sheet1!A1'}} # Simulate success for each batch

        mock_service.spreadsheets().values().append().execute.side_effect = capture_append_body
        
        with patch.object(client, '_make_api_request_with_retry', side_effect=lambda func, name: func()) as mock_retry_helper:
            success = client.export_classification_results(sample_results, "test_spreadsheet_id_batch", "ClassificationsSheetBatch")
            self.assertTrue(success)

            self.assertEqual(mock_service.spreadsheets().values().append().execute.call_count, 2) # 2 batches

            # Verify content of batches (simplified check focusing on number of rows per batch)
            self.assertEqual(len(actual_append_bodies[0]), 4) # Metadata + Header
            self.assertEqual(len(actual_append_bodies[1]), 3) # 3 Data rows

            mock_get_sheet_id.assert_called_once_with("test_spreadsheet_id_batch", "ClassificationsSheetBatch")
            mock_apply_formatting.assert_called_once()


    @patch('google_sheets_client.Credentials.from_service_account_file')
    @patch('google_sheets_client.build')
    def test_export_classification_results_empty_input(self, mock_build, mock_creds_init):
        """Test export with empty results list."""
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        client = GoogleSheetsClient()
        client.service = mock_service

        with patch('google_sheets_client.logger.warning') as mock_logger_warning:
            success = client.export_classification_results([], "test_spreadsheet_id_empty", "EmptySheet")
            self.assertTrue(success) # Should return True as it's not an error
            mock_logger_warning.assert_called_with("No classification results provided for spreadsheet 'test_spreadsheet_id_empty', worksheet 'EmptySheet'. Skipping export.")
            mock_service.spreadsheets().values().append().execute.assert_not_called()


    @patch('google_sheets_client.Credentials.from_service_account_file')
    @patch('google_sheets_client.build')
    def test_export_classification_results_malformed_input(self, mock_build, mock_creds_init):
        """Test export with malformed results list (missing key)."""
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        client = GoogleSheetsClient()
        client.service = mock_service

        sample_results = [{'nct_id': 'NCT001'}] # Missing 'classification'

        # Mock underlying calls that would happen if it proceeds
        mock_service.spreadsheets().values().append().execute.return_value = {'updates': {}}
        with patch.object(client, '_get_sheet_id', return_value=1), \
             patch.object(client, '_apply_formatting_requests', return_value=True), \
             patch('google_sheets_client.logger.warning') as mock_logger_warning, \
             patch.object(client, '_make_api_request_with_retry', side_effect=lambda func, name: func()):
            
            success = client.export_classification_results(sample_results, "test_spreadsheet_id_malformed", "MalformedSheet")
            self.assertTrue(success) # Should still proceed and succeed
            mock_logger_warning.assert_any_call("First item in 'results' may be malformed (missing 'nct_id' or 'classification'). Proceeding with export.")


    @patch('google_sheets_client.Credentials.from_service_account_file')
    @patch('google_sheets_client.build')
    def test_export_classification_results_append_api_error(self, mock_build, mock_creds_init):
        """Test export failure due to API error during data append."""
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        client = GoogleSheetsClient()
        client.service = mock_service

        sample_results = [{'nct_id': 'NCT001', 'brief_title': 'T1', 'category': 'C1', 'classification': 'CONFIDENCE LEVEL: High'}]
        
        # _make_api_request_with_retry will return None if all retries fail for append
        with patch.object(client, '_make_api_request_with_retry', return_value=None) as mock_retry_helper, \
             patch('google_sheets_client.logger.error') as mock_logger_error:
            
            success = client.export_classification_results(sample_results, "test_spreadsheet_id_apierror", "ApiErrorSheet")
            self.assertFalse(success)
            mock_retry_helper.assert_called_once() # Called for the append operation
            # Logger call from export_classification_results itself
            mock_logger_error.assert_any_call("Failed to export all classification results to spreadsheet 'test_spreadsheet_id_apierror' due to batch failure.")

    # --- Tests for export_trial_data ---
    @patch.object(GoogleSheetsClient, '_get_sheet_id', return_value=67890)
    @patch.object(GoogleSheetsClient, '_apply_formatting_requests', return_value=True)
    @patch('google_sheets_client.Credentials.from_service_account_file')
    @patch('google_sheets_client.build')
    def test_export_trial_data_success_single_batch(
        self, mock_build, mock_creds_init, mock_apply_formatting, mock_get_sheet_id
    ):
        """Test successful export of trial data in a single batch."""
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        client = GoogleSheetsClient()
        client.service = mock_service

        sample_trials = [
            {'NCT ID': 'NCT001', 'Brief Title': 'Trial 1', 'Official Title': 'Official Trial 1', 'Sponsor': 'Sponsor A', 'Phase': 'Phase 2', 'Conditions': ['Condition A', 'Condition B'], 'Interventions': {'type': 'Drug', 'name': 'Drug X'}, 'Summary': 'Summary 1'},
            {'NCT ID': 'NCT002', 'Brief Title': 'Trial 2', 'Official Title': 'Official Trial 2', 'Sponsor': 'Sponsor B', 'Phases': ['Phase 1', 'Phase 2'], 'Conditions': 'Condition C', 'Interventions': None, 'Summary': 'Summary 2'},
        ]

        expected_header_row = ["NCT ID", "Brief Title", "Official Title", "Sponsor", "Phase", "Conditions", "Interventions", "Summary"]
        expected_data_rows = [
            ['NCT001', 'Trial 1', 'Official Trial 1', 'Sponsor A', 'Phase 2', 'Condition A, Condition B', json.dumps({'type': 'Drug', 'name': 'Drug X'}), 'Summary 1'],
            ['NCT002', 'Trial 2', 'Official Trial 2', 'Sponsor B', 'Phase 1, Phase 2', 'Condition C', json.dumps(None), 'Summary 2'],
        ]
        expected_values_to_append = [expected_header_row] + expected_data_rows
        
        mock_service.spreadsheets().values().append().execute.return_value = {'updates': {'updatedRange': 'RawDataSheet!A1'}}

        with patch.object(client, '_make_api_request_with_retry', side_effect=lambda func, name: func()) as mock_retry_helper:
            success = client.export_trial_data(sample_trials, "test_spreadsheet_id_trial", "RawDataSheet")
            self.assertTrue(success)

            append_args, append_kwargs = mock_service.spreadsheets().values().append.call_args
            self.assertIn('body', append_kwargs)
            self.assertEqual(append_kwargs['body']['values'], expected_values_to_append)
            
            mock_get_sheet_id.assert_called_once_with("test_spreadsheet_id_trial", "RawDataSheet")
            mock_apply_formatting.assert_called_once()

    @patch.object(GoogleSheetsClient, '_get_sheet_id', return_value=67890)
    @patch.object(GoogleSheetsClient, '_apply_formatting_requests', return_value=True)
    @patch('google_sheets_client.Credentials.from_service_account_file')
    @patch('google_sheets_client.build')
    def test_export_trial_data_batching(
        self, mock_build, mock_creds_init, mock_apply_formatting, mock_get_sheet_id
    ):
        """Test export of trial data with batching."""
        self.mock_config['batch_size'] = 2 # Header (1) + 1 data row. 3 trials total.
                                         # Header + D1 | D2 | D3. So 3 batches.
        
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        client = GoogleSheetsClient()
        client.service = mock_service

        sample_trials = [
            {'NCT ID': 'NCT001', 'Brief Title': 'T1', 'Phase': 'P1'},
            {'NCT ID': 'NCT002', 'Brief Title': 'T2', 'Phase': 'P2'},
            {'NCT ID': 'NCT003', 'Brief Title': 'T3', 'Phase': 'P3'},
        ]
        
        actual_append_bodies = []
        def capture_append_body(*args, **kwargs):
            actual_append_bodies.append(kwargs['body']['values'])
            return {'updates': {}}
        mock_service.spreadsheets().values().append().execute.side_effect = capture_append_body

        with patch.object(client, '_make_api_request_with_retry', side_effect=lambda func, name: func()):
            success = client.export_trial_data(sample_trials, "test_spreadsheet_id_trial_batch", "RawBatch")
            self.assertTrue(success)
            self.assertEqual(mock_service.spreadsheets().values().append().execute.call_count, 2) # Batch1: H+D1, Batch2: D2+D3
            self.assertEqual(len(actual_append_bodies[0]), 2) # Header + Data1
            self.assertEqual(len(actual_append_bodies[1]), 2) # Data2 + Data3
            mock_get_sheet_id.assert_called_once()
            mock_apply_formatting.assert_called_once()


    @patch('google_sheets_client.Credentials.from_service_account_file')
    @patch('google_sheets_client.build')
    def test_export_trial_data_empty_input(self, mock_build, mock_creds_init):
        """Test export_trial_data with empty trials list."""
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        client = GoogleSheetsClient()
        client.service = mock_service

        with patch('google_sheets_client.logger.warning') as mock_logger_warning:
            success = client.export_trial_data([], "test_spreadsheet_id_trial_empty", "EmptyTrialSheet")
            self.assertTrue(success)
            mock_logger_warning.assert_called_with("No trial data provided for spreadsheet 'test_spreadsheet_id_trial_empty', worksheet 'EmptyTrialSheet'. Skipping export.")
            mock_service.spreadsheets().values().append().execute.assert_not_called()

    @patch('google_sheets_client.Credentials.from_service_account_file')
    @patch('google_sheets_client.build')
    def test_export_trial_data_malformed_input(self, mock_build, mock_creds_init):
        """Test export_trial_data with malformed trials list."""
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        client = GoogleSheetsClient()
        client.service = mock_service
        sample_trials = [{'Brief Title': 'T1'}] # Missing 'NCT ID'

        mock_service.spreadsheets().values().append().execute.return_value = {'updates': {}}
        with patch.object(client, '_get_sheet_id', return_value=1), \
             patch.object(client, '_apply_formatting_requests', return_value=True), \
             patch('google_sheets_client.logger.warning') as mock_logger_warning, \
             patch.object(client, '_make_api_request_with_retry', side_effect=lambda func, name: func()):
            
            success = client.export_trial_data(sample_trials, "test_sp_trial_malformed", "MalformedTrialSheet")
            self.assertTrue(success)
            mock_logger_warning.assert_any_call("First item in 'trials' may be malformed (missing 'NCT ID'). Proceeding with export.")


    @patch('google_sheets_client.Credentials.from_service_account_file')
    @patch('google_sheets_client.build')
    def test_export_trial_data_append_api_error(self, mock_build, mock_creds_init):
        """Test export_trial_data failure due to API error during append."""
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        client = GoogleSheetsClient()
        client.service = mock_service
        sample_trials = [{'NCT ID': 'NCT001', 'Brief Title': 'T1'}]
        
        with patch.object(client, '_make_api_request_with_retry', return_value=None) as mock_retry_helper, \
             patch('google_sheets_client.logger.error') as mock_logger_error:
            
            success = client.export_trial_data(sample_trials, "test_sp_trial_apierror", "ApiErrorTrialSheet")
            self.assertFalse(success)
            mock_retry_helper.assert_called_once()
            mock_logger_error.assert_any_call("Failed to export all trial data to spreadsheet 'test_sp_trial_apierror' due to batch failure.")

    # --- Tests for create_summary_sheet ---
    @patch.object(GoogleSheetsClient, '_get_sheet_id', return_value=98765)
    @patch.object(GoogleSheetsClient, '_apply_formatting_requests', return_value=True)
    @patch('google_sheets_client.Credentials.from_service_account_file')
    @patch('google_sheets_client.build')
    def test_create_summary_sheet_success(
        self, mock_build, mock_creds_init, mock_apply_formatting, mock_get_sheet_id
    ):
        """Test successful creation of a summary sheet."""
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        client = GoogleSheetsClient()
        client.service = mock_service

        sample_results_for_summary = [
            {'category': 'Cancer', 'classification': 'CONFIDENCE LEVEL: High'},
            {'category': 'Cancer', 'classification': 'CONFIDENCE LEVEL: Medium'},
            {'category': 'Vaccine', 'classification': 'CONFIDENCE LEVEL: High'},
            {'category': 'Cancer', 'classification': 'CONFIDENCE LEVEL: High'},
            {'category': 'Not eligible', 'classification': '...'}, # Should be ignored
            {'category': 'Vaccine', 'classification': 'CONFIDENCE LEVEL: Low'},
            {'category': 'Genetic Medicines', 'classification': 'CONFIDENCE LEVEL: Medium'},
            {'category': 'Cancer', 'classification': 'CONFIDENCE LEVEL: unknown stuff here'}, # Unknown confidence
        ]
        # Total classified for percentage: 7 (excluding "Not eligible")
        # Cancer: 4 total. High:2, Med:1, Unknown:1. Percentage: (4/7)*100 = 57.14%
        # Vaccine: 2 total. High:1, Low:1. Percentage: (2/7)*100 = 28.57%
        # Genetic Medicines: 1 total. Med:1. Percentage: (1/7)*100 = 14.29%

        expected_header = ["Category", "Count", "Percentage", "High Confidence", "Medium Confidence", "Low Confidence"]
        # Rows are sorted by count descending
        expected_rows_data_sorted = [
            ['Cancer', 4, '57.14%', 2, 1, 0], # Unknown confidence is not in Low/Med/High
            ['Vaccine', 2, '28.57%', 1, 0, 1],
            ['Genetic Medicines', 1, '14.29%', 0, 1, 0],
        ]
        expected_values_to_append = [expected_header] + expected_rows_data_sorted
        
        mock_service.spreadsheets().values().append().execute.return_value = {'updates': {}}

        with patch.object(client, '_make_api_request_with_retry', side_effect=lambda func, name: func()):
            success = client.create_summary_sheet(sample_results_for_summary, "test_sp_summary", "SummarySheet")
            self.assertTrue(success)

            append_args, append_kwargs = mock_service.spreadsheets().values().append.call_args
            self.assertIn('body', append_kwargs)
            # Compare row by row due to potential floating point precision in percentage
            self.assertEqual(append_kwargs['body']['values'][0], expected_header) # Header
            self.assertEqual(len(append_kwargs['body']['values']), 4) # Header + 3 data rows
            
            # Check data rows with tolerance for percentage string
            for i, expected_row in enumerate(expected_rows_data_sorted):
                actual_row = append_kwargs['body']['values'][i+1]
                self.assertEqual(actual_row[0], expected_row[0]) # Category
                self.assertEqual(actual_row[1], expected_row[1]) # Count
                self.assertEqual(actual_row[2], expected_row[2]) # Percentage (string match)
                self.assertEqual(actual_row[3], expected_row[3]) # High
                self.assertEqual(actual_row[4], expected_row[4]) # Medium
                self.assertEqual(actual_row[5], expected_row[5]) # Low
            
            mock_get_sheet_id.assert_called_once_with("test_sp_summary", "SummarySheet")
            mock_apply_formatting.assert_called_once()

    @patch('google_sheets_client.Credentials.from_service_account_file')
    @patch('google_sheets_client.build')
    def test_create_summary_sheet_no_classified_trials(self, mock_build, mock_creds_init):
        """Test summary creation when there are no classified trials (e.g., all 'Not eligible')."""
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        client = GoogleSheetsClient()
        client.service = mock_service

        sample_results = [{'category': 'Not eligible', 'classification': '...'}]
        
        # Expect only headers to be written, or perhaps a message. Current impl writes headers + empty data.
        expected_header = ["Category", "Count", "Percentage", "High Confidence", "Medium Confidence", "Low Confidence"]
        expected_values_to_append = [expected_header] # No data rows

        mock_service.spreadsheets().values().append().execute.return_value = {'updates': {}}
        
        with patch.object(client, '_make_api_request_with_retry', side_effect=lambda func, name: func()), \
             patch('google_sheets_client.logger.warning') as mock_logger_warning, \
             patch.object(client, '_get_sheet_id', return_value=1), \
             patch.object(client, '_apply_formatting_requests', return_value=True):
            
            success = client.create_summary_sheet(sample_results, "test_sp_summary_empty", "EmptySummary")
            self.assertTrue(success) # Still true, an empty summary is not an error
            mock_logger_warning.assert_any_call("No classified trials found to create a summary. Skipping summary sheet creation.")
            
            append_args, append_kwargs = mock_service.spreadsheets().values().append.call_args
            self.assertEqual(append_kwargs['body']['values'], expected_values_to_append)


    @patch('google_sheets_client.Credentials.from_service_account_file')
    @patch('google_sheets_client.build')
    def test_create_summary_sheet_append_api_error(self, mock_build, mock_creds_init):
        """Test summary creation failure due to API error during append."""
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        client = GoogleSheetsClient()
        client.service = mock_service
        sample_results = [{'category': 'Cancer', 'classification': 'CONFIDENCE LEVEL: High'}]

        with patch.object(client, '_make_api_request_with_retry', return_value=None) as mock_retry_helper, \
             patch('google_sheets_client.logger.error') as mock_logger_error:
            success = client.create_summary_sheet(sample_results, "test_sp_summary_apierror", "ErrorSummary")
            self.assertFalse(success)
            mock_retry_helper.assert_called_once()
            mock_logger_error.assert_any_call(f"Failed to create summary sheet in spreadsheet 'test_sp_summary_apierror' after retries.")

    # --- Tests for Helper Methods ---
    @patch('google_sheets_client.Credentials.from_service_account_file') # To allow init
    @patch('google_sheets_client.build') # To allow init
    def test_get_sheet_id_success(self, mock_build, mock_creds_init):
        """Test _get_sheet_id successful retrieval."""
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        client = GoogleSheetsClient()
        client.service = mock_service

        mock_spreadsheet_data = {
            'sheets': [
                {'properties': {'sheetId': 0, 'title': 'Sheet1'}},
                {'properties': {'sheetId': 123, 'title': 'TargetSheet'}},
                {'properties': {'sheetId': 456, 'title': 'AnotherSheet'}},
            ]
        }
        mock_service.spreadsheets().get().execute.return_value = mock_spreadsheet_data
        
        with patch.object(client, '_make_api_request_with_retry', side_effect=lambda func, name: func()) as mock_retry_helper:
            sheet_id = client._get_sheet_id("test_sp_id", "TargetSheet")
            self.assertEqual(sheet_id, 123)
            mock_retry_helper.assert_called_once()
            mock_service.spreadsheets().get.assert_called_once_with(spreadsheetId="test_sp_id")

    @patch('google_sheets_client.Credentials.from_service_account_file')
    @patch('google_sheets_client.build')
    def test_get_sheet_id_not_found(self, mock_build, mock_creds_init):
        """Test _get_sheet_id when sheet is not found."""
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        client = GoogleSheetsClient()
        client.service = mock_service

        mock_spreadsheet_data = {'sheets': [{'properties': {'sheetId': 0, 'title': 'Sheet1'}}]}
        mock_service.spreadsheets().get().execute.return_value = mock_spreadsheet_data
        
        with patch.object(client, '_make_api_request_with_retry', side_effect=lambda func, name: func()), \
             patch('google_sheets_client.logger.warning') as mock_logger_warning:
            sheet_id = client._get_sheet_id("test_sp_id", "NonExistentSheet")
            self.assertIsNone(sheet_id)
            mock_logger_warning.assert_called_with("Sheet 'NonExistentSheet' not found in spreadsheet 'test_sp_id'.")

    @patch('google_sheets_client.Credentials.from_service_account_file')
    @patch('google_sheets_client.build')
    def test_get_sheet_id_api_error(self, mock_build, mock_creds_init):
        """Test _get_sheet_id when API call fails."""
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        client = GoogleSheetsClient()
        client.service = mock_service

        # _make_api_request_with_retry returns None on error
        with patch.object(client, '_make_api_request_with_retry', return_value=None) as mock_retry_helper, \
             patch('google_sheets_client.logger.error') as mock_logger_error:
            sheet_id = client._get_sheet_id("test_sp_id", "AnySheet")
            self.assertIsNone(sheet_id)
            mock_retry_helper.assert_called_once() # Called with the lambda for spreadsheets().get()
            mock_logger_error.assert_any_call("Failed to get spreadsheet details for 'test_sp_id' to find sheet 'AnySheet'.")


    @patch('google_sheets_client.Credentials.from_service_account_file')
    @patch('google_sheets_client.build')
    def test_apply_formatting_requests_success(self, mock_build, mock_creds_init):
        """Test _apply_formatting_requests successful application."""
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        client = GoogleSheetsClient()
        client.service = mock_service

        sample_requests = [{'updateSheetProperties': {}}]
        mock_service.spreadsheets().batchUpdate().execute.return_value = {'replies': [{}]}

        with patch.object(client, '_make_api_request_with_retry', side_effect=lambda func, name: func()) as mock_retry_helper:
            success = client._apply_formatting_requests("test_sp_id_format", sample_requests)
            self.assertTrue(success)
            mock_retry_helper.assert_called_once()
            mock_service.spreadsheets().batchUpdate.assert_called_once_with(
                spreadsheetId="test_sp_id_format",
                body={'requests': sample_requests}
            )

    @patch('google_sheets_client.Credentials.from_service_account_file')
    @patch('google_sheets_client.build')
    def test_apply_formatting_requests_no_requests(self, mock_build, mock_creds_init):
        """Test _apply_formatting_requests with no requests to apply."""
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        client = GoogleSheetsClient()
        client.service = mock_service
        
        with patch.object(client, '_make_api_request_with_retry') as mock_retry_helper:
            success = client._apply_formatting_requests("test_sp_id_format_empty", [])
            self.assertTrue(success) # Should be true if no requests
            mock_retry_helper.assert_not_called() # API call should not be made

    @patch('google_sheets_client.Credentials.from_service_account_file')
    @patch('google_sheets_client.build')
    def test_apply_formatting_requests_api_error(self, mock_build, mock_creds_init):
        """Test _apply_formatting_requests when API call fails."""
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        client = GoogleSheetsClient()
        client.service = mock_service
        sample_requests = [{'updateSheetProperties': {}}]

        with patch.object(client, '_make_api_request_with_retry', return_value=None) as mock_retry_helper, \
             patch('google_sheets_client.logger.error') as mock_logger_error:
            success = client._apply_formatting_requests("test_sp_id_format_error", sample_requests)
            self.assertFalse(success)
            mock_retry_helper.assert_called_once()
            mock_logger_error.assert_any_call("Failed to apply formatting requests to spreadsheet 'test_sp_id_format_error'.")

    # --- Test for _make_api_request_with_retry ---
    @patch('google_sheets_client.time.sleep', return_value=None) # Mock time.sleep to avoid delays
    @patch('google_sheets_client.Credentials.from_service_account_file')
    @patch('google_sheets_client.build')
    def test_make_api_request_with_retry_success_on_retry(
        self, mock_build, mock_creds_init, mock_sleep
    ):
        """Test _make_api_request_with_retry successfully retries on a 429 error."""
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        client = GoogleSheetsClient()
        client.service = mock_service
        self.mock_config['retry_attempts'] = 2 # Allow for one retry

        mock_api_target = MagicMock()
        # First call raises 429, second call succeeds
        mock_api_target.execute.side_effect = [
            create_http_error(429, "Rate Limited"),
            "SuccessData"
        ]
        
        # This callable represents something like: self.service.spreadsheets().get(...).execute
        # So, we need to ensure our mock_api_target is what gets called by the lambda
        # For simplicity, let's assume the api_call lambda is just `mock_api_target.execute`
        api_call_lambda = mock_api_target.execute 

        with patch('google_sheets_client.logger.info') as mock_logger_info:
            result = client._make_api_request_with_retry(api_call_lambda, "test_retry_method")
            self.assertEqual(result, "SuccessData")
            self.assertEqual(mock_api_target.execute.call_count, 2)
            mock_logger_info.assert_any_call("Rate limit exceeded for test_retry_method. Retrying in 1.xx seconds...".split("xx")[0]) # Check start of message
            mock_sleep.assert_called_once()

    @patch('google_sheets_client.Credentials.from_service_account_file')
    @patch('google_sheets_client.build')
    def test_read_sheet_data_success(self, mock_build, mock_creds_init):
        """Test successful reading of sheet data."""
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        client = GoogleSheetsClient()
        client.service = mock_service

        # Mock response with sample data
        mock_response = {
            'values': [
                ['Name', 'Age', 'City'],  # Headers
                ['John', '30', 'New York'],
                ['Alice', '25', 'London'],
                ['Bob', '35']  # Row with missing value
            ]
        }
        mock_service.spreadsheets().values().get().execute.return_value = mock_response

        with patch.object(client, '_make_api_request_with_retry', side_effect=lambda func, name: func()):
            result = client.read_sheet_data("test_sp_id", "TestSheet")
            
            self.assertIsNotNone(result)
            self.assertEqual(len(result), 3)  # 3 data rows
            
            # Check first row
            self.assertEqual(result[0], {
                'Name': 'John',
                'Age': '30',
                'City': 'New York'
            })
            
            # Check row with missing value (should be padded with empty string)
            self.assertEqual(result[2], {
                'Name': 'Bob',
                'Age': '35',
                'City': ''
            })
            
            # Verify API call
            mock_service.spreadsheets().values().get.assert_called_once_with(
                spreadsheetId="test_sp_id",
                range="TestSheet"
            )

    @patch('google_sheets_client.Credentials.from_service_account_file')
    @patch('google_sheets_client.build')
    def test_read_sheet_data_with_range(self, mock_build, mock_creds_init):
        """Test reading sheet data with a specific range."""
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        client = GoogleSheetsClient()
        client.service = mock_service

        mock_response = {
            'values': [
                ['Name', 'Age'],
                ['John', '30'],
                ['Alice', '25']
            ]
        }
        mock_service.spreadsheets().values().get().execute.return_value = mock_response

        with patch.object(client, '_make_api_request_with_retry', side_effect=lambda func, name: func()):
            result = client.read_sheet_data("test_sp_id", "TestSheet", "A1:B3")
            
            self.assertIsNotNone(result)
            self.assertEqual(len(result), 2)  # 2 data rows
            
            # Verify API call with range
            mock_service.spreadsheets().values().get.assert_called_once_with(
                spreadsheetId="test_sp_id",
                range="TestSheet!A1:B3"
            )

    @patch('google_sheets_client.Credentials.from_service_account_file')
    @patch('google_sheets_client.build')
    def test_read_sheet_data_empty(self, mock_build, mock_creds_init):
        """Test reading from an empty sheet."""
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        client = GoogleSheetsClient()
        client.service = mock_service

        mock_response = {'values': []}
        mock_service.spreadsheets().values().get().execute.return_value = mock_response

        with patch.object(client, '_make_api_request_with_retry', side_effect=lambda func, name: func()):
            result = client.read_sheet_data("test_sp_id", "EmptySheet")
            
            self.assertEqual(result, [])  # Should return empty list, not None
            mock_service.spreadsheets().values().get.assert_called_once()

    @patch('google_sheets_client.Credentials.from_service_account_file')
    @patch('google_sheets_client.build')
    def test_read_sheet_data_failure(self, mock_build, mock_creds_init):
        """Test reading sheet data when API call fails."""
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        client = GoogleSheetsClient()
        client.service = mock_service

        # Mock API call failure
        mock_service.spreadsheets().values().get().execute.side_effect = HttpError(
            resp=MagicMock(status=404),
            content=b'Not Found'
        )

        with patch.object(client, '_make_api_request_with_retry', return_value=None):
            result = client.read_sheet_data("test_sp_id", "NonExistentSheet")
            
            self.assertIsNone(result)  # Should return None on failure
            mock_service.spreadsheets().values().get.assert_called_once()

    def test_export_to_sheet_success(self, mock_service, mock_credentials):
        """Test successful export to Google Sheets"""
        # Create test data
        data = [
            ["Header1", "Header2"],
            ["Value1", "Value2"]
        ]
        
        # Create GoogleSheetsClient instance
        client = GoogleSheetsClient()


if __name__ == '__main__':
    unittest.main()
