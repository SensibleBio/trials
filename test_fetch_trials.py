import unittest
from unittest.mock import patch, Mock, mock_open, ANY
# Assuming fetch_clinical_trials and DEFAULT_STATUSES are in fetch_trials
from fetch_trials import (
    fetch_clinical_trials, 
    process_trials_data, 
    save_trials_json, 
    save_trials_csv, 
    load_trials_json, 
    fetch_and_save_trials,
    extract_trial_info, 
    DEFAULT_STATUSES,
    # _prepare_metadata_for_saving, # Not needed directly in test file if patching fetch_trials.*
    # save_dataset_metadata        # Not needed directly in test file if patching fetch_trials.*
)
import os 
import csv 
import json 
import requests 

class TestFetchClinicalTrials(unittest.TestCase):

    @patch('fetch_trials.requests.get')
    def test_fetch_clinical_trials_parameterization_and_return_type(self, mock_get):
        mock_response_instance = Mock(spec=requests.Response) 
        mock_response_instance.status_code = 200
        mock_response_instance.url = "http://mockurl.com/success" 
        mock_response_instance.json.return_value = {"studies": [], "totalCount": 0} 
        mock_get.return_value = mock_response_instance
        default_statuses_str = ",".join(DEFAULT_STATUSES)
        result_response = fetch_clinical_trials("cancer", sponsor="TestSponsor")
        self.assertIs(result_response, mock_response_instance) 
        expected_params_sponsor = {
            "format": "json", "query.term": "cancer", 
            "filter.overallStatus": default_statuses_str,
            "filter.advanced": "AREA[StudyType]Interventional",
            "pageSize": "20", "countTotal": "true", "query.spons": "TestSponsor"
        }
        called_args, called_kwargs = mock_get.call_args
        self.assertEqual(called_kwargs.get('params'), expected_params_sponsor)

    @patch('fetch_trials.requests.get')
    def test_limit_validation_in_fetch_clinical_trials(self, mock_get):
        mock_response = Mock(spec=requests.Response)
        mock_response.status_code = 200
        mock_response.url = "http://mockurl.com/limit_validation" 
        mock_get.return_value = mock_response
        fetch_clinical_trials("test_query", limit=0)
        self.assertEqual(mock_get.call_args[1]['params']['pageSize'], "1")
        fetch_clinical_trials("test_query", limit=1500)
        self.assertEqual(mock_get.call_args[1]['params']['pageSize'], "1000")

class TestExtractTrialInfo(unittest.TestCase):
    def _create_base_study_dict(self):
        return {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT123", "officialTitle": "Official Title", "briefTitle": "Brief Title"},
                "statusModule": {"overallStatus": "Recruiting", "startDateStruct": {"date": "2023-01-01"}, "completionDateStruct": {"date": "2024-01-01"}},
                "descriptionModule": {"briefSummary": "Summary", "detailedDescription": "Detailed Info"},
                "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Sponsor Inc"}, "collaborators": [{"name": "Collab A"}]},
                "conditionsModule": {"conditions": ["Condition A"]},
                "interventionsModule": {"interventions": [{"interventionType": "Drug", "interventionName": "DrugX", "description": "Desc X"}]},
                "designModule": {"phases": ["PHASE2"], "studyType": "Interventional"}
            }
        }
    def test_extract_full_data(self):
        study_data = self._create_base_study_dict()
        expected_output = {
            "nct_id": "NCT123", "brief_title": "Brief Title", "official_title": "Official Title",
            "overall_status": "Recruiting", "start_date": "2023-01-01", "completion_date": "2024-01-01",
            "brief_summary": "Summary", "detailed_description": "Detailed Info",
            "sponsor": "Sponsor Inc", "collaborators": ["Collab A"],
            "conditions": ["Condition A"],
            "interventions": [{"type": "Drug", "name": "DrugX", "description": "Desc X"}],
            "phases": ["PHASE2"], "study_type": "Interventional"
        }
        self.assertEqual(extract_trial_info(study_data), expected_output)

    @patch('fetch_trials.logging')
    def test_missing_protocol_section(self, mock_logging):
        self.assertIsNone(extract_trial_info({}))
        mock_logging.warning.assert_called_once_with("Warning: Study missing protocolSection. Keys: dict_keys([])")

    def test_missing_optional_fields(self):
        study_data = self._create_base_study_dict()
        del study_data["protocolSection"]["descriptionModule"]["briefSummary"]
        study_data["protocolSection"]["designModule"]["phases"] = [] 
        result = extract_trial_info(study_data)
        self.assertEqual(result["brief_summary"], "No summary") # type: ignore
        self.assertEqual(result["phases"], ["Unknown phase"]) # type: ignore

class TestProcessTrialsData(unittest.TestCase):
    @patch('fetch_trials.extract_trial_info')
    @patch('fetch_trials.logging') 
    def test_process_valid_data(self, mock_logging, mock_extract_trial_info):
        mock_extract_trial_info.side_effect = lambda x: {"nct_id": x.get("protocolSection", {}).get("identificationModule", {}).get("nctId")}
        raw_data = {"studies": [{"protocolSection": {"identificationModule": {"nctId": "NCT001"}}}]}
        result = process_trials_data(raw_data)
        self.assertEqual(len(result), 1)

class TestSaveTrialsJson(unittest.TestCase):
    @patch('fetch_trials.os.makedirs')
    @patch('fetch_trials.open', new_callable=mock_open)
    @patch('fetch_trials.json.dump')
    def test_save_trials_default_folder(self, mock_json_dump, mock_file_open, mock_makedirs):
        save_trials_json([{"id": 1}], "f.json") # folder defaults to "pulled_data"
        expected_filepath = os.path.join("pulled_data", "f.json")
        mock_makedirs.assert_called_once_with("pulled_data", exist_ok=True)
        mock_file_open.assert_called_once_with(expected_filepath, "w", encoding="utf-8")

class TestSaveTrialsCsv(unittest.TestCase):
    @patch('fetch_trials.os.makedirs')
    @patch('fetch_trials.open', new_callable=mock_open)
    @patch('fetch_trials.csv.DictWriter')
    def test_save_trials_csv_basic(self, mock_dict_writer_constructor, mock_file_open, mock_makedirs):
        save_trials_csv([{"id": "N1"}], "o.csv", folder="csv_data")
        mock_makedirs.assert_called_once_with("csv_data", exist_ok=True)

class TestLoadTrialsJson(unittest.TestCase):
    @patch('fetch_trials.open', new_callable=mock_open)
    @patch('fetch_trials.json.load')
    def test_load_trials_json_success(self, mock_json_load, mock_file_open):
        mock_json_load.return_value = [{"id": "N1"}]
        self.assertIsNotNone(load_trials_json("g.json", folder="test_data"))

class TestFetchAndSaveTrials(unittest.TestCase):

    # Patch dataset_manager functions as they are used within fetch_trials.py's namespace
    @patch('fetch_trials.fetch_clinical_trials')
    @patch('fetch_trials.process_trials_data')
    @patch('fetch_trials.save_trials_json')
    @patch('fetch_trials.save_trials_csv')
    @patch('fetch_trials._prepare_metadata_for_saving') # Patched here
    @patch('fetch_trials.save_dataset_metadata')     # Patched here
    @patch('fetch_trials.logging') 
    def test_successful_run_json_and_csv_with_metadata(
        self, mock_logging, mock_save_dataset_meta, mock_prepare_meta, 
        mock_save_csv, mock_save_json, mock_process_data, mock_fetch_trials):
        
        # --- Setup Mocks ---
        mock_response = Mock(spec=requests.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"studies": [{"id": "NCT123", "protocolSection": {}}]} 
        mock_fetch_trials.return_value = mock_response
        
        processed_trials_data = [{"nct_id": "NCT123", "title": "Processed Trial"}] 
        mock_process_data.return_value = processed_trials_data

        # Mock for _prepare_metadata_for_saving
        prepared_metadata = {"created": "timestamp", "actual_count": 1}
        mock_prepare_meta.return_value = prepared_metadata

        # --- Call the function ---
        output_filename = "test_output.json"
        actual_data_save_folder = "pulled_data" # This is hardcoded in fetch_and_save_trials logic
        full_data_filepath = os.path.join(actual_data_save_folder, output_filename)

        # Parameters for fetch_and_save_trials
        query_term_param = "test_query"
        limit_param = 10
        statuses_param = ["RECRUITING"] # Example, ensure it's a list
        sponsor_param = "Test Sponsor"
        phase_param = ["PHASE1"]


        result = fetch_and_save_trials(
            query_term=query_term_param,
            limit=limit_param,
            statuses=statuses_param,
            output_file=output_filename, 
            sponsor=sponsor_param,
            phase=phase_param,
            save_csv=True
        )

        # --- Assertions ---
        self.assertEqual(result, processed_trials_data)
        mock_fetch_trials.assert_called_once_with(
            query_term=query_term_param, limit=limit_param, statuses=statuses_param, 
            sponsor=sponsor_param, phase=phase_param
        )
        mock_process_data.assert_called_once_with({"studies": [{"id": "NCT123", "protocolSection": {}}]})
        mock_save_json.assert_called_once_with(processed_trials_data, filename=output_filename, folder=actual_data_save_folder)
        
        # Metadata assertions
        expected_search_params = {
            "query_term": query_term_param, "sponsor": sponsor_param, "phases": phase_param,
            "limit_requested": limit_param, "statuses": statuses_param
        }
        mock_prepare_meta.assert_called_once_with(
            trials=processed_trials_data,
            data_filepath=full_data_filepath,
            search_params=expected_search_params
        )
        mock_save_dataset_meta.assert_called_once_with(
            metadata=prepared_metadata,
            data_filename=output_filename,
            folder=actual_data_save_folder
        )
        
        # CSV assertion
        csv_filename = "test_output.csv"
        mock_save_csv.assert_called_once_with(processed_trials_data, filename=csv_filename, folder=actual_data_save_folder)

    @patch('fetch_trials.fetch_clinical_trials')
    @patch('fetch_trials.process_trials_data')
    @patch('fetch_trials.save_trials_json')
    @patch('fetch_trials._prepare_metadata_for_saving') 
    @patch('fetch_trials.save_dataset_metadata')     
    @patch('fetch_trials.logging')
    def test_metadata_saving_exception(
        self, mock_logging, mock_save_dataset_meta, mock_prepare_meta, 
        mock_save_json, mock_process_data, mock_fetch_trials):
        
        mock_response = Mock(spec=requests.Response); mock_response.status_code = 200
        mock_response.json.return_value = {"studies": [{"id": "NCT123"}]}
        mock_fetch_trials.return_value = mock_response
        processed_trials_data = [{"nct_id": "NCT123"}]
        mock_process_data.return_value = processed_trials_data
        
        # Simulate _prepare_metadata_for_saving raising an exception
        mock_prepare_meta.side_effect = Exception("Metadata prep error")

        output_filename = "meta_error_test.json"
        result = fetch_and_save_trials(output_file=output_filename, save_csv=False) # save_csv=False to simplify

        self.assertEqual(result, processed_trials_data) # Should still return trials
        mock_save_json.assert_called_once() # JSON save should have happened
        mock_prepare_meta.assert_called_once() # Preparation was attempted
        mock_save_dataset_meta.assert_not_called() # Actual save of meta should not happen
        mock_logging.error.assert_any_call(f"Error occurred during metadata saving for '{output_filename}': Metadata prep error", exc_info=True)


    # Keeping existing failure case tests for fetch_and_save_trials
    @patch('fetch_trials.fetch_clinical_trials')
    def test_fetch_clinical_trials_returns_none(self, mock_fetch_trials):
        mock_fetch_trials.return_value = None
        self.assertEqual(fetch_and_save_trials(), [])

    @patch('fetch_trials.fetch_clinical_trials')
    def test_fetch_clinical_trials_returns_non_200_status(self, mock_fetch_trials):
        mock_response = Mock(spec=requests.Response); mock_response.status_code = 404
        mock_response.text="Not Found"
        # fetch_clinical_trials itself returns None on non-200, so simulate that
        mock_fetch_trials.return_value = None 
        self.assertEqual(fetch_and_save_trials(), [])

    @patch('fetch_trials.fetch_clinical_trials')
    def test_response_json_decode_error(self, mock_fetch_trials):
        mock_response = Mock(spec=requests.Response); mock_response.status_code = 200
        mock_response.json.side_effect = requests.exceptions.JSONDecodeError("Error", "doc", 0)
        mock_fetch_trials.return_value = mock_response
        self.assertEqual(fetch_and_save_trials(), [])

    @patch('fetch_trials.fetch_clinical_trials')
    @patch('fetch_trials.process_trials_data')
    @patch('fetch_trials.save_trials_json') 
    def test_process_data_returns_empty(self, mock_save_json, mock_process_data, mock_fetch_trials):
        mock_response = Mock(spec=requests.Response); mock_response.status_code = 200
        mock_response.json.return_value = {"studies": [{"id": "NCT789"}]}
        mock_fetch_trials.return_value = mock_response
        mock_process_data.return_value = []
        self.assertEqual(fetch_and_save_trials(), [])
        mock_save_json.assert_not_called()


if __name__ == '__main__':
    unittest.main()
