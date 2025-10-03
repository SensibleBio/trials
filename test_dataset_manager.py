import unittest
from unittest.mock import patch, mock_open, Mock, call, ANY
import os
import json # For json.JSONDecodeError
from datetime import datetime, timezone # Added for mocking datetime.now

# Import the functions to be tested
from dataset_manager import (
    load_dataset, 
    deduplicate_trials, 
    load_dataset_metadata, 
    save_dataset_metadata,
    _prepare_metadata_for_saving,
    merge_datasets,
    get_dataset_info,
    list_available_datasets,
    _save_dataset_json # Added import
)

class TestLoadDataset(unittest.TestCase):

    @patch('dataset_manager.open', new_callable=mock_open)
    @patch('dataset_manager.json.load')
    @patch('dataset_manager.logger') 
    def test_load_dataset_success(self, mock_logger, mock_json_load, mock_file_open):
        filename = "good_data.json"; folder = "test_data_folder"
        expected_filepath = os.path.join(folder, filename) 
        expected_data = [{"trial_id": "NCT123"}, {"trial_id": "NCT456"}]
        mock_json_load.return_value = expected_data
        result = load_dataset(filename, folder=folder)
        mock_file_open.assert_called_once_with(expected_filepath, 'r', encoding='utf-8')
        self.assertEqual(result, expected_data)

    @patch('dataset_manager.open', new_callable=mock_open)
    @patch('dataset_manager.logger')
    def test_load_dataset_file_not_found(self, mock_logger, mock_file_open):
        filename = "non_existent.json"; folder = "test_data_folder"
        expected_filepath = os.path.join(folder, filename)
        mock_file_open.side_effect = FileNotFoundError()
        self.assertIsNone(load_dataset(filename, folder=folder))
        mock_logger.error.assert_called_once_with(f"Dataset file not found: {expected_filepath}")

    @patch('dataset_manager.open', new_callable=mock_open)
    @patch('dataset_manager.json.load')
    @patch('dataset_manager.logger')
    def test_load_dataset_json_decode_error(self, mock_logger, mock_json_load, mock_file_open):
        filename = "bad_format.json"; folder = "test_data_folder"
        expected_filepath = os.path.join(folder, filename) 
        mock_json_load.side_effect = json.JSONDecodeError("Syntax error", "doc", 0)
        self.assertIsNone(load_dataset(filename, folder=folder))
        mock_logger.error.assert_any_call(f"Error decoding JSON from {expected_filepath}: Syntax error: line 1 column 1 (char 0)")


    @patch('dataset_manager.open', new_callable=mock_open)
    @patch('dataset_manager.json.load')
    @patch('dataset_manager.logger')
    def test_load_dataset_not_a_list(self, mock_logger, mock_json_load, mock_file_open):
        filename = "not_a_list.json"; folder = "test_data_folder"
        expected_filepath = os.path.join(folder, filename)
        mock_json_load.return_value = {"oops": "not a list"}
        self.assertIsNone(load_dataset(filename, folder=folder))
        mock_logger.error.assert_called_once_with(f"Invalid data format in {expected_filepath}: Expected a list, got dict.")

    @patch('dataset_manager.open', new_callable=mock_open)
    @patch('dataset_manager.logger')
    def test_load_dataset_other_io_error(self, mock_logger, mock_file_open):
        filename = "io_error_file.json"; folder = "test_data_folder"
        expected_filepath = os.path.join(folder, filename)
        mock_file_open.side_effect = IOError("Disk error")
        self.assertIsNone(load_dataset(filename, folder=folder))
        mock_logger.error.assert_called_once_with(f"IOError when trying to read {expected_filepath}: Disk error")

class TestDeduplicateTrials(unittest.TestCase):
    @patch('dataset_manager.logger')
    def test_empty_list(self, mock_logger):
        self.assertEqual(deduplicate_trials([]), [])

    @patch('dataset_manager.logger')
    def test_no_duplicates(self, mock_logger):
        trials = [{"nct_id": "NCT001"}, {"nct_id": "NCT002"}]
        self.assertEqual(deduplicate_trials(list(trials)), trials)

    @patch('dataset_manager.logger')
    def test_with_duplicates(self, mock_logger):
        trials = [{"nct_id": "NCT001", "v":1}, {"nct_id": "NCT002"}, {"nct_id": "NCT001", "v":2}]
        expected = [{"nct_id": "NCT001", "v":1}, {"nct_id": "NCT002"}]
        self.assertEqual(deduplicate_trials(list(trials)), expected)

    @patch('dataset_manager.logger')
    def test_missing_nct_id(self, mock_logger):
        trials = [{"nct_id": "NCT001"}, {"title": "No NCT ID"}, {"nct_id": ""}, {"nct_id": "NCT002"}]
        expected = [{"nct_id": "NCT001"}, {"nct_id": "NCT002"}]
        self.assertEqual(deduplicate_trials(list(trials)), expected)
        self.assertEqual(mock_logger.warning.call_count, 2)

    @patch('dataset_manager.logger')
    def test_item_not_a_dict(self, mock_logger):
        trials = [{"nct_id": "NCT001"}, "string_item", {"nct_id": "NCT002"}]
        expected = [{"nct_id": "NCT001"}, {"nct_id": "NCT002"}]
        self.assertEqual(deduplicate_trials(list(trials)), expected) # type: ignore
        mock_logger.warning.assert_called_with("Item at index 1 is not a dictionary (type: str), skipping.")

class TestLoadDatasetMetadata(unittest.TestCase):
    @patch('dataset_manager.open', new_callable=mock_open)
    @patch('dataset_manager.json.load')
    @patch('dataset_manager.logger')
    def test_load_metadata_success(self, mock_logger, mock_json_load, mock_file_open):
        data_fn = "d.json"; folder = "meta_folder"
        meta_fn = data_fn + ".meta"; expected_fp = os.path.join(folder, meta_fn)
        expected_meta = {"name": "Test", "version": "1"}
        mock_json_load.return_value = expected_meta
        self.assertEqual(load_dataset_metadata(data_fn, folder=folder), expected_meta)
        mock_file_open.assert_called_once_with(expected_fp, 'r', encoding='utf-8')

    @patch('dataset_manager.open', new_callable=mock_open)
    @patch('dataset_manager.logger')
    def test_load_metadata_file_not_found(self, mock_logger, mock_file_open):
        data_fn="no.json"; folder="f"; meta_fn=data_fn+".meta"; expected_fp=os.path.join(folder,meta_fn)
        mock_file_open.side_effect = FileNotFoundError()
        self.assertIsNone(load_dataset_metadata(data_fn, folder=folder))
        mock_logger.warning.assert_called_with(f"Metadata file not found: {expected_fp} (for data file {data_fn})")

    @patch('dataset_manager.open', new_callable=mock_open)
    @patch('dataset_manager.json.load')
    @patch('dataset_manager.logger')
    def test_load_metadata_json_decode_error(self, mock_logger, mock_json_load, mock_file_open):
        data_fn="bad.json";folder="f";meta_fn=data_fn+".meta";expected_fp=os.path.join(folder,meta_fn)
        mock_json_load.side_effect = json.JSONDecodeError("e", "d", 0)
        self.assertIsNone(load_dataset_metadata(data_fn, folder=folder))
        mock_logger.error.assert_any_call(f"Error decoding JSON from metadata file {expected_fp}: e: line 1 column 1 (char 0)")


    @patch('dataset_manager.open', new_callable=mock_open)
    @patch('dataset_manager.json.load')
    @patch('dataset_manager.logger')
    def test_load_metadata_not_a_dict(self, mock_logger, mock_json_load, mock_file_open):
        data_fn="list.json";folder="f";meta_fn=data_fn+".meta";expected_fp=os.path.join(folder,meta_fn)
        mock_json_load.return_value = ["not a dict"]
        self.assertIsNone(load_dataset_metadata(data_fn, folder=folder))
        mock_logger.error.assert_called_with(f"Invalid metadata format in {expected_fp}: Expected a dictionary, got list.")

class TestSaveDatasetMetadata(unittest.TestCase):
    @patch('dataset_manager.os.makedirs')
    @patch('dataset_manager.open', new_callable=mock_open)
    @patch('dataset_manager.json.dump')
    @patch('dataset_manager.logger')
    def test_save_metadata_success(self, mock_logger, mock_json_dump, mock_file_open, mock_makedirs):
        meta = {"v": "1"}; data_fn = "d.json"; folder = "out"
        meta_fn = data_fn + ".meta"; expected_fp = os.path.join(folder, meta_fn)
        save_dataset_metadata(meta, data_fn, folder=folder)
        mock_makedirs.assert_called_once_with(folder, exist_ok=True)
        mock_file_open.assert_called_once_with(expected_fp, 'w', encoding='utf-8')
        mock_json_dump.assert_called_once_with(meta, mock_file_open(), indent=4)

    @patch('dataset_manager.logger')
    def test_save_invalid_metadata(self, mock_logger): 
        save_dataset_metadata(None, "no.json") # type: ignore
        mock_logger.error.assert_called_with("Invalid metadata provided; must be a non-empty dictionary. Metadata not saved.")

class TestPrepareMetadataForSaving(unittest.TestCase):
    @patch('dataset_manager.datetime')
    @patch('dataset_manager.os.path.exists')
    @patch('dataset_manager.os.path.getsize')
    def test_prepare_metadata_success(self, mock_getsize, mock_exists, mock_datetime):
        mock_now = Mock(); mock_now.isoformat.return_value = "NOW"; mock_datetime.now.return_value = mock_now
        mock_exists.return_value = True; mock_getsize.return_value = 100
        trials, data_filepath, params = [{"id":1}], "fp_success", {"query_term": "q", "sponsor":"s", "phases":["P1"], "statuses":["S1"], "limit_requested":10}
        meta = _prepare_metadata_for_saving(trials, data_filepath, params)
        self.assertEqual(meta["created"], "NOW")
        self.assertEqual(meta["actual_count"], 1)
        self.assertEqual(meta["file_size_bytes"], 100)
        self.assertEqual(meta["query_term"], "q")

    @patch('dataset_manager.datetime')
    @patch('dataset_manager.os.path.exists')
    @patch('dataset_manager.os.path.getsize') 
    @patch('dataset_manager.logger')
    def test_prepare_metadata_file_not_exist(self, mock_logger, mock_getsize, mock_exists, mock_datetime):
        mock_exists.return_value = False
        data_filepath = "fp_not_exist"
        meta = _prepare_metadata_for_saving([], data_filepath, {})
        self.assertEqual(meta["file_size_bytes"], -1)
        mock_getsize.assert_not_called()
        mock_logger.warning.assert_called_with(f"Data file {data_filepath} not found for size calculation. Size will be set to -1.")

    @patch('dataset_manager.datetime')
    @patch('dataset_manager.os.path.exists')
    @patch('dataset_manager.os.path.getsize')
    @patch('dataset_manager.logger')
    def test_prepare_metadata_getsize_os_error(self, mock_logger, mock_getsize, mock_exists, mock_datetime):
        mock_exists.return_value = True
        data_filepath = "fp_os_error"
        mock_getsize.side_effect = OSError("Permission denied")
        meta = _prepare_metadata_for_saving([], data_filepath, {})
        self.assertEqual(meta["file_size_bytes"], -1)
        mock_logger.warning.assert_called_with(f"Could not get file size for {data_filepath} due to OSError: Permission denied. Size will be set to -1.")

    @patch('dataset_manager.datetime')
    @patch('dataset_manager.os.path.exists')
    @patch('dataset_manager.os.path.getsize')
    def test_prepare_metadata_trials_is_none(self, mock_getsize, mock_exists, mock_datetime):
        mock_exists.return_value = True; mock_getsize.return_value = 0
        meta = _prepare_metadata_for_saving(None, "fp_none_trials", {}) # type: ignore
        self.assertEqual(meta["actual_count"], 0)

class TestMergeDatasets(unittest.TestCase):

    @patch('dataset_manager.load_dataset')
    @patch('dataset_manager.load_dataset_metadata')
    @patch('dataset_manager._save_dataset_json') # Updated from _save_json_data_temp
    @patch('dataset_manager._prepare_metadata_for_saving')
    @patch('dataset_manager.save_dataset_metadata')
    @patch('dataset_manager.logger')
    def test_merge_basic_scenario(self, mock_logger, mock_save_meta, mock_prep_meta, mock_save_json, mock_load_meta, mock_load_ds): # Renamed mock_save_temp
        mock_load_ds.side_effect = [[{"nct_id": "NCT001", "data": "D1"}], [{"nct_id": "NCT002", "data": "D2"}]]
        mock_load_meta.side_effect = [{"created": "T1"}, {"created": "T2"}]
        mock_save_json.return_value = True 
        mock_prep_meta.return_value = {"merged": True}
        input_files = ["a.json", "b.json"]; output_file = "m.json"
        result = merge_datasets(input_files, output_file, data_folder="in", output_folder="out")
        self.assertEqual(len(result), 2) # type: ignore
        mock_save_json.assert_called_once() # Check if the save function was called
        mock_save_meta.assert_called_once_with({"merged": True}, output_file, folder="out")

    @patch('dataset_manager.load_dataset')
    @patch('dataset_manager.logger')
    def test_merge_load_dataset_fails(self, mock_logger, mock_load_ds):
        mock_load_ds.return_value = None 
        self.assertEqual(merge_datasets(["f1.json"], "o.json"), []) 

    @patch('dataset_manager.load_dataset')
    @patch('dataset_manager.load_dataset_metadata')
    @patch('dataset_manager.logger')
    def test_merge_metadata_load_fails(self, mock_logger, mock_load_meta, mock_load_ds):
        mock_load_ds.return_value = [{"nct_id": "N1"}] 
        mock_load_meta.return_value = None 
        self.assertEqual(merge_datasets(["f1.json"], "o.json"), []) 
        
    @patch('dataset_manager.load_dataset')
    @patch('dataset_manager.load_dataset_metadata')
    @patch('dataset_manager.logger')
    def test_merge_metadata_missing_created_ts(self, mock_logger, mock_load_meta, mock_load_ds):
        mock_load_ds.return_value = [{"nct_id": "N1"}]
        mock_load_meta.return_value = {"other": "v"}
        self.assertEqual(merge_datasets(["f1.json"], "o.json"), [])

    @patch('dataset_manager.logger')
    def test_merge_empty_input_filenames(self, mock_logger):
        self.assertIsNone(merge_datasets([], "o.json"))

    @patch('dataset_manager.load_dataset')
    @patch('dataset_manager.load_dataset_metadata')
    @patch('dataset_manager._save_dataset_json') # Updated from _save_json_data_temp
    @patch('dataset_manager._prepare_metadata_for_saving') 
    @patch('dataset_manager.save_dataset_metadata')
    @patch('dataset_manager.logger')
    def test_merge_save_merged_data_fails(self, mock_logger, mock_save_meta, mock_prep_meta, mock_save_json, mock_load_meta, mock_load_ds): # Renamed
        mock_load_ds.return_value = [{"nct_id": "N1"}]
        mock_load_meta.return_value = {"created": "T1"}
        mock_save_json.return_value = False 
        result = merge_datasets(["a.json"], "m.json")
        self.assertEqual(len(result), 1) # type: ignore
        mock_prep_meta.assert_not_called() 
        mock_save_meta.assert_not_called() 

class TestGetDatasetInfo(unittest.TestCase):

    @patch('dataset_manager.load_dataset_metadata')
    @patch('dataset_manager.logger')
    def test_get_dataset_info_success(self, mock_logger, mock_load_meta):
        data_filename = "my_trials.json"; folder = "my_folder"
        expected_metadata = {"version": "1.0", "count": 100}
        mock_load_meta.return_value = expected_metadata
        result = get_dataset_info(data_filename, folder=folder)
        self.assertEqual(result, expected_metadata)
        mock_load_meta.assert_called_once_with(data_filename, folder=folder)

    @patch('dataset_manager.load_dataset_metadata')
    @patch('dataset_manager.logger')
    def test_get_dataset_info_no_metadata(self, mock_logger, mock_load_meta):
        data_filename = "no_metadata_trials.json"; folder = "another_folder"
        mock_load_meta.return_value = None 
        result = get_dataset_info(data_filename, folder=folder)
        self.assertIsNone(result)
        mock_logger.info.assert_any_call(f"Could not retrieve dataset info for {data_filename} as its metadata was not found, is invalid, or an error occurred during loading.")

    @patch('dataset_manager.load_dataset_metadata')
    @patch('dataset_manager.logger')
    def test_get_dataset_info_default_folder(self, mock_logger, mock_load_meta):
        data_filename = "default_folder_trials.json"
        expected_metadata = {"source": "test_source"}
        mock_load_meta.return_value = expected_metadata
        result = get_dataset_info(data_filename) 
        self.assertEqual(result, expected_metadata)
        mock_load_meta.assert_called_once_with(data_filename, folder="pulled_data")

class TestListAvailableDatasets(unittest.TestCase):

    @patch('dataset_manager.os.path.isdir')
    @patch('dataset_manager.logger')
    def test_folder_not_exist_or_not_dir(self, mock_logger, mock_isdir):
        mock_isdir.return_value = False
        self.assertEqual(list_available_datasets(folder="non_existent_folder"), [])
        mock_logger.warning.assert_called_once_with("Dataset folder 'non_existent_folder' does not exist or is not a directory.")

    @patch('dataset_manager.os.path.isdir')
    @patch('dataset_manager.os.listdir')
    @patch('dataset_manager.logger')
    def test_os_listdir_error(self, mock_logger, mock_listdir, mock_isdir):
        mock_isdir.return_value = True
        mock_listdir.side_effect = OSError("Permission denied")
        self.assertEqual(list_available_datasets(folder="error_folder"), [])
        mock_logger.error.assert_called_once_with("OSError when trying to list datasets in 'error_folder': Permission denied")

    @patch('dataset_manager.os.path.isdir')
    @patch('dataset_manager.os.listdir')
    @patch('dataset_manager.logger')
    def test_empty_folder(self, mock_logger, mock_listdir, mock_isdir):
        mock_isdir.return_value = True
        mock_listdir.return_value = []
        self.assertEqual(list_available_datasets(folder="empty_folder"), [])
        mock_logger.info.assert_any_call("Found 0 available datasets with valid metadata in 'empty_folder'.")

    @patch('dataset_manager.os.path.isdir')
    @patch('dataset_manager.os.listdir')
    @patch('dataset_manager.os.path.exists')
    @patch('dataset_manager.get_dataset_info')
    @patch('dataset_manager.logger')
    def test_finds_valid_datasets(self, mock_logger, mock_get_info, mock_os_exists, mock_listdir, mock_isdir):
        mock_isdir.return_value = True
        mock_listdir.return_value = ["data1.json", "data2.json", "data1.json.meta", "otherfile.txt", "data3.json.lock"]
        def os_exists_side_effect(path): return path.endswith("data1.json.meta")
        mock_os_exists.side_effect = os_exists_side_effect
        meta1 = {"name": "Dataset1"}
        mock_get_info.side_effect = lambda df, folder: meta1 if df == "data1.json" else None
        result = list_available_datasets(folder="my_data")
        self.assertEqual(len(result), 1); self.assertEqual(result[0], meta1)
        mock_get_info.assert_called_once_with("data1.json", folder="my_data")

    @patch('dataset_manager.os.path.isdir')
    @patch('dataset_manager.os.listdir')
    @patch('dataset_manager.os.path.exists')
    @patch('dataset_manager.get_dataset_info')
    @patch('dataset_manager.logger')
    def test_ignores_meta_json_files_and_corrupt_meta(self, mock_logger, mock_get_info, mock_os_exists, mock_listdir, mock_isdir):
        mock_isdir.return_value = True
        mock_listdir.return_value = ["data1.json", "data1.json.meta", "data2.json", "data2.json.meta", "archive.meta.json"]
        mock_os_exists.return_value = True 
        meta1 = {"name": "Dataset1"}
        mock_get_info.side_effect = lambda df, folder: meta1 if df == "data1.json" else None
        result = list_available_datasets(folder="my_data")
        self.assertEqual(len(result), 1); self.assertEqual(result[0], meta1)
        mock_logger.warning.assert_called_once_with("Could not retrieve info for dataset 'data2.json' despite metadata file presence (metadata might be corrupt or invalid).")

class TestSaveDatasetJsonHelper(unittest.TestCase):

    @patch('dataset_manager.os.makedirs')
    @patch('dataset_manager.open', new_callable=mock_open)
    @patch('dataset_manager.json.dump')
    @patch('dataset_manager.logger')
    def test_save_success(self, mock_logger, mock_json_dump, mock_file_open, mock_makedirs):
        trials = [{"id": "NCT001", "data": "Test Data"}]
        filename = "output.json"
        folder = "test_output_folder"
        expected_filepath = os.path.join(folder, filename)

        result = _save_dataset_json(trials, filename, folder=folder)

        self.assertTrue(result)
        mock_makedirs.assert_called_once_with(folder, exist_ok=True)
        mock_file_open.assert_called_once_with(expected_filepath, 'w', encoding='utf-8')
        mock_json_dump.assert_called_once_with(trials, mock_file_open(), indent=4)
        mock_logger.info.assert_any_call(f"Attempting to save dataset to {expected_filepath}...")
        mock_logger.info.assert_any_call(f"Successfully saved {len(trials)} trials to {expected_filepath}.")

    @patch('dataset_manager.os.makedirs')
    @patch('dataset_manager.open', new_callable=mock_open)
    @patch('dataset_manager.json.dump')
    @patch('dataset_manager.logger')
    def test_save_empty_list(self, mock_logger, mock_json_dump, mock_file_open, mock_makedirs):
        trials = []
        filename = "empty_list.json"
        folder = "test_output_folder"
        expected_filepath = os.path.join(folder, filename)

        result = _save_dataset_json(trials, filename, folder=folder)

        self.assertTrue(result) # Saving an empty list is considered a success
        mock_makedirs.assert_called_once_with(folder, exist_ok=True)
        mock_file_open.assert_called_once_with(expected_filepath, 'w', encoding='utf-8')
        mock_json_dump.assert_called_once_with(trials, mock_file_open(), indent=4)
        mock_logger.info.assert_any_call(f"Successfully saved 0 trials to {expected_filepath}.")


    @patch('dataset_manager.logger')
    def test_save_no_filename(self, mock_logger):
        trials = [{"id": "NCT001"}]
        result = _save_dataset_json(trials, filename="") # Empty filename
        self.assertFalse(result)
        mock_logger.error.assert_called_once_with("Filename not provided for saving dataset. Aborting save.")

    @patch('dataset_manager.os.makedirs')
    @patch('dataset_manager.open', new_callable=mock_open)
    @patch('dataset_manager.logger')
    def test_save_io_error(self, mock_logger, mock_file_open, mock_makedirs):
        trials = [{"id": "NCT001"}]
        filename = "io_error.json"
        folder = "test_output_folder"
        expected_filepath = os.path.join(folder, filename)
        mock_file_open.side_effect = IOError("Disk full")

        result = _save_dataset_json(trials, filename, folder=folder)
        
        self.assertFalse(result)
        mock_makedirs.assert_called_once_with(folder, exist_ok=True) # makedirs is called before open
        mock_logger.error.assert_called_once_with(f"Error writing dataset to {expected_filepath}: Disk full")

    @patch('dataset_manager.os.makedirs')
    @patch('dataset_manager.open', new_callable=mock_open)
    @patch('dataset_manager.json.dump')
    @patch('dataset_manager.logger')
    def test_save_type_error(self, mock_logger, mock_json_dump, mock_file_open, mock_makedirs):
        # trials containing non-serializable data (like a datetime object without a custom handler)
        trials = [{"id": "NCT001", "timestamp": datetime.now()}] 
        filename = "type_error.json"
        folder = "test_output_folder"
        expected_filepath = os.path.join(folder, filename)
        
        # Configure json.dump mock to raise TypeError
        mock_json_dump.side_effect = TypeError("Object of type datetime is not JSON serializable")

        result = _save_dataset_json(trials, filename, folder=folder)

        self.assertFalse(result)
        mock_makedirs.assert_called_once_with(folder, exist_ok=True)
        mock_file_open.assert_called_once_with(expected_filepath, 'w', encoding='utf-8')
        mock_logger.error.assert_called_once_with(f"TypeError: Data for {expected_filepath} may not be JSON serializable: Object of type datetime is not JSON serializable")


if __name__ == '__main__':
    unittest.main()
