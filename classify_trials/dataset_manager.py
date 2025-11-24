"""Manages clinical trial datasets, including loading, merging, deduplication, and metadata handling."""

import json
import os
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any

# Configure logging (similar to fetch_trials.py)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

def _save_dataset_json(trials: List[Dict], filename: str, folder: str = "pulled_data") -> bool:
    """
    Saves a list of trial data to a JSON file in a specified folder.
    Internal helper function.

    Args:
        trials: List of dictionaries containing trial information.
        filename: Name of the file to save data to (e.g., "my_trials.json").
        folder: The directory to save the file in. Defaults to "pulled_data".

    Returns:
        True if saving was successful, False otherwise.
    """
    # Saving an empty list of trials is a valid operation.
    # if not trials: 
    #     logger.warning(f"No trials provided to save for {filename} in folder {folder}. File will not be created.")
    #     return False # Or True if an empty file is desired and successfully created.

    if not filename:
        logger.error("Filename not provided for saving dataset. Aborting save.")
        return False

    filepath = os.path.join(folder, filename)
    logger.info(f"Attempting to save dataset to {filepath}...")

    try:
        os.makedirs(folder, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(trials, f, indent=4) # Using indent=4 for readability
        count_to_log = len(trials) if trials is not None else 0 # Handle if trials is None, though type hint is List[Dict]
        logger.info(f"Successfully saved {count_to_log} trials to {filepath}.")
        return True
    except (IOError, OSError) as e: # Catch file/OS related errors
        logger.error(f"Error writing dataset to {filepath}: {e}")
        return False
    except TypeError as e: # For non-serializable data
        logger.error(f"TypeError: Data for {filepath} may not be JSON serializable: {e}")
        return False
    except Exception as e: # Catch any other unexpected errors
        logger.error(f"An unexpected error occurred while saving dataset to {filepath}: {e}")
        return False

def _prepare_metadata_for_saving(
    trials: List[Dict], 
    data_filepath: str, # Full path to the data file
    search_params: Dict
) -> Dict:
    """
    Prepares a metadata dictionary for a dataset.

    Args:
        trials: The list of trial dictionaries saved to the data file.
        data_filepath: The full path to the saved data file (for size calculation).
        search_params: A dictionary of parameters used to fetch the dataset.
                       Expected keys include "query_term", "sponsor", "phases",
                       "limit_requested", and "statuses".

    Returns:
        A dictionary structured for dataset metadata.
    """
    logger.info(f"Preparing metadata for dataset at {data_filepath}...")

    file_size = -1 # Default if size cannot be determined
    try:
        if os.path.exists(data_filepath): # Check if file exists before getting size
            file_size = os.path.getsize(data_filepath)
        else:
            logger.warning(f"Data file {data_filepath} not found for size calculation. Size will be set to -1.")
    except OSError as e: # Catch potential OS errors from getsize or exists
        logger.warning(f"Could not get file size for {data_filepath} due to OSError: {e}. Size will be set to -1.")


    metadata = {
        "created": datetime.now(timezone.utc).isoformat(),
        "query_term": search_params.get("query_term"),
        "sponsor": search_params.get("sponsor"),
        "phases": search_params.get("phases"),
        "statuses": search_params.get("statuses"),
        "limit_requested": search_params.get("limit_requested"),
        "actual_count": len(trials) if trials is not None else 0, # Handle if trials is None
        "source": "ClinicalTrials.gov API v2", 
        "file_size_bytes": file_size
    }
    
    logger.info(f"Metadata prepared for {data_filepath}: {metadata['actual_count']} trials, size {metadata['file_size_bytes']} bytes.")
    return metadata

def load_dataset(filename: str, folder: str = "pulled_data") -> Optional[List[Dict]]:
    """
    Loads a dataset from a specified JSON file.

    Args:
        filename: The name of the JSON file to load (e.g., "my_trials.json").
        folder: The directory where the file is located. Defaults to "pulled_data".

    Returns:
        A list of dictionaries representing the trial data, or None if an error occurs
        (e.g., file not found, invalid JSON, data not a list).
    """
    filepath = os.path.join(folder, filename)
    logger.info(f"Attempting to load dataset from {filepath}...")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        if not isinstance(dataset, list):
            logger.error(f"Invalid data format in {filepath}: Expected a list, got {type(dataset).__name__}.")
            return None
        
        # Optional: Deeper validation if all items in list are dicts
        # for i, item in enumerate(dataset):
        #     if not isinstance(item, dict):
        #         logger.error(f"Invalid item format in {filepath} at index {i}: Expected a dict, got {type(item).__name__}.")
        #         return None
            
        logger.info(f"Successfully loaded {len(dataset)} records from {filepath}.")
        return dataset
    except FileNotFoundError:
        logger.error(f"Dataset file not found: {filepath}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {filepath}: {e}")
        return None
    except IOError as e: # Catches other file I/O errors
        logger.error(f"IOError when trying to read {filepath}: {e}")
        return None
    except Exception as e: # Generic catch-all for other unexpected errors
        logger.error(f"An unexpected error occurred while loading dataset from {filepath}: {e}")
        return None

def deduplicate_trials(trials: List[Dict]) -> List[Dict]:
    """
    Removes duplicate trials from a list based on 'nct_id'.

    Keeps the first occurrence of each trial and discards subsequent duplicates.
    Trials without an 'nct_id', with an empty 'nct_id', or items that are not 
    dictionaries are logged and skipped.

    Args:
        trials: A list of trial dictionaries. Each dictionary is expected to have
                an 'nct_id' key.

    Returns:
        A new list of trial dictionaries with duplicates removed.
    """
    if not trials: # Handles None or empty list
        logger.info("Received no trials or an empty list for deduplication.")
        return []

    logger.info(f"Starting deduplication for {len(trials)} trials...")
    
    seen_nct_ids = set()
    deduplicated_list = []
    duplicates_count = 0
    skipped_count = 0

    for i, trial in enumerate(trials):
        if not isinstance(trial, dict):
            logger.warning(f"Item at index {i} is not a dictionary (type: {type(trial).__name__}), skipping.")
            skipped_count +=1
            continue

        nct_id = trial.get("nct_id")

        if not nct_id: # Handles None or empty string nct_id values
            # Try to get some other identifier for logging, like brief_title or official_title
            trial_identifier = trial.get('brief_title') or trial.get('official_title', f"Unknown trial at index {i}")
            logger.warning(f"Trial '{trial_identifier}' is missing 'nct_id' or 'nct_id' is empty. Skipping.")
            skipped_count +=1
            continue

        if nct_id not in seen_nct_ids:
            seen_nct_ids.add(nct_id)
            deduplicated_list.append(trial)
        else:
            logger.info(f"Duplicate trial found with NCT ID '{nct_id}'. Skipping.")
            duplicates_count += 1
            
    logger.info(f"Deduplication complete. Original trials: {len(trials)}, "
                f"Deduplicated trials: {len(deduplicated_list)}, "
                f"Duplicates removed: {duplicates_count}, Items skipped: {skipped_count}.")
    return deduplicated_list

def load_dataset_metadata(data_filename: str, folder: str = "pulled_data") -> Optional[Dict]:
    """
    Loads metadata for a given dataset file.

    The metadata is expected to be in a JSON file named like the data file
    but with a '.meta' extension (e.g., data_file.json -> data_file.json.meta).

    Args:
        data_filename: The filename of the data JSON file (e.g., "my_trials.json").
        folder: The directory where the data and metadata files are located.
                Defaults to "pulled_data".

    Returns:
        A dictionary containing the dataset metadata, or None if an error occurs
        (e.g., metadata file not found, invalid JSON, metadata not a dict).
    """
    metadata_filename = data_filename + ".meta"
    filepath = os.path.join(folder, metadata_filename)
    logger.info(f"Attempting to load dataset metadata from {filepath} (for data file {data_filename})...")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        if not isinstance(metadata, dict):
            logger.error(f"Invalid metadata format in {filepath}: Expected a dictionary, got {type(metadata).__name__}.")
            return None
            
        logger.info(f"Successfully loaded metadata from {filepath}.")
        return metadata
    except FileNotFoundError:
        # Logging as warning because metadata might be optional or created later.
        logger.warning(f"Metadata file not found: {filepath} (for data file {data_filename})")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from metadata file {filepath}: {e}")
        return None
    except IOError as e:
        logger.error(f"IOError when trying to read metadata file {filepath}: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading metadata from {filepath}: {e}")
        return None

def save_dataset_metadata(metadata: Dict, data_filename: str, folder: str = "pulled_data") -> None:
    """
    Saves the provided metadata dictionary to a .meta file.

    The metadata file is named based on the data_filename (e.g.,
    data_file.json -> data_file.json.meta). The metadata is saved as JSON.

    Args:
        metadata: The dictionary of metadata to save.
        data_filename: The filename of the corresponding data file.
        folder: The directory to save the metadata file in. Defaults to "pulled_data".
    """
    if not metadata or not isinstance(metadata, dict):
        logger.error("Invalid metadata provided; must be a non-empty dictionary. Metadata not saved.")
        return

    metadata_filename = data_filename + ".meta"
    filepath = os.path.join(folder, metadata_filename)
    
    logger.info(f"Attempting to save dataset metadata to {filepath} (for data file {data_filename})...")

    try:
        os.makedirs(folder, exist_ok=True) # Ensure directory exists
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4) # Use indent for readability
        logger.info(f"Successfully saved metadata to {filepath}.")
    except IOError as e:
        logger.error(f"IOError when trying to write metadata to {filepath}: {e}")
        # Depending on desired behavior, could re-raise e or handle more specifically
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving metadata to {filepath}: {e}")
        # Depending on desired behavior, could re-raise e

def get_dataset_info(data_filename: str, folder: str = "pulled_data") -> Optional[Dict]:
    """
    Retrieves information (metadata) for a specified dataset.

    This function loads metadata from the corresponding '.meta' file.
    The metadata should contain details like creation date, source parameters,
    file size, and actual record count.

    Args:
        data_filename: The filename of the data JSON file (e.g., "my_trials.json").
        folder: The directory where the data and metadata files are located.
                Defaults to "pulled_data".

    Returns:
        A dictionary containing the dataset's metadata if found and loaded
        successfully, otherwise None.
    """
    logger.info(f"Attempting to get dataset info for {data_filename} in folder '{folder}'...")
    
    metadata = load_dataset_metadata(data_filename, folder=folder)
    
    if metadata:
        logger.info(f"Successfully retrieved dataset info (metadata) for {data_filename}.")
        return metadata
    else:
        # load_dataset_metadata already logs a warning if file not found, or error for other issues.
        logger.info(f"Could not retrieve dataset info for {data_filename} as its metadata was not found, is invalid, or an error occurred during loading.")
        return None

def list_available_datasets(folder: str = "pulled_data") -> List[Dict]:
    """
    Lists available datasets in the specified folder by finding .json files
    that have corresponding .json.meta metadata files.

    Args:
        folder: The directory to scan for datasets. Defaults to "pulled_data".

    Returns:
        A list of dictionaries, where each dictionary is the metadata
        (dataset information) for an available dataset. Returns an empty
        list if the folder doesn't exist or no valid datasets are found.
    """
    logger.info(f"Scanning folder '{folder}' for available datasets...")
    
    if not os.path.isdir(folder):
        logger.warning(f"Dataset folder '{folder}' does not exist or is not a directory.")
        return []

    available_datasets_info = []
    try:
        for entry_name in os.listdir(folder):
            # Check for .json file, but exclude .meta.json and other potential interfering .json files
            if entry_name.endswith(".json") and not entry_name.endswith(".meta.json") and not entry_name.endswith(".json.lock"): # Example exclusion
                data_filename = entry_name
                metadata_filename = data_filename + ".meta"
                # Construct full path for os.path.exists check, not just for get_dataset_info
                metadata_filepath = os.path.join(folder, metadata_filename)

                if os.path.exists(metadata_filepath):
                    logger.debug(f"Found potential dataset: {data_filename} with metadata file: {metadata_filename}")
                    dataset_info = get_dataset_info(data_filename, folder=folder)
                    if dataset_info:
                        available_datasets_info.append(dataset_info)
                    else:
                        logger.warning(f"Could not retrieve info for dataset '{data_filename}' despite metadata file presence (metadata might be corrupt or invalid).")
                else:
                    logger.debug(f"Data file {data_filename} found without a corresponding .meta file. Skipping.")
        
        logger.info(f"Found {len(available_datasets_info)} available datasets with valid metadata in '{folder}'.")
        return available_datasets_info
        
    except FileNotFoundError: 
        logger.error(f"Dataset folder '{folder}' became inaccessible after initial check during listdir.")
        return []
    except OSError as e:
        logger.error(f"OSError when trying to list datasets in '{folder}': {e}")
        return []

# Temporary helper _save_json_data_temp is now removed.

def merge_datasets(
    dataset_filenames: List[str], 
    output_filename: str, 
    data_folder: str = "pulled_data", 
    output_folder: str = "pulled_data"
) -> Optional[List[Dict]]:
    """
    Merges multiple datasets into a single dataset, deduplicating trials based on NCT ID.

    If duplicate NCT IDs are found across datasets, the trial from the dataset
    with the most recent 'created' timestamp in its metadata is kept.
    Trials without NCT IDs or from datasets without valid metadata/timestamps are skipped.

    Args:
        dataset_filenames: A list of data filenames to merge.
        output_filename: The filename for the new merged dataset (JSON).
        data_folder: Directory where input datasets and their metadata are located.
        output_folder: Directory where the merged dataset and its metadata will be saved.

    Returns:
        A list of merged and deduplicated trial dictionaries, or None if critical errors occur
        or if no valid datasets are provided.
    """
    if not dataset_filenames:
        logger.warning("No dataset filenames provided for merging. Returning None.")
        return None

    logger.info(f"Starting dataset merge process for {len(dataset_filenames)} files. Output to: {os.path.join(output_folder, output_filename)}")

    # Stores {nct_id: {'trial_data': Dict, 'dataset_created_timestamp': str}}
    merged_trials_info: Dict[str, Dict[str, Any]] = {} 
    processed_files_count = 0
    
    for filename in dataset_filenames:
        logger.info(f"Processing dataset file: {filename} in folder: {data_folder}")
        dataset = load_dataset(filename, folder=data_folder)
        if not dataset:
            logger.warning(f"Failed to load dataset {filename} from {data_folder}. Skipping.")
            continue

        metadata = load_dataset_metadata(filename, folder=data_folder)
        if not metadata:
            logger.warning(f"Failed to load metadata for {filename}. Skipping this dataset for merge.")
            continue
        
        dataset_created_ts = metadata.get("created")
        if not dataset_created_ts:
            logger.warning(f"Metadata for {filename} is missing 'created' timestamp. Skipping this dataset for merge.")
            continue
        
        processed_files_count += 1
        for i, trial in enumerate(dataset):
            if not isinstance(trial, dict):
                logger.warning(f"Item at index {i} in {filename} is not a dictionary. Skipping.")
                continue

            nct_id = trial.get("nct_id")
            if not nct_id:
                trial_identifier = trial.get('brief_title') or trial.get('official_title', f"Unknown trial at index {i} in {filename}")
                logger.warning(f"Trial '{trial_identifier}' in {filename} is missing 'nct_id'. Skipping.")
                continue

            current_trial_info_for_merge = {'trial_data': trial, 'dataset_created_timestamp': dataset_created_ts}

            if nct_id not in merged_trials_info:
                merged_trials_info[nct_id] = current_trial_info_for_merge
                logger.debug(f"Adding new trial {nct_id} from {filename} (dataset timestamp: {dataset_created_ts}).")
            else:
                existing_trial_info = merged_trials_info[nct_id]
                existing_ts_str = existing_trial_info['dataset_created_timestamp']
                
                # Compare ISO 8601 timestamps as strings (lexicographical comparison works)
                if dataset_created_ts > existing_ts_str:
                    merged_trials_info[nct_id] = current_trial_info_for_merge
                    logger.info(f"Updating trial {nct_id} with data from newer dataset {filename} (new ts: {dataset_created_ts}, old ts: {existing_ts_str}).")
                elif dataset_created_ts == existing_ts_str:
                    logger.info(f"Trial {nct_id} from {filename} has same dataset timestamp as existing ({dataset_created_ts}). Keeping first encountered.")
                else:
                    logger.info(f"Skipping trial {nct_id} from {filename} (dataset ts: {dataset_created_ts}) as existing trial is newer or same (ts: {existing_ts_str}).")

    if not merged_trials_info:
        logger.warning("No trials were eligible for merging after processing all files.")
        return None if processed_files_count > 0 else [] # Return None if files were processed but all empty/invalid, [] if no files processed

    final_merged_trials = [info['trial_data'] for info in merged_trials_info.values()]
    logger.info(f"Merge process complete. Total unique trials: {len(final_merged_trials)}.")

    # Save the merged dataset
    if _save_dataset_json(final_merged_trials, output_filename, output_folder): # Changed to _save_dataset_json
        # Prepare and save metadata for the merged dataset
        merged_data_filepath = os.path.join(output_folder, output_filename)
        search_params_for_merged = {
            "query_term": "merged_dataset", # Or a more descriptive name
            "source_files": dataset_filenames,
            "limit_requested": None, # Not applicable or sum if meaningful
            "sponsor": None, # Could be a list of unique sponsors or None
            "phases": None,  # Could be a list of unique phases or None
            "statuses": None # Could be a list of unique statuses or None
        }
        
        new_metadata = _prepare_metadata_for_saving(
            final_merged_trials, 
            merged_data_filepath, 
            search_params_for_merged
        )
        save_dataset_metadata(new_metadata, output_filename, folder=output_folder)
        logger.info(f"Merged dataset and its metadata saved to {output_folder}.")
    else:
        logger.error(f"Failed to save the merged dataset to {output_filename} in {output_folder}. Metadata will not be saved.")
        # Potentially return None here if saving is critical, or just the trials if partial success is okay
        # For now, returning trials even if saving failed, as data is processed.
        
    return final_merged_trials

if __name__ == '__main__':
    # This block can be used for basic testing or demonstration of the module later.
    logger.info(f"Module {__name__} loaded. Intended for dataset management operations.")
    # Example: Demonstrate that the logger is working
    # logger.warning("This is a sample warning from dataset_manager.py's main block.")
    # logger.error("This is a sample error from dataset_manager.py's main block.")
    pass
