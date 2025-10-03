"""
fetch_trials.py - Fetches and processes mRNA clinical trials from ClinicalTrials.gov API v2.

Functions:
- fetch_clinical_trials: Gets mRNA trial data from the API
- extract_trial_info: Extracts key details from each trial
- save_trials: Saves processed trials to JSON
- fetch_and_save_trials: Main function combining the above steps

Example usage:
    from fetch_trials import fetch_and_save_trials

    # Fetch mRNA trials with custom limit
    trials = fetch_and_save_trials(limit=100, output_file="mrna_trials.json")
"""

import requests
import json
import os
import time
from typing import Dict, List, Optional 
import logging 
import requests 
import csv 
from dataset_manager import _prepare_metadata_for_saving, save_dataset_metadata # Added

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_STATUSES = ["NOT_YET_RECRUITING", "RECRUITING", "ACTIVE_NOT_RECRUITING"]

def fetch_clinical_trials(query_term: str,
                          limit: int = 20,
                          statuses: Optional[List[str]] = None,
                          sponsor: Optional[str] = None,
                          phase: Optional[List[str]] = None) -> Optional[requests.Response]:
    """
    Fetch clinical trials from ClinicalTrials.gov API v2.

    Args:
        query_term: The main search query term (e.g., "mRNA", "cancer vaccine"). Mandatory.
        limit: Number of trials to fetch (default: 20). Valid range 1-1000.
        statuses: List of study statuses to include. Defaults to active/recruiting if None.
        sponsor: Optional name of the sponsor to filter by.
        phase: Optional list of study phases (e.g., ["PHASE1", "PHASE2"]) to filter by.

    Returns:
        requests.Response object if successful (status 200), None otherwise.
    """
    if not isinstance(limit, int):
        logging.warning("Limit parameter must be an integer. Using default of 20.")
        limit = 20
    if limit < 1:
        logging.warning(f"Limit {limit} is less than 1. Setting to 1.")
        limit = 1
    elif limit > 1000:
        logging.warning(f"Limit {limit} exceeds maximum of 1000. Setting to 1000.")
        limit = 1000

    # Use provided statuses or default if None
    current_statuses = statuses if statuses is not None else DEFAULT_STATUSES
    
    logging.info(f"Fetching up to {limit} clinical trials for query: '{query_term}' with statuses: {','.join(current_statuses)}")

    url = "https://clinicaltrials.gov/api/v2/studies"

    params = {
        "format": "json",
        "query.term": query_term,
        "pageSize": str(limit),
        "countTotal": "true",
        "filter.advanced": "AREA[StudyType]Interventional" 
    }

    if current_statuses: # Add statuses if list is not empty
        params["filter.overallStatus"] = ",".join(current_statuses)
    
    if sponsor:
        params["query.spons"] = sponsor
    
    if phase and isinstance(phase, list) and phase: # Ensure phase is a non-empty list
        params["filter.phase"] = ",".join(phase)

    try:
        response = requests.get(url, params=params)
        logging.info(f"Request URL: {response.url}")
        logging.info(f"Response status code: {response.status_code}")

        if response.status_code == 200:
            return response
        else:
            logging.error(f"API error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {e}")
        return None

def extract_trial_info(study: Dict) -> Optional[Dict]:
    """
    Extract relevant information from a clinical trial study.

    Args:
        study: Dictionary containing study data

    Returns:
        Dictionary with extracted trial information if successful, None otherwise
    """
    if "protocolSection" not in study:
        logging.warning(f"Warning: Study missing protocolSection. Keys: {study.keys()}")
        return None

    protocol = study["protocolSection"]

    # Extract identification information
    identification = protocol.get("identificationModule", {})
    nct_id = identification.get("nctId", "Unknown ID")
    official_title = identification.get("officialTitle", "No official title")
    brief_title = identification.get("briefTitle", "No brief title")

    # Extract status information
    status_module = protocol.get("statusModule", {})
    overall_status = status_module.get("overallStatus", "Unknown status")
    start_date = status_module.get("startDateStruct", {}).get("date", "Unknown")
    completion_date = status_module.get("completionDateStruct", {}).get("date", "Unknown")

    # Extract description information
    description = protocol.get("descriptionModule", {})
    brief_summary = description.get("briefSummary", "No summary")
    detailed_description = description.get("detailedDescription", "No detailed description")

    # Extract sponsor information
    sponsor_module = protocol.get("sponsorCollaboratorsModule", {})
    lead_sponsor = sponsor_module.get("leadSponsor", {}).get("name", "Unknown sponsor")
    collaborators = [collab.get("name", "Unknown collaborator") 
                    for collab in sponsor_module.get("collaborators", [])]

    # Extract condition information
    conditions_module = protocol.get("conditionsModule", {})
    conditions = conditions_module.get("conditions", [])

    # Extract intervention information
    interventions_module = protocol.get("interventionsModule", {})
    interventions = interventions_module.get("interventions", [])
    intervention_details = []

    for intervention in interventions:
        intervention_details.append({
            "type": intervention.get("interventionType", "Unknown type"),
            "name": intervention.get("interventionName", "Unknown name"),
            "description": intervention.get("description", "No description")
        })

    # Extract design information
    design_module = protocol.get("designModule", {})
    phases_data = design_module.get("phases") 
    # Corrected logic for phases: if phases_data is an empty list or None, it should default to ["Unknown phase"]
    phases = phases_data if phases_data and isinstance(phases_data, list) and len(phases_data) > 0 else ["Unknown phase"]
    study_type = design_module.get("studyType", "Unknown type")

    return {
        "nct_id": nct_id,
        "brief_title": brief_title,
        "official_title": official_title,
        "overall_status": overall_status,
        "start_date": start_date,
        "completion_date": completion_date,
        "brief_summary": brief_summary,
        "detailed_description": detailed_description,
        "sponsor": lead_sponsor,
        "collaborators": collaborators,
        "conditions": conditions,
        "interventions": intervention_details,
        "phases": phases,
        "study_type": study_type
    }

def process_trials_data(raw_api_data: Dict) -> List[Dict]:
    """
    Extracts and processes trial information from the raw API response data.

    Args:
        raw_api_data: A dictionary representing the parsed JSON response from the ClinicalTrials.gov API.
                      Expected to contain a "studies" key with a list of trial objects.

    Returns:
        A list of dictionaries, where each dictionary contains cleaned information for a single trial.
        Returns an empty list if no studies are found or if input data is invalid.
    """
    if not raw_api_data or "studies" not in raw_api_data or raw_api_data["studies"] is None:
        logging.warning("No studies found in raw API data or data is invalid for processing.")
        return []

    studies_raw = raw_api_data.get("studies", [])
    total_raw_studies = len(studies_raw)
    logging.info(f"Received {total_raw_studies} studies from API response for processing.")

    processed_trials = []
    for i, study_raw_item in enumerate(studies_raw): # Renamed to avoid conflict with 'study' module if imported elsewhere
        # The extract_trial_info function expects the 'study' object which is each item in the 'studies' list
        trial_info = extract_trial_info(study_raw_item) 
        if trial_info:
            processed_trials.append(trial_info)
        else:
            nct_id_default = f"Unknown ID at index {i}"
            nct_id = nct_id_default
            if isinstance(study_raw_item, dict):
                # Safely navigate the dictionary using chained .get()
                protocol_section = study_raw_item.get("protocolSection")
                if isinstance(protocol_section, dict):
                    identification_module = protocol_section.get("identificationModule")
                    if isinstance(identification_module, dict):
                        nct_id = identification_module.get("nctId", nct_id_default)
            
            logging.warning(f"Failed to extract information for trial {nct_id}. Skipping.")
    
    logging.info(f"Successfully processed {len(processed_trials)} out of {total_raw_studies} trials.")
    return processed_trials

def save_trials_json(trials: List[Dict], filename: str, folder: str = "pulled_data") -> None:
    """
    Save trial data to a JSON file in a specified folder.

    Args:
        trials: List of dictionaries containing trial information.
        filename: Name of the file to save data to (e.g., "my_trials.json").
        folder: The directory to save the file in. Defaults to "pulled_data".
    """
    if not trials:
        logging.warning(f"No trials provided to save for {filename} in folder {folder}.")
        return

    filepath = "" # Initialize for logging in case of early error before filepath is set
    try:
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)
        logging.info(f"Attempting to save {len(trials)} trials to {filepath}...")
        with open(filepath, "w", encoding='utf-8') as f: # Added encoding='utf-8'
            json.dump(trials, f, indent=2)
        logging.info(f"Trials saved successfully to {filepath}")
    except IOError as e:
        logging.error(f"Error saving trials to {filepath}: {e}")
    except Exception as e: # Catch other potential errors
        logging.error(f"An unexpected error occurred while saving trials to {filepath}: {e}")

def save_trials_csv(trials: List[Dict], filename: str, folder: str = "pulled_data") -> None:
    """
    Saves a list of trial data to a CSV file in a specified folder.

    Args:
        trials: List of dictionaries, where each dictionary represents a trial.
        filename: Name of the CSV file (e.g., "my_trials.csv").
                  If .csv extension is missing, it will be added.
        folder: The directory to save the file in. Defaults to "pulled_data".
    """
    if not trials:
        logging.warning(f"No trials provided to save for CSV {filename} in folder {folder}.")
        return

    if not filename.lower().endswith('.csv'):
        original_filename = filename
        filename += '.csv'
        logging.info(f"Filename '{original_filename}' did not end with .csv, appended extension to '{filename}'.")

    filepath = "" # Initialize for logging in case of early error
    try:
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)
        logging.info(f"Attempting to save {len(trials)} trials to CSV file {filepath}...")

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            # Ensure trials is not empty again just before accessing trials[0]
            if not trials: # This check is technically redundant due to the one at the start
                logging.warning(f"Trial list became empty before writing headers for {filepath}.")
                return

            headers = list(trials[0].keys()) 
            headers.sort() # Sort headers for consistent column order

            writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore', quoting=csv.QUOTE_NONNUMERIC)
            writer.writeheader()
            for trial in trials:
                writer.writerow(trial)
        logging.info(f"Trials saved successfully to CSV file {filepath}")

    except IOError as e:
        logging.error(f"Error saving trials to CSV {filepath}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while saving trials to CSV {filepath}: {e}")

def load_trials_json(filename: str, folder: str = "pulled_data") -> Optional[List[Dict]]:
    """
    Loads trial data from a specified JSON file.

    Args:
        filename: The name of the JSON file to load (e.g., "my_trials.json").
        folder: The directory where the file is located. Defaults to "pulled_data".

    Returns:
        A list of dictionaries containing trial data, or None if an error occurs.
    """
    filepath = os.path.join(folder, filename)
    logging.info(f"Attempting to load trials from {filepath}...")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            trials_data = json.load(f)
        
        if not isinstance(trials_data, list):
            logging.error(f"Invalid data format in {filepath}: Expected a list of trials, got {type(trials_data)}.")
            return None
        
        # Optional: Deeper validation if items in list are dicts can be added here if necessary
        # for item in trials_data:
        #     if not isinstance(item, dict):
        #         logging.error(f"Invalid item format in {filepath}: Expected a dict, got {type(item)}.")
        #         return None

        logging.info(f"Successfully loaded {len(trials_data)} trials from {filepath}.")
        return trials_data
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {filepath}: {e}")
        return None
    except IOError as e: # Broader than FileNotFoundError, good to have for other read issues
        logging.error(f"IOError when trying to read {filepath}: {e}")
        return None
    except Exception as e: # Catch-all for other unexpected errors
        logging.error(f"An unexpected error occurred while loading {filepath}: {e}")
        return None

def fetch_and_save_trials(
    query_term: str = "mRNA",
    limit: int = 20,
    statuses: Optional[List[str]] = None,
    output_file: str = "trials_output.json", # This is the JSON filename
    sponsor: Optional[str] = None,
    phase: Optional[List[str]] = None,
    save_csv: bool = False  # New parameter
) -> List[Dict]:
    """
    Orchestrates fetching, processing, and saving clinical trial data.

    Args:
        query_term: The primary search term for the clinical trials API.
        limit: Number of trials to fetch.
        statuses: List of study statuses to include. Defaults to active/recruiting.
        output_file: Filename for the output JSON file (e.g., "trials.json").
                     The CSV filename will be derived from this if save_csv is True.
        sponsor: Optional sponsor to filter by.
        phase: Optional list of trial phases to filter by.
        save_csv: If True, also saves the processed trials to a CSV file.

    Returns:
        A list of processed trial dictionaries. Returns an empty list if errors occur.
    """
    # Use module-level DEFAULT_STATUSES, no need to redefine here
    current_statuses = statuses if statuses is not None else DEFAULT_STATUSES

    logging.info(
        f"Starting trial fetch and save process: query='{query_term}', limit={limit}, "
        f"json_output='{output_file}', save_csv={save_csv}"
    )

    # 1. Fetch
    logging.info("Fetching clinical trials from API...")
    response = fetch_clinical_trials(
        query_term=query_term, limit=limit, statuses=current_statuses, sponsor=sponsor, phase=phase
    )

    if not response:
        # fetch_clinical_trials already logs specifics, so this message is slightly redundant but confirms the abortion.
        logging.error("API request failed or no response received (fetch_clinical_trials returned None). Aborting process.")
        return []
    
    # The following check is redundant because fetch_clinical_trials returns None if status_code is not 200.
    # if response.status_code != 200:
    #     logging.error(f"API request failed with status code {response.status_code}. Response text: {response.text[:500]}...")
    #     return []

    try:
        # Using requests.exceptions.JSONDecodeError as it's more specific for response.json()
        raw_api_data = response.json()
    except requests.exceptions.JSONDecodeError as e: # Corrected exception type
        logging.error(f"Failed to decode JSON from API response: {e}")
        return []

    # 2. Process
    logging.info("Processing trial data...")
    trials = process_trials_data(raw_api_data)

    if not trials:
        logging.info("No trials were processed. Output files will not be saved.")
        return [] # Return empty list as no trials to save or return

    # 3. Save
    logging.info(f"Saving {len(trials)} processed trials to JSON: {output_file}")
    # Determine the actual save folder, aligning with save_trials_json/csv defaults if not overridden
    actual_data_save_folder = "pulled_data" 

    # save_trials_json uses its default folder "pulled_data" unless specified otherwise
    # For consistency, if fetch_and_save_trials were to manage folders, it would pass it here.
    # Since it doesn't, we use the known default for constructing paths for metadata.
    save_trials_json(trials, filename=output_file, folder=actual_data_save_folder) 

    # --- BEGIN METADATA SAVING ---
    try:
        full_data_filepath = os.path.join(actual_data_save_folder, output_file)
        
        search_params_for_meta = {
            "query_term": query_term, 
            "sponsor": sponsor,       
            "phases": phase,          
            "limit_requested": limit, 
            "statuses": current_statuses 
        }

        logging.info(f"Preparing metadata for data file '{output_file}' with search parameters: {search_params_for_meta}")
        metadata_to_save = _prepare_metadata_for_saving(
            trials=trials,
            data_filepath=full_data_filepath,
            search_params=search_params_for_meta
        )
        
        logging.info(f"Saving metadata for data file '{output_file}'...")
        save_dataset_metadata(
            metadata=metadata_to_save,
            data_filename=output_file, 
            folder=actual_data_save_folder 
        )
    except Exception as e:
        logging.error(f"Error occurred during metadata saving for '{output_file}': {e}", exc_info=True)
        # Continue without re-raising, as main data is saved.
    # --- END METADATA SAVING ---

    if save_csv:
        base_filename, _ = os.path.splitext(output_file)
        csv_filename = base_filename + ".csv"
        logging.info(f"Saving {len(trials)} processed trials to CSV: {csv_filename} in folder {actual_data_save_folder}")
        save_trials_csv(trials, filename=csv_filename, folder=actual_data_save_folder) 
    
    logging.info("Trial fetch and save process completed.")
    return trials

if __name__ == "__main__":
    logging.info("Script started in main execution block.")

    # --- Configuration Variables ---
    # Modify these to change search parameters
    query_term_main = "mRNA"      # Primary search term for trials
    sponsor_name = "BioNTech SE"  # Set to None or a string for sponsor/collaborator filter
    trial_phases = ["PHASE2"]     # Set to None or a list of strings like ["PHASE1", "PHASE2"] to filter by phase
    search_limit = 25             # Integer for the number of trials to fetch
    
    # Define a list of "active" trial statuses. Set to None to use the default statuses 
    # defined in fetch_clinical_trials (which is DEFAULT_STATUSES at module level).
    # Example: active_trial_statuses = ["NOT_YET_RECRUITING", "RECRUITING", "ACTIVE_NOT_RECRUITING", "ENROLLING_BY_INVITATION"]
    trial_statuses = None         # Using None here will apply the default statuses

    also_save_csv = True          # Set to True to also save a CSV file in addition to JSON

    # --- Dynamic Filename Generation ---
    def sanitize_filename_part(part_text: Optional[str]) -> str:
        """Helper to sanitize parts of a filename."""
        if not part_text:
            return ""
        # Basic sanitizer: allow alphanumeric, underscore, hyphen; replace space with underscore.
        # For more robust sanitization, consider a library or more comprehensive regex.
        sanitized = "".join(c for c in part_text if c.isalnum() or c in [" ", "_", "-"]).strip()
        sanitized = sanitized.replace(" ", "_")
        return sanitized

    filename_components = ["trials"] # Base prefix
    if query_term_main and query_term_main.lower() != "mrna": # Add if not default/generic
        filename_components.append(sanitize_filename_part(query_term_main))
    elif not query_term_main: # Should not happen with current setup as query_term is mandatory for fetch_clinical_trials
         filename_components.append("all")


    if sponsor_name:
        filename_components.append(sanitize_filename_part(sponsor_name))
    
    if trial_phases: # Add phases to filename if specified
        filename_components.append("phases_" + "_".join(sanitize_filename_part(p) for p in trial_phases))

    filename_components.append(str(search_limit)) # Add limit to filename

    # Construct the base filename, removing any empty parts resulting from None values
    base_output_filename = "_".join(filter(None, filename_components))
    json_output_filename = base_output_filename + ".json"
    
    logging.info(f"Generated JSON output filename: {json_output_filename}")
    if also_save_csv:
        csv_filename_derived = base_output_filename + ".csv"
        logging.info(f"CSV output filename will be: {csv_filename_derived} (saved in the same folder as JSON)")


    # --- Execute Fetch and Save ---
    logging.info(f"Executing fetch_and_save_trials with configured parameters...")
    processed_trials_main = fetch_and_save_trials(
        query_term=query_term_main,
        limit=search_limit,
        statuses=trial_statuses,
        output_file=json_output_filename, # This is for the JSON file
        sponsor=sponsor_name,
        phase=trial_phases,
        save_csv=also_save_csv
    )

    if processed_trials_main:
        logging.info(f"Successfully fetched and processed {len(processed_trials_main)} trials (main block execution).")
        
        # Example: Load the data back to verify (optional)
        # logging.info(f"Attempting to load back the saved JSON file: {json_output_filename}")
        # loaded_trials = load_trials_json(filename=json_output_filename) # Uses default folder "pulled_data"
        # if loaded_trials:
        #     logging.info(f"Successfully loaded back {len(loaded_trials)} trials from {json_output_filename}.")
        # else:
        #     logging.warning(f"Could not load back trials from {json_output_filename}.")
    else:
        logging.warning("Main execution block: No trials were processed or saved. This might be due to API errors or no trials matching criteria.")
    
    logging.info("Script finished main execution block.")