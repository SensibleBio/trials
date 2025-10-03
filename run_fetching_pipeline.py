import sys
import json
import os
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any

# Add the project directory to sys.path to allow importing modules
# Assuming this script is run from the project root
sys.path.append('.')

from fetch_trials import fetch_and_save_trials, load_trials_json
from dataset_manager import merge_datasets, load_dataset, save_dataset_metadata, _prepare_metadata_for_saving, deduplicate_trials, _save_dataset_json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

def run_pipeline(
    query_term: str,
    limit: int,
    companies_filepath: str,
    output_folder: str,
    main_query_filename_prefix: str = "trials",
    merged_output_filename: str = "merged_trials.json",
    save_company_files: bool = False # Option to save individual company fetch files
):
    """
    Runs the data fetching and merging pipeline.

    1. Fetches trials for the main query term (e.g., "mRNA").
    2. Fetches trials for each company listed in the companies file.
    3. Merges and deduplicates all fetched datasets.
    4. Saves the final merged dataset and its metadata.

    Results for each run are saved in a new timestamped subfolder within the output_folder.

    Args:
        query_term (str): The main search query term (e.g., "mRNA").
        limit (int): The maximum number of trials to fetch per API call.
        companies_filepath (str): Path to the JSON file containing company names
                                   (e.g., "mrna_companies.json").
        output_folder (str): Base directory to save the fetched and merged data.
        main_query_filename_prefix (str): Prefix for the initial main query output file.
        merged_output_filename (str): Filename for the final merged dataset.
        save_company_files (bool): If True, saves individual company fetch results
                                   to temporary files (within the timestamped folder).
    """
    logger.info("Starting data fetching and merging pipeline.")
    
    # Create a timestamped directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_folder = os.path.join(output_folder, timestamp)
    os.makedirs(run_output_folder, exist_ok=True)
    logger.info(f"Saving all output for this run to: {run_output_folder}")

    # --- Step 1: Fetch trials for the main query term ---
    main_query_filename = f"{main_query_filename_prefix}.json"
    # The full path for saving will now include the run_output_folder

    logger.info(f"Fetching trials for main query '{query_term}'...")
    main_trials = fetch_and_save_trials(
        query_term=query_term,
        limit=limit,
        output_file=main_query_filename,
        folder=run_output_folder # Pass the timestamped folder
    )

    if not main_trials:
        logger.warning(f"No trials fetched for main query '{query_term}'. Proceeding with company fetches if any.")
        all_fetched_trials = []
    else:
         all_fetched_trials = main_trials

    # --- Step 2 & 3: Fetch trials for each company ---
    company_trial_files = []
    try:
        with open(companies_filepath, 'r', encoding='utf-8') as f:
            companies_data = json.load(f)
            if not isinstance(companies_data, dict) or "companies" not in companies_data or not isinstance(companies_data["companies"], list):
                 logger.error(f"Invalid format in {companies_filepath}. Expected a dict with a 'companies' list.")
                 companies_list = []
            else:
                companies_list = [c.get("name") for c in companies_data["companies"] if isinstance(c, dict) and c.get("name")]

    except FileNotFoundError:
        logger.error(f"Companies file not found at {companies_filepath}. Skipping company fetches.")
        companies_list = []
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {companies_filepath}: {e}. Skipping company fetches.")
        companies_list = []
    except Exception as e:
        logger.error(f"An unexpected error occurred while reading {companies_filepath}: {e}. Skipping company fetches.")
        companies_list = []

    if companies_list:
        logger.info(f"Found {len(companies_list)} companies to fetch trials for.")
        for company_name in companies_list:
            logger.info(f"Fetching trials for sponsor: '{company_name}'...")
            sanitized_company_name = company_name.replace(" ", "_").replace(".", "").replace(",", "") # Simple sanitization
            company_filename = f"{sanitized_company_name}_trials.json" # Filename without timestamp here
            
            company_trials = fetch_and_save_trials(
                query_term=None, # No general query term needed when filtering by sponsor
                limit=limit,
                sponsor=company_name,
                output_file=company_filename,
                folder=run_output_folder # Pass the timestamped folder
            )

            if company_trials:
                 logger.info(f"Fetched {len(company_trials)} trials for {company_name}.")
                 if save_company_files:
                      company_trial_files.append(company_filename) # Append just the filename relative to run_output_folder
                 else:
                      all_fetched_trials.extend(company_trials)
            else:
                 logger.warning(f"No trials fetched for sponsor '{company_name}'.")

    # --- Step 4 & 5: Merge and deduplicate datasets ---
    logger.info("Merging and deduplicating datasets...")

    if not save_company_files and all_fetched_trials:
         final_merged_trials = deduplicate_trials(all_fetched_trials)
         logger.info(f"Deduplication complete in memory. Final count: {len(final_merged_trials)}")

         # Save the final merged dataset in the timestamped folder
         merged_output_filepath = os.path.join(run_output_folder, merged_output_filename)
         logger.info(f"Saving final merged dataset to {merged_output_filepath}...")

         # Use the internal save function from dataset_manager for consistency with metadata saving
         if _save_dataset_json(final_merged_trials, merged_output_filename, folder=run_output_folder): # Pass timestamped folder
             try:
                 search_params_summary: Dict[str, Any] = {
                     "main_query_term": query_term,
                     "fetched_by_companies_count": len(companies_list),
                     "limit_per_fetch": limit,
                 }
                 # metadata will be saved in the run_output_folder
                 merged_metadata = _prepare_metadata_for_saving(
                     trials=final_merged_trials,
                     data_filepath=merged_output_filepath,
                     search_params=search_params_summary
                 )
                 save_dataset_metadata(merged_metadata, merged_output_filename, folder=run_output_folder) # Pass timestamped folder
                 logger.info(f"Merged dataset and metadata saved successfully to {run_output_folder}.")
             except Exception as e:
                  logger.error(f"Error saving metadata for merged dataset: {e}", exc_info=True)
                  logger.warning("Merged dataset was saved, but metadata saving failed.")
         else:
             logger.error(f"Failed to save the final merged dataset to {merged_output_filename} in {run_output_folder}. Metadata not saved.")

    elif save_company_files:
         # If company files were saved, use merge_datasets function
         # The merge_datasets function handles loading, merging, deduplication,
         # and saving the result and its metadata.
         # The filenames in company_trial_files are relative to run_output_folder
         all_filenames_to_merge = [main_query_filename] + company_trial_files # These are filenames *within* run_output_folder
         logger.info(f"Merging and deduplicating files within {run_output_folder}: {all_filenames_to_merge}")

         # merge_datasets expects dataset_filenames to be just the filenames, and needs the data_folder
         final_merged_trials = merge_datasets(
             dataset_filenames=all_filenames_to_merge,
             output_filename=merged_output_filename,
             data_folder=run_output_folder, # Pass the timestamped folder as the data source
             output_folder=run_output_folder # Pass the timestamped folder as the output destination
         )

         if final_merged_trials is None: # merge_datasets returns None on critical failure
             logger.error("Merging process failed. Final merged dataset not created.")
         else:
              # merge_datasets already saves the output and metadata if successful
             logger.info(f"Merging process completed. Final merged dataset saved to {os.path.join(run_output_folder, merged_output_filename)}")

    else:
         logger.warning("No trials were fetched or processed for merging.")
         final_merged_trials = []

    logger.info(f"Pipeline finished. Output saved to {run_output_folder}.")
    return final_merged_trials

if __name__ == "__main__":
    default_query = os.environ.get("PIPELINE_QUERY_TERM", "mRNA")
    default_limit = int(os.environ.get("PIPELINE_LIMIT", "100"))
    default_companies_file = os.environ.get("PIPELINE_COMPANIES_FILE", "prompts/reference_data/mrna_companies.json") # Default to the correct path
    default_output_folder = os.environ.get("PIPELINE_OUTPUT_FOLDER", "pulled_data")
    default_merged_filename = os.environ.get("PIPELINE_MERGED_FILENAME", "merged_trials.json")
    default_save_company_files = os.environ.get("PIPELINE_SAVE_COMPANY_FILES", "False").lower() == "true"

    args = sys.argv[1:]
    query_term = args[0] if len(args) > 0 else default_query
    limit = int(args[1]) if len(args) > 1 else default_limit
    companies_filepath = args[2] if len(args) > 2 else default_companies_file
    output_folder = args[3] if len(args) > 3 else default_output_folder
    merged_output_filename = args[4] if len(args) > 4 else default_merged_filename
    save_company_files = args[5].lower() == "true" if len(args) > 5 else default_save_company_files

    logger.info(f"Running pipeline with parameters:")
    logger.info(f"  Query Term: {query_term}")
    logger.info(f"  Limit Per Fetch: {limit}")
    logger.info(f"  Companies File: {companies_filepath}")
    logger.info(f"  Output Folder (Base): {output_folder}")
    logger.info(f"  Merged Output Filename: {merged_output_filename}")
    logger.info(f"  Save Individual Company Files: {save_company_files}")

    run_pipeline(
        query_term=query_term,
        limit=limit,
        companies_filepath=companies_filepath,
        output_folder=output_folder,
        merged_output_filename=merged_output_filename,
        save_company_files=save_company_files
    ) 