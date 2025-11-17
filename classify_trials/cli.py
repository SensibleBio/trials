"""
cli.py - Command-line interface for fetching and classifying mRNA clinical trials.

This script provides two main modes:
1. `full`: Fetches trials for a main query term and optionally for a list of companies,
   merges, deduplicates, and then classifies the trials using an LLM.
2. `classify-only`: Classifies trials from a user-defined input file (JSON or ICTRP XML).
"""

import os
import sys
import logging
import argparse
import json
from typing import Dict, List, Any, Optional

# Optional: Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass # python-dotenv is optional

# Initialize logger for cli.py
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import functions from other modules
from classify_trials.run_fetching_pipeline import run_pipeline
from classify_trials.llm_sort_trials import (
    get_llm_provider, get_provider_type, classify_multiple_trials,
    write_csv_from_results, parse_ictrp_xml
)
from classify_trials.prompt_manager import PromptManager, LLMProviderType

# Placeholder functions for original cli.py logic that are not defined elsewhere
# These would ideally be refactored into a separate module or integrated directly.
def classify_and_export_trials(
    trials: List[Dict],
    output_file: str,
    export_to_sheets: bool = False,
    llm_provider_name: str = "openai",
    delay: int = 0
) -> List[Dict]:
    """
    Placeholder: Classifies trials and exports them.
    In a real scenario, this would orchestrate LLM classification and Google Sheets export.
    For now, it just calls classify_multiple_trials.
    """
    logger.info("Using placeholder classify_and_export_trials.")
    
    # Only allow API keys via environment variables: OPEN_API_KEY or GEMINI_API_KEY
    api_env_var = "OPEN_API_KEY" if llm_provider_name == "openai" else "GEMINI_API_KEY"
    api_key = os.getenv(api_env_var)
    if not api_key:
        logger.error(f"API key not found in environment. Please set the {api_env_var} environment variable (or create a .env file).")
        sys.exit(1)
    
    llm = get_llm_provider(llm_provider_name, api_key)
    global prompt_manager
    prompt_manager = PromptManager(base_path="prompts", provider_type=get_provider_type(llm_provider_name))

    results = classify_multiple_trials(trials, output_file=output_file, delay=delay, llm_provider=llm)

    if export_to_sheets:
        logger.warning("Google Sheets export is not implemented in this placeholder function.")
    
    return results

def generate_classification_summary(results: List[Dict]) -> Dict[str, Any]:
    """Placeholder: Generates a summary of classification results."""
    logger.info("Using placeholder generate_classification_summary.")
    if results and "_metadata" in results[0]:
        return results[0]["_metadata"].get("category_counts", {})
    return {"No Category": len(results) - 1 if results else 0}

def print_classification_summary(categories: Dict[str, int]):
    """Placeholder: Prints the classification summary to the console."""
    logger.info("Using placeholder print_classification_summary.")
    logger.info("\n--- Classification Summary ---")
    for category, count in categories.items():
        logger.info(f"{category}: {count} trials")
    logger.info("----------------------------")

def main():
    parser = argparse.ArgumentParser(
        description='CLI for fetching and classifying mRNA clinical trials.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')

    # --- Full pipeline mode ---
    full_parser = subparsers.add_parser(
        'full',
        help='Run the full pipeline: fetch trials, then classify them.',
        description='Fetches trials for a main query term and optionally for a list of companies, '
                    'merges, deduplicates, and then classifies the trials using an LLM.'
    )
    full_parser.add_argument(
        '--query-term',
        type=str,
        default="mRNA",
        help='Main search query term for fetching (e.g., "mRNA"). Defaults to "mRNA".'
    )
    full_parser.add_argument(
        '--limit',
        type=int,
        default=100,
        help='Maximum number of trials to fetch per API call. Defaults to 100.'
    )
    full_parser.add_argument(
        '--companies-filepath',
        type=str,
        default="prompts/reference_data/mrna_companies.json",
        help='Path to JSON file containing company names for fetching. Defaults to "prompts/reference_data/mrna_companies.json".'
    )
    full_parser.add_argument(
        '--output-folder',
        type=str,
        default="pulled_data",
        help='Base directory to save fetched and merged data. Defaults to "pulled_data".'
    )
    full_parser.add_argument(
        '--merged-filename',
        type=str,
        default="merged_trials.json",
        help='Filename for the final merged dataset. Defaults to "merged_trials.json".'
    )
    full_parser.add_argument(
        '--save-company-files',
        type=lambda x: x.lower() == 'true',
        default=False,
        help='If True, saves individual company fetch results. Defaults to False.'
    )
    full_parser.add_argument(
        '--llm-provider',
        choices=['openai', 'gemini'],
        default='openai',
        help='LLM provider to use for classification (default: openai). '
               'Uses OPEN_API_KEY or GEMINI_API_KEY environment variable.'
    )
    full_parser.add_argument(
        '--no-save-csv',
        action='store_true',
        help='Do not save classification results to a CSV file.'
    )
    full_parser.add_argument(
        '--delay',
        type=int,
        default=0,
        help='Delay in seconds between LLM API calls during classification.'
    )

    # --- Classify-only mode ---
    classify_parser = subparsers.add_parser(
        'classify-only',
        help='Classify trials from an existing input file.',
        description='Classifies trials from a user-defined input file (JSON or ICTRP XML).'
    )
    classify_parser.add_argument(
        'input_file',
        type=str,
        help='Path to the input JSON or ICTRP XML file containing trials to classify.'
    )
    classify_parser.add_argument(
        'output_file',
        type=str,
        help='Path to the output JSON file to save classification results.'
    )
    classify_parser.add_argument(
        '--llm-provider',
        choices=['openai', 'gemini'],
        default='openai',
        help='LLM provider to use for classification (default: openai). '
             'Uses OPEN_API_KEY or GEMINI_API_KEY environment variable.'
    )
    classify_parser.add_argument(
        '--no-save-csv',
        action='store_true',
        help='Do not save classification results to a CSV file.'
    )
    classify_parser.add_argument(
        '--delay',
        type=int,
        default=0,
        help='Delay in seconds between LLM API calls.'
    )

    args = parser.parse_args()

    # Determine output folder for classification results (needed for both modes)
    # For 'full' mode, this will be within the timestamped folder created by run_pipeline
    # For 'classify-only', it's derived from the output_file
    classification_output_folder = None
    if args.mode == 'full':
        # The actual output folder for run_pipeline is timestamped, so we need to get it dynamically
        # For now, we'll set it to the base output folder for classification, and refine later.
        classification_output_folder = args.output_folder # This needs to be updated after run_pipeline
        classification_output_filename_base = args.merged_filename
    elif args.mode == 'classify-only':
        classification_output_folder = os.path.dirname(args.output_file)
        if not classification_output_folder: # If output_file is just a filename, assume current dir
            classification_output_folder = "."
        classification_output_filename_base = os.path.basename(args.output_file)

    if args.mode == 'full':
        logger.info("Running pipeline in FULL mode: fetching trials then classifying.")
        
        # --- Step 1: Fetch trials ---
        trials_to_classify, run_output_folder_path = run_pipeline(
            query_term=args.query_term,
            limit=args.limit,
            companies_filepath=args.companies_filepath,
            output_folder=args.output_folder,
            merged_output_filename=args.merged_filename,
            save_company_files=args.save_company_files
        )

        if not trials_to_classify:
            logger.error("No trials were fetched by the pipeline. Exiting classification.")
            sys.exit(1)

        # The output_file for classification will be based on the merged_filename from run_pipeline
        # and should be placed in the *same timestamped folder*.
        output_file_for_classification = os.path.join(
            run_output_folder_path,
            f"classified_{os.path.splitext(args.merged_filename)[0]}.json" # e.g., classified_merged_trials.json
        )

        logger.info(f"Starting classification for {len(trials_to_classify)} trials...")
        classification_results = classify_and_export_trials(
            trials=trials_to_classify,
            output_file=output_file_for_classification,
            export_to_sheets=False, # Google Sheets export not enabled for now
            llm_provider_name=args.llm_provider,
            delay=args.delay
        )

        if classification_results:
            logger.info("Generating console classification summary...")
            categories = generate_classification_summary(classification_results)
            print_classification_summary(categories)
            logger.info(f"Local classification results also saved to: {output_file_for_classification}")
            if not args.no_save_csv:
                write_csv_from_results(classification_results, output_file_for_classification)
        else:
            logger.error("Classification process did not return any results.")

    elif args.mode == 'classify-only':
        logger.info(f"Running pipeline in CLASSIFY-ONLY mode for input: {args.input_file}")

        # Load trials from input file (supports JSON and ICTRP XML)
        trials_to_classify = []
        if not os.path.exists(args.input_file):
            logger.error(f"Input file {args.input_file} not found. Please ensure it exists.")
            sys.exit(1)
        
        try:
            if args.input_file.lower().endswith('.xml'):
                logger.info(f"Detected XML input file: {args.input_file}, parsing as ICTRP export...")
                trials_to_classify = parse_ictrp_xml(args.input_file)
            else:
                with open(args.input_file, "r") as f:
                    trials_to_classify = json.load(f)
            logger.info(f"Loaded {len(trials_to_classify)} trials from {args.input_file}")
        except json.JSONDecodeError:
            logger.error(f"{args.input_file} is not a valid JSON file.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"An error occurred while loading {args.input_file}: {e}")
            sys.exit(1)

        if not trials_to_classify:
            logger.warning("No trials found in the input file. Exiting.")
            sys.exit(0)

        logger.info(f"Starting classification for {len(trials_to_classify)} trials...")
        classification_results = classify_and_export_trials(
            trials=trials_to_classify,
            output_file=args.output_file,
            export_to_sheets=False, # Google Sheets export not enabled for now
            llm_provider_name=args.llm_provider,
            delay=args.delay
        )

        if classification_results:
            logger.info("Generating console classification summary...")
            categories = generate_classification_summary(classification_results)
            print_classification_summary(categories)
            logger.info(f"Local classification results also saved to: {args.output_file}")
            if not args.no_save_csv:
                write_csv_from_results(classification_results, args.output_file)
        else:
            logger.error("Classification process did not return any results.")

    else:
        parser.print_help()

    logger.info("Process completed.")

if __name__ == "__main__":
    main()