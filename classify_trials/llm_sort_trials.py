"""
classify_trials.py - Module for classifying mRNA clinical trials using various LLM APIs.

This module provides functions to:
1. Classify mRNA clinical trials into predefined categories using various LLM APIs
2. Process multiple trials in batch
3. Generate a summary of classifications

Usage:
    from classify_trials import classify_mrna_trial, classify_multiple_trials
    import json

    # Load previously fetched trials
    with open("mrna_trials.json", "r") as f:
        trials = json.load(f)

    # Classify a single trial
    classification = classify_mrna_trial(trials[0])
    print(classification)

    # Classify multiple trials and save results
    results = classify_multiple_trials(trials, output_file="classifications.json")
"""

import json
import os
import time
import logging
from typing import Dict, List, Optional
import openai
import sys
import argparse
from abc import ABC, abstractmethod

from .google_sheets_client import GoogleSheetsClient
from .config import GOOGLE_SHEETS_CONFIG
from .prompt_manager import PromptManager, LLMProviderType

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def get_completion(self, system_prompt: str, user_prompt: str) -> str:
        """Get completion from the LLM provider."""
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI API provider implementation."""
    
    def __init__(self, api_key: str):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except (ImportError, AttributeError):
            openai.api_key = api_key
            self.client = None
    
    def get_completion(self, system_prompt: str, user_prompt: str) -> str:
        try:
            if self.client:
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3
                )
                return response.choices[0].message.content
            else:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3
                )
                return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

class GeminiProvider(LLMProvider):
    """Google Gemini API provider implementation."""
    
    def __init__(self, api_key: str):
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
        except ImportError:
            logger.error("Google Generative AI package not found. Please install it with: pip install google-generativeai")
            raise
    
    def get_completion(self, system_prompt: str, user_prompt: str) -> str:
        try:
            # Combine system and user prompts for Gemini
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = self.model.generate_content(combined_prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

def get_llm_provider(provider_name: str, api_key: str) -> LLMProvider:
    """Factory function to get the appropriate LLM provider."""
    providers = {
        "openai": OpenAIProvider,
        "gemini": GeminiProvider
    }
    
    if provider_name.lower() not in providers:
        raise ValueError(f"Unsupported LLM provider: {provider_name}. Supported providers: {list(providers.keys())}")
    
    return providers[provider_name.lower()](api_key)

def get_provider_type(provider_name: str) -> LLMProviderType:
    """Convert provider name to LLMProviderType enum."""
    provider_map = {
        "openai": LLMProviderType.OPENAI,
        "gemini": LLMProviderType.GEMINI
    }
    return provider_map.get(provider_name.lower(), LLMProviderType.OPENAI)

# Initialize PromptManager with default provider type
prompt_manager = PromptManager(base_path="prompts", provider_type=LLMProviderType.OPENAI)

# Load the classification template
CLASSIFICATION_TEMPLATE_GROUP = "screen_class_combo"
CLASSIFICATION_TEMPLATE_VERSION = "v1"

def classify_mrna_trial(trial_info: Dict, llm_provider: LLMProvider) -> str:
    """
    Classify an mRNA trial using the specified LLM provider.

    Args:
        trial_info: Dictionary containing trial information
        llm_provider: LLM provider instance to use for classification

    Returns:
        String containing the classification result
    """
    # Render the template with trial information
    template_result = prompt_manager.render_template(
        CLASSIFICATION_TEMPLATE_GROUP,
        CLASSIFICATION_TEMPLATE_VERSION,
        **trial_info
    )

    system_prompt = "You are an expert in clinical trial analysis specializing in mRNA-based therapeutics."
    
    try:
        if isinstance(template_result, tuple):
            system_prompt, user_prompt = template_result
        else:
            user_prompt = template_result
        return llm_provider.get_completion(system_prompt, user_prompt)
    except Exception as e:
        logger.error(f"Error during trial classification: {e}")
        return f"Error in classification: {e}"

def classify_multiple_trials(
        trials: List[Dict],
        output_file: str = "mrna_trial_classifications.json",
        delay: int = 0,
        llm_provider: Optional[LLMProvider] = None) -> List[Dict]:
    """
    Classify multiple mRNA trials and save results to a file.

    Args:
        trials: List of dictionaries containing trial information
        output_file: Name of file to save classification results to
        delay: Delay in seconds between API calls to avoid rate limiting
        llm_provider: Optional LLM provider instance. If not provided, will use OpenAI by default.

    Returns:
        List of dictionaries containing classification results
    """
    total_trials = len(trials)
    logger.info(f"Starting classification for {total_trials} trials...")

    # Initialize category counts
    category_counts = {
        "Not eligible": 0,
        "Cancer": 0,
        "Protein Replacement": 0,
        "Genetic Medicines": 0,
        "mRNA-Encoded Biologics": 0,
        "Cell Therapies": 0,
        "Infectious Disease Vaccines": 0
    }

    # Add metadata about the classification run
    results = [{
        "_metadata": {
            "classification_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input_file": os.path.basename(output_file.replace("classified_data/", "").replace("_classifications.json", ".json")),
            "total_trials": total_trials,
            "category_counts": category_counts,  # Will be updated as we process trials
            "llm_provider": llm_provider.__class__.__name__ if llm_provider else "OpenAIProvider"
        }
    }]

    for i, trial in enumerate(trials):
        logger.info(f"\nClassifying trial {i+1}/{total_trials}: {trial['brief_title']}")

        # Classify the trial
        classification = classify_mrna_trial(trial, llm_provider)

        # Extract key indicators from the classification string
        key_indicators = []
        if "KEY INDICATORS:" in classification:
            key_indicators_section = classification.split("KEY INDICATORS:")[1].strip()
            key_indicators = [line.strip() for line in key_indicators_section.split('\n') if line.strip()]

        # Get category and update counts
        category = get_category_from_classification(classification)
        category_counts[category] = category_counts.get(category, 0) + 1

        # Create result by copying all input attributes and adding classification info
        result = trial.copy()  # Copy all input attributes
        result.update({
            "classification": classification,
            "category": category,
            "key_indicators": key_indicators
        })

        # Add to results
        results.append(result)

        # Print classification result
        logger.info(f"\nClassification for {trial['nct_id']}:\n{classification}")

        # Avoid rate limiting
        if i < total_trials - 1:  # Don't sleep after the last item
            logger.info(f"Waiting {delay} seconds before next classification...")
            time.sleep(delay)

    # Update metadata with final category counts
    results[0]["_metadata"]["category_counts"] = category_counts

    # Save results to file
    logger.info(f"\nSaving classification results to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Classification results saved successfully to {output_file}")
    return results


def get_category_from_classification(classification: str) -> str:
    """
    Extract the category from a classification result.

    Args:
        classification: Classification result string

    Returns:
        Category name string
    """
    if "CLASSIFICATION:" not in classification:
        return "Not eligible"

    # Extract classification from the result
    classification_line = [
        line for line in classification.split('\n')
        if "CLASSIFICATION:" in line
    ]
    if not classification_line:
        return "Not eligible"

    class_text = classification_line[0].replace("CLASSIFICATION:", "").strip()

    # Determine category
    if "[1]" in class_text or "Cancer" in class_text:
        return "Cancer"
    elif "[2]" in class_text or "Protein Replacement" in class_text:
        return "Protein Replacement"
    elif "[3]" in class_text or "Genetic Medicines" in class_text:
        return "Genetic Medicines"
    elif "[4]" in class_text or "mRNA-Encoded Biologics" in class_text:
        return "mRNA-Encoded Biologics"
    elif "[5]" in class_text or "Cell Therapies" in class_text:
        return "Cell Therapies"
    elif "[6]" in class_text or "Infectious Disease Vaccines" in class_text:
        return "Infectious Disease Vaccines"
    else:
        return "Not eligible"


def classify_and_export_trials(
    trials: List[Dict],
    output_file: str = "classifications.json",
    export_to_sheets: bool = False,
    spreadsheet_id: Optional[str] = None,
    new_spreadsheet_title: Optional[str] = None
) -> List[Dict]:
    """
    Classifies a list of clinical trials and optionally exports the results
    (classifications, summary, and raw trial data) to Google Sheets.

    Args:
        trials (List[Dict]): A list of dictionaries, where each dictionary
                             represents a clinical trial's data.
        output_file (str): Path to save the JSON file with classification results.
                           Defaults to "classifications.json".
        export_to_sheets (bool): If True, results will be exported to Google Sheets.
                                 Defaults to False.
        spreadsheet_id (Optional[str]): The ID of an existing Google Spreadsheet to export to.
                                        If None, and `new_spreadsheet_title` is also None,
                                        it will try to use the default ID from config.
        new_spreadsheet_title (Optional[str]): If provided and `spreadsheet_id` is None,
                                               a new spreadsheet with this title will be created.

    Returns:
        List[Dict]: The list of classification results.
    """
    logger.info("Starting trial classification and export process...")
    results = classify_multiple_trials(trials, output_file)

    if not export_to_sheets:
        logger.info("Google Sheets export is disabled. Skipping export.")
        return results

    logger.info("Initializing GoogleSheetsClient...")
    exporter = GoogleSheetsClient()

    if not exporter.service:
        logger.error("GoogleSheetsClient authentication failed. Cannot export to Google Sheets.")
        return results

    target_spreadsheet_id = None
    created_new_sheet = False

    if spreadsheet_id:
        logger.info(f"Using provided existing spreadsheet ID: {spreadsheet_id}")
        target_spreadsheet_id = spreadsheet_id
    elif new_spreadsheet_title:
        logger.info(f"Attempting to create a new spreadsheet with title: '{new_spreadsheet_title}'")
        target_spreadsheet_id = exporter.create_new_spreadsheet(new_spreadsheet_title)
        if target_spreadsheet_id:
            created_new_sheet = True
            logger.info(f"Successfully created new spreadsheet with ID: {target_spreadsheet_id}")
        else:
            logger.error(f"Failed to create new spreadsheet with title: '{new_spreadsheet_title}'.")
    else:
        default_id_from_config = GOOGLE_SHEETS_CONFIG.get('default_spreadsheet_id')
        if default_id_from_config:
            logger.info(f"Using default spreadsheet ID from config: {default_id_from_config}")
            target_spreadsheet_id = default_id_from_config
        else:
            logger.error("No spreadsheet_id provided, no new_spreadsheet_title specified, and no default_spreadsheet_id found in config. Cannot export to Google Sheets.")

    if not target_spreadsheet_id:
        logger.warning("No target spreadsheet ID determined. Skipping Google Sheets export.")
        return results

    logger.info(f"Exporting classification results to spreadsheet ID: {target_spreadsheet_id}")
    class_export_success = exporter.export_classification_results(results, target_spreadsheet_id)
    if class_export_success:
        logger.info("Successfully exported classification results.")
    else:
        logger.error("Failed to export classification results.")

    logger.info(f"Creating summary sheet in spreadsheet ID: {target_spreadsheet_id}")
    summary_export_success = exporter.create_summary_sheet(results, target_spreadsheet_id)
    if summary_export_success:
        logger.info("Successfully created summary sheet.")
    else:
        logger.error("Failed to create summary sheet.")

    logger.info(f"Exporting raw trial data to spreadsheet ID: {target_spreadsheet_id}")
    # Assuming 'trials' is the list of raw trial dicts passed to this function
    trial_data_export_success = exporter.export_trial_data(trials, target_spreadsheet_id)
    if trial_data_export_success:
        logger.info("Successfully exported raw trial data.")
    else:
        logger.error("Failed to export raw trial data.")
    
    sheet_url_base = "https://docs.google.com/spreadsheets/d/"
    if created_new_sheet:
        logger.info(f"All exports completed to new sheet: {sheet_url_base}{target_spreadsheet_id}")
    else:
        logger.info(f"All exports completed to existing sheet: {sheet_url_base}{target_spreadsheet_id}")

    return results


def generate_classification_summary(results: List[Dict]) -> Dict[str, int]:
    """
    Generate a summary of classification results.

    Args:
        results: List of dictionaries containing classification results

    Returns:
        Dictionary mapping category names to counts
    """
    categories = {
        "Not eligible": 0,
        "Cancer": 0,
        "Protein Replacement": 0,
        "Genetic Medicines": 0,
        "mRNA-Encoded Biologics": 0,
        "Cell Therapies": 0,
        "Infectious Disease Vaccines": 0
    }

    for result in results:
        category = result.get("category", "Unknown")
        if category in categories:
            categories[category] += 1
        else:
            categories["Not eligible"] += 1

    return categories


def print_classification_summary(categories: Dict[str, int]) -> None:
    """
    Print a summary of classification results.

    Args:
        categories: Dictionary mapping category names to counts
    """
    logger.info("\nClassification Summary:")
    for category, count in categories.items():
        logger.info(f"- {category}: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify mRNA clinical trials')
    parser.add_argument('input_file', help='Input JSON file containing trials to classify')
    parser.add_argument('output_file', help='Output JSON file to save classification results')
    parser.add_argument('--provider', choices=['openai', 'gemini'], default='openai',
                      help='LLM provider to use (default: openai)')
    parser.add_argument('--api-key-file', default=None,
                      help='File containing the API key (default: provider-specific key file)')
    args = parser.parse_args()

    # Determine API key file based on provider if not specified
    if args.api_key_file is None:
        args.api_key_file = f"{args.provider}_api_key.txt"

    # Load API key
    try:
        with open(args.api_key_file, 'r') as f:
            api_key = f.read().strip()
    except FileNotFoundError:
        logger.error(f"API key file not found: {args.api_key_file}")
        sys.exit(1)

    # Initialize LLM provider
    try:
        llm_provider = get_llm_provider(args.provider, api_key)
        # Update PromptManager with the correct provider type
        prompt_manager.set_provider_type(get_provider_type(args.provider))
    except Exception as e:
        logger.error(f"Failed to initialize LLM provider: {e}")
        sys.exit(1)

    logger.info(f"Loading trials from {args.input_file}...")
    try:
        with open(args.input_file, "r") as f:
            trials_data = json.load(f)

        logger.info(f"Loaded {len(trials_data)} trials")
        
        # Classify trials and save to local JSON
        logger.info("Classifying trials and saving to local JSON...")
        results_data = classify_multiple_trials(trials_data, output_file=args.output_file, llm_provider=llm_provider)

        # Generate and print summary
        if results_data:
            categories_summary = generate_classification_summary(results_data)
            print_classification_summary(categories_summary)

    except FileNotFoundError:
        logger.error(f"File not found: {args.input_file}")
        logger.error("Please check if the input file exists")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
