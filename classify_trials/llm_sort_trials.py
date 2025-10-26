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
import csv
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
        raise NotImplementedError()


class OpenAIProvider(LLMProvider):
    """OpenAI API provider implementation."""

    def __init__(self, api_key: str):
        try:
            # Try new OpenAI client
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except Exception:
            # Fallback to legacy openai module usage
            import openai as _openai
            _openai.api_key = api_key
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
                import openai as _openai
                response = _openai.ChatCompletion.create(
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
            # Using a high-level model helper if available; this may need updates
            self.model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
        except ImportError:
            logger.error("Google Generative AI package not found. Please install it with: pip install google-generativeai")
            raise

    def get_completion(self, system_prompt: str, user_prompt: str) -> str:
        try:
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


# prompt_manager will be initialized in main after provider selection so the correct
# provider-specific templates can be selected.
prompt_manager: Optional[PromptManager] = None


CLASSIFICATION_TEMPLATE_GROUP = "screen_class_combo"
CLASSIFICATION_TEMPLATE_VERSION = "v1"


def classify_mrna_trial(trial_info: Dict, llm_provider: LLMProvider) -> Dict:
    """
    Classify an mRNA trial using the specified LLM provider.

    Returns a dict containing the original trial data plus classification metadata.
    """
    global prompt_manager

    if prompt_manager is None:
        # Initialize with a sensible default if not set (should be set in main normally)
        prompt_manager = PromptManager(base_path="prompts", provider_type=LLMProviderType.OPENAI)

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

        classification_text = llm_provider.get_completion(system_prompt, user_prompt)

        # Extract key indicators if present
        key_indicators = []
        if "KEY INDICATORS:" in classification_text:
            key_section = classification_text.split("KEY INDICATORS:", 1)[1].strip()
            key_indicators = [line.strip() for line in key_section.splitlines() if line.strip()]

        category = get_category_from_classification(classification_text)

        result = {
            **trial_info,
            "classification": classification_text,
            "category": category,
            "key_indicators": key_indicators
        }
        return result
    except Exception as e:
        logger.error(f"Error during trial classification: {e}")
        return {**trial_info, "classification": f"Error in classification: {e}", "category": "Not eligible", "key_indicators": []}


def classify_multiple_trials(
        trials: List[Dict],
        output_file: str = "mrna_trial_classifications.json",
        delay: int = 0,
        llm_provider: Optional[LLMProvider] = None) -> List[Dict]:
    """
    Classify multiple mRNA trials and save results to a JSON file.

    Returns a list where the first element is metadata and the following elements are per-trial results.
    """
    total_trials = len(trials)
    logger.info(f"Starting classification for {total_trials} trials...")

    if llm_provider is None:
        raise ValueError("llm_provider must be provided")

    # Initialize category counts
    category_counts: Dict[str, int] = {}

    # Prepare results with metadata
    metadata = {
        "classification_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_file": os.path.basename(output_file),
        "total_trials": total_trials,
        "category_counts": category_counts,
        "llm_provider": llm_provider.__class__.__name__
    }

    results: List[Dict] = [{"_metadata": metadata}]

    for i, trial in enumerate(trials):
        logger.info(f"Classifying trial {i+1}/{total_trials}: {trial.get('brief_title', 'unknown')}")
        classified = classify_mrna_trial(trial, llm_provider)

        cat = classified.get("category", "Not eligible")
        category_counts[cat] = category_counts.get(cat, 0) + 1

        results.append(classified)

        if delay and i < total_trials - 1:
            time.sleep(delay)

    # Update metadata counts
    results[0]["_metadata"]["category_counts"] = category_counts

    # Save JSON
    logger.info(f"Saving classification results to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Classification results saved successfully")
    return results


def get_category_from_classification(classification: str) -> str:
    """
    Extract the category from a classification result string.
    """
    if not classification or "CLASSIFICATION:" not in classification:
        return "Not eligible"

    lines = [l for l in classification.splitlines() if "CLASSIFICATION:" in l]
    if not lines:
        return "Not eligible"

    class_text = lines[0].split("CLASSIFICATION:", 1)[1].strip()

    # Simple heuristics to map classification to category
    if any(tok in class_text for tok in ("[1]", "Cancer")):
        return "Cancer"
    if any(tok in class_text for tok in ("[2]", "Protein Replacement")):
        return "Protein Replacement"
    if any(tok in class_text for tok in ("[3]", "Genetic Medicines")):
        return "Genetic Medicines"
    if any(tok in class_text for tok in ("[4]", "mRNA-Encoded Biologics")):
        return "mRNA-Encoded Biologics"
    if any(tok in class_text for tok in ("[5]", "Cell Therapies")):
        return "Cell Therapies"
    if any(tok in class_text for tok in ("[6]", "Infectious Disease Vaccines")):
        return "Infectious Disease Vaccines"

    return "Not eligible"


def write_csv_from_results(results: List[Dict], json_output_path: str) -> str:
    """Write a CSV file derived from the JSON results file name.

    Returns the CSV filepath written.
    """
    # Derive CSV filename
    base, ext = os.path.splitext(json_output_path)
    csv_path = f"{base}.csv" if ext.lower() == ".json" else f"{json_output_path}.csv"

    # Skip metadata element when building headers/rows
    rows = [r for r in results if not ("_metadata" in r)]

    # Build header as union of all keys, place common keys first
    common_order = ["nct_id", "brief_title", "phase", "sponsor", "category", "classification", "key_indicators"]
    keys = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)

    # Reorder keys: common fields first, then the rest
    ordered_keys = [k for k in common_order if k in keys] + [k for k in keys if k not in common_order]

    def serialize_value(v):
        if v is None:
            return ""
        if isinstance(v, list):
            return "; ".join(str(x) for x in v)
        if isinstance(v, dict):
            return json.dumps(v, ensure_ascii=False)
        return str(v)

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(ordered_keys)
        for r in rows:
            writer.writerow([serialize_value(r.get(k, "")) for k in ordered_keys])

    logger.info(f"CSV results written to {csv_path}")
    return csv_path


def str2bool(v: str) -> bool:
    return str(v).lower() in ("yes", "true", "t", "1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify mRNA clinical trials')
    parser.add_argument('input_file', help='Input JSON file containing trials to classify')
    parser.add_argument('output_file', help='Output JSON file to save classification results')
    parser.add_argument('--provider', choices=['openai', 'gemini'], default='openai',
                        help='LLM provider to use (default: openai)')
    parser.add_argument('--save-csv', default='true',
                        help='Whether to also save CSV output alongside JSON (true/false). Default: true')
    parser.add_argument('--delay', type=int, default=0, help='Delay in seconds between API calls')
    args = parser.parse_args()

    # Optionally load .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        # python-dotenv is optional; environment variables may already be set
        pass

    save_csv = str2bool(args.save_csv)

    provider = args.provider.lower()

    # Choose API key from environment variables
    api_env_var = "OPENAI_API_KEY" if provider == "openai" else "GEMINI_API_KEY"
    api_key = os.getenv(api_env_var)
    if not api_key:
        logger.error(f"API key not found in environment. Please set the {api_env_var} environment variable (or create a .env file).")
        sys.exit(1)

    # Initialize LLM provider
    llm = get_llm_provider(provider, api_key)

    # Initialize prompt manager with the chosen provider type
    prompt_manager = PromptManager(base_path="prompts", provider_type=get_provider_type(provider))

    # Load trials from input file
    with open(args.input_file, "r") as f:
        trials = json.load(f)

    results = classify_multiple_trials(trials, output_file=args.output_file, delay=args.delay, llm_provider=llm)

    if save_csv:
        write_csv_from_results(results, args.output_file)

