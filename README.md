# Therapeutic mRNA Clinical Trials

The goal of this tool is to fetch ongoing clinical trials involving therapeutic/interventional mRNA therapies, and thereafter to categorize these trials based on the application.

## Setting up environment

To run this tool, this package and its dependencies can be installed in a virtual environment with `venv` and `pip`.

First, set up the virtual environment (if on a Unix-based system):

```bash
python -m venv .venv # will be created in current directory
source .venv/bin/activate
```

Then, install the required packages with pip:

```bash
pip install -r requirements.txt
```

TODO: The google-generativeai package is now deprecated; need to update this dependency to google-genai.

An LLM is used to classify clinical trials fetched from clinicaltrials.gov. For now, the two supported providers are Google Gemini and OpenAI. The API key should be provided as an envionment variable, and the recommended way to do this is within a `.env` file in your project root directory. For example, you can provide one of the following:

```
# For Gemini
GEMINI_API_KEY=your_api_key_here
# Or for OpenAI
OPENAI_API_KEY=your_api_key_here
```

## Usage

For now, this usage is separated into three main steps while development is ongoing: fetching trials, classifying trials, and outputting results.

### Fetching and Processing Trials

From `run_fetching_pipeline.py`:
1. Fetches trials for the query search term
2. Fetches trials for each company listed in the companies file
3. Merges and deduplicates all trials
4. Saves the final dataset and its metadata

Of note here, the number of trials that can be pulled per clinicaltrials.gov API request is limited to 500. For the initial mRNA search term, this might be limiting (if not now, then it could be over time as the number of trials presumably increases). Perhaps can optimize fields to search mRNA for so as to improve efficiency.

To run just this submodule, use:

```bash
python -m classify_trials.run_fetching_pipeline
```

### Classifying Trials

From `llm_sort_trials.py`:
1. Loads the trials JSON file
2. For each trial, prompts the LLM to classify the trial into one of several categories
3. Saves the classified trials to a new JSON file

To run this submodule, use:

```bash
python -m classify_trials.llm_sort_trials trials_class_test.json trials_class_test_classified.json --provider gemini
```

The `--provider` flag accepts either `openai` or `gemini`, currently.
