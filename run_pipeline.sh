#!/bin/bash

# Define input parameters here:

# Primary search term for clinical trials (e.g., "mRNA")
PIPELINE_QUERY_TERM="mRNA"

# Maximum number of trials to fetch per API call (integer)
PIPELINE_LIMIT="500" # Increased limit to 500

# Path to the JSON file containing company names
PIPELINE_COMPANIES_FILE="prompts/reference_data/mrna_companies.json"

# Directory to save the fetched and merged data
# Make sure this matches the 'folder' parameter used in save functions if they are hardcoded
PIPELINE_OUTPUT_FOLDER="pulled_data"

# Filename for the final merged dataset (JSON)
PIPELINE_MERGED_FILENAME="final_merged_trials.json"

# Set to "true" to save individual company fetch results to separate files
# Set to "false" to merge in memory without saving intermediate files
PIPELINE_SAVE_COMPANY_FILES="false"


# --- End of user-configurable parameters ---

# Ensure the output folder exists
mkdir -p $PIPELINE_OUTPUT_FOLDER

# Run the Python script with the defined parameters
# Pass parameters as command-line arguments
python run_fetching_pipeline.py \
    "$PIPELINE_QUERY_TERM" \
    "$PIPELINE_LIMIT" \
    "$PIPELINE_COMPANIES_FILE" \
    "$PIPELINE_OUTPUT_FOLDER" \
    "$PIPELINE_MERGED_FILENAME" \
    "$PIPELINE_SAVE_COMPANY_FILES"

# You can add more commands here if needed, e.g., for classification or export
# echo "Pipeline finished. Check $PIPELINE_OUTPUT_FOLDER for results." 