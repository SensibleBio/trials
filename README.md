# Therapeutic mRNA Clinical Trials

The goal of this tool is to fetch ongoing clinical trials involving therapeutic/interventional mRNA therapies, and thereafter to categorize these trials based on the application.

## Usage

Some notes on the way (I think) this is used:

It looks like `run_pipeline.sh` only runs the fetching step, as defined below. Thereafter, the classification steps seem to be run from `main.py`. 

### Fetching and Processing Trials

From `run_fetching_pipeline.py`:
1. Fetches trials for the query search term
2. Fetches trials for each company listed in the companies file
3. Merges and deduplicates all trials
4. Saves the final dataset and its metadata

Of note here, the number of trials that can be pulled per clinicaltrials.gov API request is limited to 500. For the initial mRNA search term, this might be limiting (if not now, then it could be over time as the number of trials presumably increases). Perhaps can optimize fields to search mRNA for so as to improve efficiency.

TODO: Remove extraneous function calls
TODO: Execute fetching pipeline from `main.py`

### Classifying Trials

Notes on `main.py`
- Currently, all inputs are hard-coded.

## Requirements

TODO: Add in requirements used by the second portion of the tool.