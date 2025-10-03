import json
from collections import Counter

def analyze_mrna_trials_json(filename="mrna_trials_500_classified.json"):
    """
    Analyzes an mRNA clinical trials JSON file to provide segment counts and summary statistics.

    Args:
        filename (str, optional): The name of the JSON file to analyze.
            Defaults to "mrna_trials_500_classified.json".

    Returns:
        dict: A dictionary containing analysis results, including segment counts,
            total trials, top sponsors, and sponsor counts.  Returns an empty dictionary if the
            file cannot be read.
    """
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: File '{filename}' is not a valid JSON file.")
        return {}
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return {}

    segments = [trial['segment'] for trial in data]
    sponsors = [trial['sponsor'] for trial in data]

    segment_counts = Counter(segments)
    total_trials = len(data)
    sponsor_counts = Counter(sponsors)
    top_sponsors = sponsor_counts.most_common(5)  # Top 5 sponsors

    analysis_results = {
        "segment_counts": segment_counts,
        "total_trials": total_trials,
        "top_sponsors": top_sponsors,
        "sponsor_counts": sponsor_counts  # Include full sponsor counts for other purposes
    }

    return analysis_results


if __name__ == "__main__":
    results = analyze_mrna_trials_json()

    if results:
        print("Analysis Results:")
        print(f"Total Trials: {results['total_trials']}")
        print("\nSegment Counts:")
        for segment, count in results['segment_counts'].items():
            print(f"  {segment}: {count}")

        print("\nTop Sponsors:")
        for sponsor, count in results['top_sponsors']:
            print(f"  {sponsor}: {count}")