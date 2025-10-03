from classify_trials import classify_and_export_trials

# Sample trial data for testing
sample_trials = [
    {
        "NCT ID": "NCT123456",
        "Brief Title": "Test Trial 1",
        "Official Title": "Official Test Trial 1",
        "Sponsor": "Test Sponsor",
        "Phase": "Phase 2",
        "Conditions": ["Test Condition 1", "Test Condition 2"],
        "Interventions": {"type": "Drug", "name": "Test Drug"},
        "Summary": "This is a test trial summary"
    },
    {
        "NCT ID": "NCT789012",
        "Brief Title": "Test Trial 2",
        "Official Title": "Official Test Trial 2",
        "Sponsor": "Another Test Sponsor",
        "Phase": "Phase 3",
        "Conditions": ["Test Condition 3"],
        "Interventions": {"type": "Device", "name": "Test Device"},
        "Summary": "Another test trial summary"
    }
]

# Call the classification and export function
results = classify_and_export_trials(
    trials=sample_trials,
    output_file="test_classifications.json",
    export_to_sheets=True  # Enable Google Sheets export
)

print("Export completed. Check your Google Sheet!") 