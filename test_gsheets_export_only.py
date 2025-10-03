from google_sheets_client import GoogleSheetsClient

# Dummy data for testing
dummy_trials = [
    {
        "NCT ID": "NCT000001",
        "Brief Title": "Dummy Trial 1",
        "Official Title": "Official Dummy Trial 1",
        "Sponsor": "Dummy Sponsor",
        "Phase": "Phase 1",
        "Conditions": ["Condition A"],
        "Interventions": {"type": "Drug", "name": "Dummy Drug"},
        "Summary": "A dummy summary"
    }
]

exporter = GoogleSheetsClient()
sheet_id = "1HpyjKdt_n929Pmca0rAyDt1mrw2YDExjoO9CGgIuCso"  # Your sheet ID
new_worksheet_name = "New Test Tab 2" # A different new sheet name to test creation again

# First, try to create the new worksheet
print(f"Attempting to create worksheet: {new_worksheet_name}")
creation_success = exporter.create_worksheet(sheet_id, new_worksheet_name)

if creation_success:
    print(f"Successfully created worksheet: {new_worksheet_name}")
    # If creation is successful, export data to the new worksheet
    print(f"Attempting to export data to worksheet: {new_worksheet_name}")
    export_success = exporter.export_trial_data(dummy_trials, sheet_id, worksheet_name=new_worksheet_name)
    print("Export success:", export_success)
else:
    print(f"Failed to create worksheet: {new_worksheet_name}. Cannot proceed with data export.")
    print("Export success: False") 