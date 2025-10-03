import json
import logging
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from google_sheets_client import GoogleSheetsClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set the font to DM Sans
plt.rcParams['font.family'] = 'DM Sans'

def abbreviate_sponsor(sponsor):
    if sponsor == 'National Institute of Allergy and Infectious Diseases (NIAID)':
        return 'NIAID'
    elif sponsor == 'Sanofi Pasteur, a Sanofi Company':
        return 'Sanofi Past.'
    elif 'Therapeutics' in sponsor:
        return sponsor.replace('Therapeutics', 'Tx.')
    elif 'University Hospital' in sponsor:
        return sponsor.replace('University Hospital', 'Univ. Hosp.')
    return sponsor

def main():
    json_file = "20250602_223022_RawData_filtered.json"
    spreadsheet_id = "1HpyjKdt_n929Pmca0rAyDt1mrw2YDExjoO9CGgIuCso"
    worksheet_name = "sponsor_by_phase"
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        trials = data["trials"]
        if not trials:
            logger.error("No trial data found in JSON.")
            return

        df = pd.DataFrame(trials)
        logger.info("Column names in the DataFrame:")
        logger.info(df.columns.tolist())

        # Flatten the phases column (each phase in a list, even if only one)
        df = df.explode('phases')

        # Create pivot table: sponsors as rows, phases as columns, counts as values
        pivot = pd.pivot_table(df, index='sponsor', columns='phases', values='nct_id', aggfunc='count', fill_value=0)
        logger.info("Sponsor by Phase table:")
        print(pivot)

        # Add a 'Total Trials' column (sum of trials per sponsor)
        pivot['Total Trials'] = pivot.sum(axis=1)

        # Add a 'Total Trials' row (sum of trials per phase)
        pivot.loc['Total Trials'] = pivot.sum(axis=0)

        logger.info("Updated Sponsor by Phase table with totals:")
        print(pivot)

        # Save to CSV
        pivot.to_csv('sponsor_by_phase.csv')
        logger.info("Saved sponsor by phase table to sponsor_by_phase.csv")

        # Export to Google Sheets
        client = GoogleSheetsClient()
        if not client.service:
            logger.error("Failed to initialize Google Sheets Client.")
            return

        # Sort the pivot table by 'Total Trials' in descending order
        pivot_sorted = pivot.sort_values(by='Total Trials', ascending=False)

        # Prepare data for export: header row + data rows
        headers = [str(col) for col in pivot_sorted.columns.tolist()]
        rows = [[abbreviate_sponsor(str(idx))] + [int(val) for val in row] for idx, row in zip(pivot_sorted.index, pivot_sorted.values)]
        export_data = [["sponsor"] + headers] + rows

        # Check if worksheet exists, create if not
        method_name = "list_sheets"
        api_call = lambda: client.service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
        response = client._make_api_request_with_retry(api_call, method_name)
        sheet_titles = [sheet.get('properties', {}).get('title') for sheet in response['sheets']] if response and 'sheets' in response else []
        if worksheet_name not in sheet_titles:
            logger.info(f"Worksheet '{worksheet_name}' does not exist. Creating it...")
            if not client.create_worksheet(spreadsheet_id, worksheet_name):
                logger.error(f"Failed to create worksheet '{worksheet_name}'.")
                return

        # Export the table
        logger.info(f"Exporting sponsor by phase table to worksheet '{worksheet_name}'...")
        try:
            client.service.spreadsheets().values().update(
                spreadsheetId=spreadsheet_id,
                range=f"{worksheet_name}!A1",
                valueInputOption='RAW',
                body={'values': export_data}
            ).execute()
            logger.info(f"Successfully exported sponsor by phase table to worksheet '{worksheet_name}'")
        except Exception as e:
            logger.error(f"Failed to export sponsor by phase table to worksheet '{worksheet_name}': {e}")

        # Visualizations
        # 1. Bar chart for total trials per sponsor (top 10, excluding 'Total Trials' row)
        top_sponsors = pivot_sorted.drop('Total Trials')['Total Trials'].head(10)
        top_sponsors.index = [abbreviate_sponsor(idx) for idx in top_sponsors.index]
        plt.figure(figsize=(12, 6))
        ax = top_sponsors.plot(kind='bar', color='skyblue')
        plt.title('Top 10 Sponsors by Total Trials', fontsize=18)
        plt.xlabel('Sponsor', fontsize=16)
        plt.ylabel('Total Trials', fontsize=16)
        plt.xticks(rotation=45, ha='right', fontsize=14)
        for i, v in enumerate(top_sponsors):
            ax.text(i, v + 0.1, str(v), ha='center', fontsize=14)
        plt.tight_layout()
        plt.savefig('top_sponsors.png')
        logger.info("Saved top sponsors bar chart to top_sponsors.png")

        # 2. Pie chart for total trials per phase
        phase_totals = pivot.loc['Total Trials'].drop('Total Trials')
        plt.figure(figsize=(10, 10))
        phase_totals.plot(kind='pie', autopct='%1.1f%%', startangle=90, fontsize=12)
        plt.title('Total Trials by Phase', fontsize=16)
        plt.ylabel('')
        plt.tight_layout()
        plt.savefig('phase_distribution.png')
        logger.info("Saved phase distribution pie chart to phase_distribution.png")

    except FileNotFoundError:
        logger.error(f"File {json_file} not found.")
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in {json_file}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main() 