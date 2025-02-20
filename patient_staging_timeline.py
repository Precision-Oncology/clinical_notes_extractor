import pandas as pd
import pyarrow.parquet as pq
import os
from datetime import datetime
from typing import Dict, List

def load_staging_data(staging_dir: str) -> pd.DataFrame:
    """Load staging results from extract_staging.py output"""
    dfs = []
    for file in os.listdir(staging_dir):
        if file.endswith(".parquet"):
            path = os.path.join(staging_dir, file)
            df = pd.read_parquet(path)
            dfs.append(df)
    return pd.concat(dfs)

def load_note_encounter_mapping(mapping_path: str) -> pd.DataFrame:
    """Load output from map_encounter_dates.py"""
    return pd.read_parquet(mapping_path)

def load_encounter_patients(encounterfact_dir: str) -> pd.DataFrame:
    """Load encounter-patient mapping from encounterfact Parquets"""
    dfs = []
    for file in os.listdir(encounterfact_dir):
        if file.endswith(".parquet"):
            path = os.path.join(encounterfact_dir, file)
            df = pd.read_parquet(path)[['encounterkey', 'patientdurablekey', 'datekeyvalue']]
            dfs.append(df)
    return pd.concat(dfs)

def create_patient_timelines(
    staging_dir: str,
    note_encounter_mapping_path: str,
    encounterfact_dir: str,
    output_path: str
) -> None:
    # Load all data sources
    staging_df = load_staging_data(staging_dir)
    note_encounter_df = load_note_encounter_mapping(note_encounter_mapping_path)
    encounter_patient_df = load_encounter_patients(encounterfact_dir)
    
    # Merge staging data with encounter mapping
    merged_df = staging_df.merge(
        note_encounter_df,
        left_on='note_id',
        right_on='deid_note_key',
        how='inner'
    ).merge(
        encounter_patient_df,
        on='encounterkey',
        how='inner'
    )
    
    # Filter to only notes with staging information
    staged_encounters = merged_df[merged_df['stage'].notna()]
    
    # Convert date strings to datetime objects for sorting
    staged_encounters['encounter_date'] = pd.to_datetime(
        staged_encounters['datekeyvalue'],
        errors='coerce'
    )
    
    # Group by patient and sort encounters chronologically
    patient_timelines = (
        staged_encounters
        .sort_values(['patientdurablekey', 'encounter_date'])
        .groupby('patientdurablekey')
        .apply(lambda x: x[['encounterkey', 'stage', 'encounter_date']].to_dict('records'))
        .reset_index(name='encounters')
    )
    
    # Save final output
    patient_timelines.to_parquet(output_path, index=False)
    print(f"Saved patient timelines to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Create patient staging timelines')
    parser.add_argument('--staging_dir', required=True,
                       help='Directory from extract_staging.py output')
    parser.add_argument('--note_encounter_mapping', required=True,
                       help='Path to note_encounter_mapping.parquet from map_encounter_dates.py')
    parser.add_argument('--encounterfact_dir', required=True,
                       help='Directory with encounterfact Parquet files')
    parser.add_argument('--output', required=True,
                       help='Output Parquet file path')
    args = parser.parse_args()
    
    create_patient_timelines(
        args.staging_dir,
        args.note_encounter_mapping,
        args.encounterfact_dir,
        args.output
    ) 