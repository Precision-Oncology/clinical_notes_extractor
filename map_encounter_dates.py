import pandas as pd
import pyarrow.parquet as pq
import os

def map_note_to_encounter(note_metadata_dir):
    """Map deid_note_key to encounterkey"""
    dfs = []
    for file in os.listdir(note_metadata_dir):
        if file.endswith(".parquet"):
            path = os.path.join(note_metadata_dir, file)
            df = pd.read_parquet(path)
            dfs.append(df[['deid_note_key', 'encounterkey']])
    return pd.concat(dfs)

def map_encounter_to_date(encounterfact_dir):
    """Map encounterkey to datekeyvalue"""
    dfs = []
    for file in os.listdir(encounterfact_dir):
        if file.endswith(".parquet"):
            path = os.path.join(encounterfact_dir, file)
            df = pd.read_parquet(path)
            dfs.append(df[['encounterkey', 'datekeyvalue']])
    return pd.concat(dfs)

def create_full_mapping(note_metadata_dir, encounterfact_dir, output_path):
    # Create note -> encounter mapping
    note_encounter = map_note_to_encounter(note_metadata_dir)
    
    # Create encounter -> date mapping
    encounter_date = map_encounter_to_date(encounterfact_dir)
    
    # Join the mappings
    full_mapping = note_encounter.merge(
        encounter_date,
        on='encounterkey',
        how='left'
    )
    
    # Save results
    full_mapping.to_parquet(output_path, index=False)
    print(f"Saved mapping to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--note_metadata_dir', required=True,
                       help='Path to note_metadata Parquet files')
    parser.add_argument('--encounterfact_dir', required=True,
                       help='Path to encounterfact Parquet files')
    parser.add_argument('--output', required=True,
                       help='Output Parquet file path')
    args = parser.parse_args()
    
    create_full_mapping(args.note_metadata_dir, args.encounterfact_dir, args.output) 