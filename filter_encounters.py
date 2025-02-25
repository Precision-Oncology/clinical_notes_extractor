#!/usr/bin/env python3

import pandas as pd
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pyarrow as pa
import argparse
import duckdb
import os


def filter_encounters(patient_ids_path: str, input_dir: str, output_dir: str):
    """
    Filters encounter records stored in partitioned Parquet files based on patient IDs.

    Detailed steps:
      - Loads patient IDs from the provided CSV file.
      - Converts the list of patient IDs into a set for O(1) membership checking.
      - Reads the encounter data from a given input directory containing Parquet files.
      - Filters the dataset to only include records where 'patientdurablekey' is among the loaded patient IDs.
      - Writes the filtered dataset into an output directory in Parquet format, partitioning the data by 'patientdurablekey'.

    Parameters:
      patient_ids_path (str): Path to the CSV file containing patient IDs under the column 'patient_id'.
      input_dir (str): Directory containing input encounter Parquet files.
      output_dir (str): Directory where the filtered encounter records will be saved.
    """
    
    print(f"Loading patient IDs from {patient_ids_path}")
    # Read the CSV and extract the patient IDs
    patient_df = pd.read_csv(patient_ids_path)
    print(f"Patient ID column name in CSV: {patient_df.columns.tolist()}")
    patient_ids = patient_df['patient_id'].tolist()
    print(f"Loaded {len(patient_ids)} unique patient IDs")
    print(f"Sample patient IDs from CSV: {patient_ids[:3]}")
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Define the columns we want to keep
    columns_to_keep = [
        'patientdurablekey', 
        'encounterkey',
        'datekey',
        'datekeyvalue',
        'enddatekey', 
        'enddatekeyvalue',
        'admissiondatekey',
        'admissiondatekeyvalue',
        'dischargedatekey',
        'dischargedatekeyvalue'
    ]
    
    # First, let's try DuckDB approach like the user's example
    print("Attempting to filter using DuckDB...")
    
    # Connect to DuckDB
    con = duckdb.connect()
    
    # Sample a few records from the dataset to verify structure
    sample_query = f"""
    SELECT patientdurablekey
    FROM read_parquet('{input_path}/*.parquet')
    LIMIT 10
    """
    try:
        sample_df = con.execute(sample_query).df()
        print(f"Sample patient IDs from dataset using DuckDB: {sample_df['patientdurablekey'].tolist()}")
    except Exception as e:
        print(f"Error sampling dataset: {e}")
        # Try using glob to find actual parquet files
        import glob
        parquet_files = glob.glob(f"{input_path}/**/*.parquet", recursive=True)
        print(f"Found {len(parquet_files)} parquet files at path: {input_path}")
        if parquet_files:
            print(f"Example files: {parquet_files[:3]}")
            # Try with a specific file
            sample_query = f"""
            SELECT patientdurablekey
            FROM read_parquet('{parquet_files[0]}')
            LIMIT 10
            """
            try:
                sample_df = con.execute(sample_query).df()
                print(f"Sample patient IDs from first parquet file: {sample_df['patientdurablekey'].tolist()}")
            except Exception as e:
                print(f"Error sampling first parquet file: {e}")
    
    # List a few patient IDs to check format
    quoted_ids = [f"'{pid}'" for pid in patient_ids[:10]]
    id_list_str = ",".join(quoted_ids)
    
    # Main query to filter records
    query = f"""
    SELECT {', '.join(columns_to_keep)}
    FROM read_parquet('{input_path}/**/*.parquet')
    WHERE patientdurablekey IN ({id_list_str})
    """
    
    try:
        print("Executing DuckDB query to filter encounters...")
        filtered_df = con.execute(query).df()
        print(f"Found {len(filtered_df)} matching records using DuckDB")
        
        if len(filtered_df) > 0:
            print(f"Sample of matched records: {filtered_df.head(3).to_dict()}")
        else:
            # Try with case-insensitive comparison
            print("Trying case-insensitive comparison...")
            case_insensitive_query = f"""
            SELECT {', '.join(columns_to_keep)}
            FROM read_parquet('{input_path}/**/*.parquet')
            WHERE LOWER(patientdurablekey) IN ({','.join([f"LOWER('{pid}')" for pid in patient_ids[:10]])})
            """
            try:
                filtered_df = con.execute(case_insensitive_query).df()
                print(f"Found {len(filtered_df)} matching records with case-insensitive comparison")
            except Exception as e:
                print(f"Error with case-insensitive query: {e}")
        
        # Clean up DuckDB connection
        con.close()
        
        if len(filtered_df) == 0:
            print("No matches found, falling back to PyArrow method for more debugging...")
            # Continue with original PyArrow approach for more debugging
            dataset = ds.dataset(input_path, format="parquet")
            print(f"Available columns in dataset: {dataset.schema.names}")
            
            # Verify all columns exist in the dataset
            for col in columns_to_keep:
                if col not in dataset.schema.names:
                    print(f"WARNING: Column '{col}' not found in dataset")
            
            # Create a scanner with just the patientdurablekey column for faster processing
            scanner = dataset.scanner(columns=['patientdurablekey'])
            
            # Get the first 1000 patient IDs from the dataset for comparison
            first_chunk = scanner.to_table().to_pandas()
            unique_dataset_ids = set(first_chunk['patientdurablekey'].unique())
            print(f"Found {len(unique_dataset_ids)} unique patient IDs in first chunk")
            print(f"Sample IDs from dataset: {list(unique_dataset_ids)[:5]}")
            
            # Check a wider range of ID formats - case sensitivity, prefixes, etc.
            patient_id_variations = set()
            for pid in patient_ids:
                patient_id_variations.add(pid)  # Original
                patient_id_variations.add(pid.upper())  # Uppercase
                patient_id_variations.add(pid.lower())  # Lowercase
                # Add without 'D' prefix if it exists
                if pid.startswith('D'):
                    patient_id_variations.add(pid[1:])
                # Try with different formats
                clean_pid = pid.replace('D', '')
                patient_id_variations.add(clean_pid)
            
            print(f"Created {len(patient_id_variations)} variations of patient IDs to check")
            
            # Check for any overlaps
            potential_matches = unique_dataset_ids.intersection(patient_id_variations)
            print(f"Found {len(potential_matches)} potential matches with ID variations")
            if potential_matches:
                print(f"Sample potential matches: {list(potential_matches)[:5]}")
                
            # Exit with message since no matching records were found
            print("ERROR: No matching records found. Please check the patient IDs and dataset.")
            return
        
        # If we have matches, create a PyArrow table from the DataFrame
        print("Converting filtered DataFrame to PyArrow table...")
        filtered_table = pa.Table.from_pandas(filtered_df)
        
    except Exception as e:
        print(f"Error with DuckDB approach: {e}")
        print("Falling back to original PyArrow method...")
        
        # Original PyArrow approach
        dataset = ds.dataset(input_path, format="parquet")
        scanner = dataset.scanner(columns=columns_to_keep)
        table = scanner.to_table()
        
        # Convert to pandas for easier filtering
        df = table.to_pandas()
        filtered_df = df[df['patientdurablekey'].isin(patient_ids)]
        print(f"Found {len(filtered_df)} matching records using PyArrow")
        
        if len(filtered_df) == 0:
            print("No matching records found. Exiting.")
            return
        
        # Convert back to PyArrow table
        filtered_table = pa.Table.from_pandas(filtered_df)
    
    # Create a partition map using a simple modulo approach
    print("Creating partition mapping...")
    # Partition patients so that each partition (file) contains at most 1000 patients.
    sorted_patient_ids = sorted(list(patient_ids))
    partition_map = {pid: str(i // 1000) for i, pid in enumerate(sorted_patient_ids)}
    
    # Debug information
    print(f"Created partition map with {len(partition_map)} entries")
    if len(partition_map) > 0:
        sample_entries = list(partition_map.items())[:3]
        print(f"Sample partition map entries: {sample_entries}")

    # Process the dataset in manageable chunks
    print("Processing records for partitioning...")
    
    # Add partition column to the table
    patient_ids_col = filtered_table['patientdurablekey'].to_pandas()
    partition_keys = [partition_map.get(pid, "unknown") for pid in patient_ids_col]
    
    # Create a new table with the partition column
    tables = [filtered_table]
    tables.append(pa.Table.from_arrays([pa.array(partition_keys)], names=['partition_key']))
    
    # Combine the tables
    final_table = pa.concat_tables(tables, promote=True)
    print(f"Final table schema: {final_table.schema}")

    print("Writing partitioned dataset...")
    # Write the table as a partitioned dataset
    pq.write_to_dataset(
        final_table,
        root_path=str(output_path),
        partition_cols=['partition_key']
    )
    print(f"Filtered encounters saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter encounters by patient IDs')
    parser.add_argument('--patient_ids', required=True, help='Path to CSV file with patient IDs')
    parser.add_argument('--input_dir', required=True, help='Input directory containing encounter Parquet files')
    parser.add_argument('--output_dir', required=True, help='Output directory for filtered encounters')
    
    args = parser.parse_args()
    filter_encounters(args.patient_ids, args.input_dir, args.output_dir)

# To run as an individual script:
# python filter_encounters.py --patient_ids data/input/patient_ids.csv --input_dir /wynton/protected/project/ic/data/parquet/DEID_CDW/encounterfact --output_dir /scratch/brtan/filtered_encounters