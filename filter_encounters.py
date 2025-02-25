#!/usr/bin/env python3

import pandas as pd
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
import argparse
import duckdb


def filter_encounters(patient_ids_path: str, input_dir: str, output_dir: str):
    """
    Filters encounter records stored in partitioned Parquet files based on patient IDs.
    Uses DuckDB for efficient filtering and PyArrow for creating partitioned output.
    
    Parameters:
      patient_ids_path (str): Path to the CSV file containing patient IDs under the column 'patient_id'.
      input_dir (str): Directory containing input encounter Parquet files.
      output_dir (str): Directory where the filtered encounter records will be saved.
    """
    
    print(f"Loading patient IDs from {patient_ids_path}")
    # Read the CSV and extract the patient IDs
    patient_ids = pd.read_csv(patient_ids_path)['patient_id'].tolist()
    print(f"Loaded {len(patient_ids)} unique patient IDs")
    # print(f"Sample patient IDs from CSV: {patient_ids[:3]}")
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Define columns to keep
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
    
    # Connect to DuckDB and filter the data
    print("Connecting to DuckDB...")
    con = duckdb.connect()
    
    # Query to filter records
    print("Filtering encounters by exact patient ID match...")
    # Convert each patient ID to a quoted string for SQL compatibility
    # This is necessary because patientdurablekey is likely stored as a string in the database
    quoted_ids = [f"'{pid}'" for pid in patient_ids]
    # Join all quoted IDs with commas to create a valid SQL IN clause
    id_list_str = ",".join(quoted_ids)
    
    # Construct SQL query to filter encounter records:
    # 1. Select only the columns we need to keep (defined earlier)
    # 2. Read from all Parquet files in the input directory (using glob pattern)
    # 3. Filter where patientdurablekey matches any of our target patient IDs
    query = f"""
    SELECT {', '.join(columns_to_keep)}
    FROM read_parquet('{input_path}/**/*.parquet')
    WHERE patientdurablekey IN ({id_list_str})
    """
    # Note: This approach loads the list of IDs directly into the query
    # For very large patient lists, this could cause performance issues
    # or exceed query size limits
    filtered_df = con.execute(query).df()
    print(f"Found {len(filtered_df)} exact matching encounter records")
    
    # Close DuckDB connection
    con.close()
    
    if len(filtered_df) == 0:
        print("No matching records found after all attempts. Exiting.")
        return
    
    print(f"Final result: Found {len(filtered_df)} matching encounter records")
    if len(filtered_df) > 0:
        print(f"Sample of matched patientdurablekey values: {filtered_df['patientdurablekey'].head(3).tolist()}")
    
    # Create a partition map - each partition contains ~1000 patients
    print("Creating partition mapping...")
    sorted_patient_ids = sorted(list(set(patient_ids)))
    partition_map = {pid: str(i // 1000) for i, pid in enumerate(sorted_patient_ids)}
    
    # Convert filtered data to PyArrow table
    filtered_table = pa.Table.from_pandas(filtered_df)
    
    # Add partition column 
    patient_ids_col = filtered_table['patientdurablekey'].to_pandas()
    partition_keys = [partition_map.get(pid, "unknown") for pid in patient_ids_col]
    
    # Add partition column to the table
    tables = [filtered_table]
    tables.append(pa.Table.from_arrays([pa.array(partition_keys)], names=['partition_key']))
    
    # Combine tables
    final_table = pa.concat_tables(tables)
    
    # Write partitioned dataset
    print(f"Writing {len(filtered_df)} filtered encounters to {output_path}...")
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