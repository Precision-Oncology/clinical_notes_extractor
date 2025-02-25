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
      patient_ids_path (str): Path to the CSV file containing patient IDs under the column 'patientdurablekey'.
      input_dir (str): Directory containing input encounter Parquet files.
      output_dir (str): Directory where the filtered encounter records will be saved.
    """
    
    print(f"Loading patient IDs from {patient_ids_path}")
    # Read the CSV and extract the patient IDs
    patient_ids = pd.read_csv(patient_ids_path)['patientdurablekey'].tolist()
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
    
    # Query to filter records in batches of 500 IDs at a time
    print("Filtering encounters by exact patient ID match (in batches of 500)...")
    
    # Convert each patient ID to a quoted string for SQL compatibility
    quoted_ids = [f"'{pid}'" for pid in patient_ids]
    
    # Process in batches of 500 IDs
    batch_size = 500
    all_results = []
    
    for i in range(0, len(quoted_ids), batch_size):
        batch = quoted_ids[i:i+batch_size]
        id_list_str = ",".join(batch)
        
        print(f"Processing batch {i//batch_size + 1}/{(len(quoted_ids) + batch_size - 1)//batch_size} ({len(batch)} IDs)")
        
        # Construct SQL query for this batch
        query = f"""
        SELECT {', '.join(columns_to_keep)}
        FROM read_parquet('{input_path}/**/*.parquet')
        WHERE patientdurablekey IN ({id_list_str})
        """
        
        batch_results = con.execute(query).df()
        print(f"  Found {len(batch_results)} matching records in this batch")
        
        if not batch_results.empty:
            all_results.append(batch_results)
    
    # Combine all batch results
    if all_results:
        filtered_df = pd.concat(all_results, ignore_index=True)
        print(f"Total: Found {len(filtered_df)} exact matching encounter records across all batches") # Found 2232104 exact matching encounter records across all batches
    else:
        filtered_df = pd.DataFrame(columns=columns_to_keep)
        print("No matching records found in any batch")
    
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
    
    # Add partition column to the filtered dataframe instead of creating a separate table
    filtered_df['partition_key'] = partition_keys
    
    # Convert the updated dataframe with partition column to PyArrow table
    final_table = pa.Table.from_pandas(filtered_df)
    
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
# python filter_encounters.py --patient_ids data/input/patient_ids_with_durablekey.csv --input_dir /wynton/protected/project/ic/data/parquet/DEID_CDW/encounterfact --output_dir /wynton/protected/home/zack/brtan/Stage_2_Staging_Extractor/data/output