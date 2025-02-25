# Pseudo Code

# 1. Start with list of patient ids /wynton/protected/home/zack/brtan/Stage_2_Staging_Extractor/data/input/patient_ids.csv 
# 2. Match by patientdurablekey
# 3. Get all encounters from the database /wynton/protected/project/ic/data/parquet/DEID_CDW/encounterfact, columns_to_keep = ['patientdurablekey', 'encounterkey', 'datekey', 'datekeyvalue', 'enddatekey', 'enddatekeyvalue', 'admissiondatekey', 'admissiondatekeyvalue', 'dischargedatekey', 'dischargedatekeyvalue']
# 4. Save the filtered encounterskeys to a new table



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

    # Use DuckDB to read the patient IDs CSV
    print("Reading patient IDs using DuckDB...")
    patient_ids_query = f"""
    SELECT patient_id
    FROM read_csv('{patient_ids_path}')
    """
    patient_ids_df = con.execute(patient_ids_query).df()
    print(f"Loaded {len(patient_ids_df)} unique patient IDs")
    
    # Query to filter records
    print("Filtering encounters by exact patient ID match...")
    
    # Construct SQL query to filter encounter records using JOIN instead of IN clause
    query = f"""
    SELECT e.{', e.'.join(columns_to_keep)}
    FROM read_parquet('{input_path}/**/*.parquet') AS e
    JOIN (SELECT patient_id AS patientdurablekey FROM read_csv('{patient_ids_path}')) AS p
    ON e.patientdurablekey = p.patientdurablekey
    """
    
    filtered_df = con.execute(query).df()
    print(f"Found {len(filtered_df)} exact matching encounter records")
    
    # Close DuckDB connection
    con.close()





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter encounters by patient IDs')
    parser.add_argument('--patient_ids', required=True, help='Path to CSV file with patient IDs')
    parser.add_argument('--input_dir', required=True, help='Input directory containing encounter Parquet files')
    parser.add_argument('--output_dir', required=True, help='Output directory for filtered encounters')
    
    args = parser.parse_args()
    filter_encounters(args.patient_ids, args.input_dir, args.output_dir)


# To run as an individual script:
# python new_filter_encounters.py --patient_ids data/input/patient_ids.csv --input_dir /wynton/protected/project/ic/data/parquet/DEID_CDW/encounterfact --output_dir /scratch/brtan/filtered_encounters


# # DEBUG:
# 1. change new_filter to filter
# 2. >>> import pandas as pd; print(pd.read_parquet('part-00000-d75a1394-d6d1-4131-84b3-44ed4b8564f7-c000.snappy.parquet').head())
#    deidlds patientdurablekey    encounterkey  ... hasfollowingadmission  count medicationliststatusproviderdurablekey
# 0  deid_uf    D5F9BE36F574E7  DD19A4B937254C  ...                   NaN      1                                     -1
# 1  deid_uf    D9FFDE696ACD21  D004461966DD1E  ...                   NaN      1                                     -1
# 2  deid_uf    DE01E0D6020E8A  D57064FE226EDC  ...                   NaN      1                                     -1
# 3  deid_uf    DE01E0D6020E8A  DE02B170A28959  ...                   NaN      1                                     -1
# 4  deid_uf    DCEDF9155E5B78  D76AEB7611EBC1  ...                   NaN      1                                     -1