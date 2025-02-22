import pandas as pd
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pyarrow as pa
import argparse


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
    # Read the CSV and extract the patient IDs, converting them into a set for fast lookup
    patient_ids = set(pd.read_csv(patient_ids_path)['patient_id'])
    print(f"Loaded {len(patient_ids)} unique patient IDs")
    
    
    input_path = Path(input_dir);output_path = Path(output_dir) # Convert provided directory paths to Path objects for easier file operations.
    output_path.mkdir(exist_ok=True) # Create the output directory (and any necessary parent directories) if it doesn't exist.
    
    print("Creating dataset...")
    dataset = ds.dataset(input_path, format="parquet")
    
    # Filter by lambda function: For each record, convert 'patientdurablekey' from Arrow scalar to Python value and check if it exists in patient_id set.
    print("Starting filtering and writing process...")
    filtered_dataset = dataset.filter(lambda x: x['patientdurablekey'].as_py() in patient_ids)

    # Select only the required columns
    filtered_dataset = filtered_dataset.select([
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
    ])

    print(f"Dataset schema: {filtered_dataset.schema}")

    # Partition patients so that each partition (file) contains at most 1000 patients.
    partition_map = {pid: str(i // 1000) for i, pid in enumerate(sorted(list(patient_ids)))}
    def create_partition_key(patient_id):
        # Return the partition key for the given patient, ensuring a maximum of 1000 patients per partition.
        return partition_map.get(patient_id, "unknown")

    print("Creating partitioned dataset...")
    # Add the partition column to the dataset
    filtered_dataset = filtered_dataset.map(lambda batch: {
        **batch,
        'partition_key': batch['patientdurablekey'].map(lambda x: create_partition_key(x))
    })

    print("Writing partitioned dataset...")
    # Write the dataset with the new partitioning scheme
    ds.write_dataset(
        filtered_dataset,
        output_path,
        format="parquet",
        partitioning=ds.partitioning(pa.schema([("partition_key", pa.string())])),
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