import pandas as pd
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pyarrow as pa

def filter_encounters(patient_ids_path: str, input_dir: str, output_dir: str):
    """Filter encounters by patient IDs in chunks"""
    patient_ids = set(pd.read_csv(patient_ids_path)['patient_id'])
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create dataset from partitioned Parquet files
    dataset = ds.dataset(input_path, format="parquet")
    
    # Filter and write in batches
    (ds.dataset(dataset)
       .filter(lambda x: x['patientdurablekey'].as_py() in patient_ids)
       .write_dataset(output_path, 
                     format="parquet",
                     partitioning=ds.partitioning(
                         pa.schema([("patientdurablekey", pa.string())])
                     ))
    )
    print(f"Filtered encounters saved to {output_path}") 