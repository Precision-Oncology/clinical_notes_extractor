import dask.dataframe as dd
import dask
from pathlib import Path
import argparse
import os
import time
import math
import sys
from dask.distributed import Client, LocalCluster

def filter_notes(encounters_dir: str, note_meta_dir: str, note_text_dir: str, output_dir: str, 
                 scratch_dir: str = None, chunk_size: int = 100000):
    """Join notes with filtered encounters using Dask"""
    print(f"Starting note filtering process...")
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Use scratch directory for temporary files if provided
    os.makedirs(scratch_dir, exist_ok=True)
    print(f"Using scratch directory for temporary files: {scratch_dir}")
    dask_tmp_dir = os.path.join(scratch_dir, "dask-worker-space")
    os.makedirs(dask_tmp_dir, exist_ok=True)
    dask.config.set({"temporary-directory": dask_tmp_dir})
    print(f"Set Dask temporary directory to: {dask_tmp_dir}")
    
    # Step 1: Load filtered encounters
    print(f"Loading filtered encounters from {encounters_dir}...")
    encounters = dd.read_parquet(
        Path(encounters_dir)/"*.parquet",
        columns=['patientdurablekey', 'encounterkey', 'datekeyvalue', 'enddatekeyvalue']
    )
    print(f"Loaded encounters dataframe")
    encounters_count = len(encounters)
    print(f"Total encounters loaded: {encounters_count} rows")

    # Step 2: Load and filter note metadata
    print(f"Loading note metadata from {note_meta_dir}...")
    note_meta = dd.read_parquet(
        note_meta_dir,
        columns=['patientdurablekey', 'patientepicid', 'encounterkey', 'deid_note_key', 'deid_note_id', 'note_type']
    )
    print(f"Loaded note metadata")
    # Step 3: Merge note metadata with encounters
    print("Filtering note metadata by patient and encounter...")
    filtered_meta = note_meta.merge(encounters, on=['encounterkey'])
    
    # Step 4: Save filtered metadata in chunks of 100,000 rows
    print("Saving filtered metadata in chunks of 100,000 rows...")
    filtered_meta_dir = os.path.join(output_dir, "filtered_metadata")
    os.makedirs(filtered_meta_dir, exist_ok=True)
    
    # Compute the merged data
    computed_meta = filtered_meta.compute()
    total_rows = len(computed_meta)
    print(f"Total rows after merge: {total_rows}")
    
    # Save in chunks of 100,000 rows
    chunk_size = 100000
    num_chunks = math.ceil(total_rows / chunk_size)
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_rows)
        
        # Extract chunk
        chunk_df = computed_meta.iloc[start_idx:end_idx]
        
        # Save chunk
        chunk_path = os.path.join(filtered_meta_dir, f"filtered_metadata_chunk_{i+1}.parquet")
        chunk_df.to_parquet(chunk_path)
        print(f"Saved chunk {i+1}/{num_chunks} with {len(chunk_df)} rows to {chunk_path}")

    print(f"Step 4 done, Move to filter_notetext.py")
    exit()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter notes by encounters')
    parser.add_argument('--encounters_dir', required=False, default='/wynton/protected/home/zack/brtan/Stage_2_Staging_Extractor/data/output/filtered_encounters', help='Path to filtered encounters')
    parser.add_argument('--note_meta_dir', required=True, default='/wynton/protected/project/ic/data/parquet/DEID_CDW/note_metadata', help='Path to note metadata')
    parser.add_argument('--output_dir', required=False, default='/scratch/brtan/filtered_notes', help='Path to save filtered notes')
    parser.add_argument('--scratch_dir', required=False, default='/scratch/brtan/temp_note_processing', help='Path to scratch directory for temporary files')
    parser.add_argument('--chunk_size', type=int, default=50000, help='Number of note keys to process in each batch')
    args = parser.parse_args()
    
    # Expand environment variables in scratch_dir
    if args.scratch_dir:
        args.scratch_dir = os.path.expandvars(args.scratch_dir)
    
    # Call the main function with parsed arguments
    filter_notes(args.encounters_dir, args.note_meta_dir, args.note_text_dir, args.output_dir, 
                 args.scratch_dir, args.chunk_size)

# To run as an individual script:
# python filter_notemetadata.py --encounters_dir /wynton/protected/home/zack/brtan/Stage_2_Staging_Extractor/data/output/filtered_encounters --note_meta_dir /wynton/protected/project/ic/data/parquet/DEID_CDW/note_metadata --output_dir /wynton/protected/home/zack/brtan/Stage_2_Staging_Extractor/data/output --scratch_dir /scratch/brtan/temp_note_processing

# New Run Logs
# (dask_distribution_env) (base) (venv) [brtan@pgpudev1 Stage_2_Staging_Extractor]$ python filter_notes.py --encounters_dir /wynton/protected/home/zack/brtan/Stage_2_Staging_Extractor/data/output/filtered_encounters --note_meta_dir /wynton/protected/project/ic/data/parquet/DEID_CDW/note_metadata --note_text_dir /wynton/protected/project/ic/data/parquet/DEID_CDW/note_text --output_dir /scratch/brtan/filtered_notes --scratch_dir /scratch/brtan/temp_note_processing

# Starting note filtering process...
# Output directory: /scratch/brtan/filtered_notes
# Using scratch directory for temporary files: /scratch/brtan/temp_note_processing
# Set Dask temporary directory to: /scratch/brtan/temp_note_processing/dask-worker-space
# Loading filtered encounters from /wynton/protected/home/zack/brtan/Stage_2_Staging_Extractor/data/output/filtered_encounters...
# Loaded encounters dataframe
# Total encounters loaded: 2232104 rows
# Loading note metadata from /wynton/protected/project/ic/data/parquet/DEID_CDW/note_metadata...
# Loaded note metadata
# Filtering note metadata by patient and encounter...
# Saving filtered metadata in chunks of 100,000 rows...
# Total rows after merge: 2007426
# Pause. Step 4 done.


'/scratch/brtan/filtered_notes/filtered_metadata/filtered_metadata_chunk*.parquet'