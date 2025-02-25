import dask.dataframe as dd
import dask
from pathlib import Path
import argparse
import os
import time
import math
import sys

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
    print(f"Loaded encounters dataframe with {len(encounters.divisions)} partitions")
    
    # Print total rows in encounters - compute in chunks to avoid memory issues
    print("Counting total encounters (chunked computation)...")
    encounters_count = 0
    for i in range(0, len(encounters.divisions), 5):
        chunk = encounters.partitions[i:min(i+5, len(encounters.divisions))]
        chunk_count = len(chunk.compute())
        encounters_count += chunk_count
        print(f"  Processed {min(i+5, len(encounters.divisions))}/{len(encounters.divisions)} partitions, chunk count: {chunk_count}")
    print(f"Total encounters loaded: {encounters_count} rows")

    # Step 2: Load and filter note metadata
    print(f"Loading note metadata from {note_meta_dir}...")
    note_meta = dd.read_parquet(
        note_meta_dir,
        columns=['patientdurablekey', 'patientepicid', 'encounterkey', 'deid_note_key', 'deid_note_id', 'note_type']
    )
    print(f"Loaded note metadata with {len(note_meta.divisions)} partitions")
    
    # Step 3: Merge note metadata with encounters
    print("Filtering note metadata by patient and encounter...")
    filtered_meta = note_meta.merge(encounters, on=['patientdurablekey', 'encounterkey'])
    
    # Step 4: Compute filtered metadata in chunks to reduce memory usage
    print("Computing filtered metadata in chunks...")
    filtered_meta_count = 0
    all_note_keys = []
    
    # Process in chunks to avoid memory issues
    num_chunks = math.ceil(len(filtered_meta.divisions) / 10)
    for i in range(0, len(filtered_meta.divisions), 10):
        print(f"  Processing chunk {i//10 + 1}/{num_chunks}")
        chunk = filtered_meta.partitions[i:min(i+10, len(filtered_meta.divisions))].persist()
        chunk_df = chunk.compute()
        chunk_count = len(chunk_df)
        filtered_meta_count += chunk_count
        chunk_keys = chunk_df['deid_note_key'].unique().tolist()
        all_note_keys.extend(chunk_keys)
        print(f"    Chunk {i//10 + 1} processed: {chunk_count} rows, {len(chunk_keys)} unique keys")
        # Clear memory
        del chunk
        del chunk_df
        
    # Remove duplicates from note keys
    note_keys = list(set(all_note_keys))
    print(f"Total filtered metadata: {filtered_meta_count} rows")
    print(f"Found {len(note_keys)} unique note keys to extract")

    # Step 5: Process note text in batches to avoid memory issues
    print(f"Loading and processing note text from {note_text_dir} in batches...")
    
    # Create a temporary directory for intermediate results
    temp_output_dir = os.path.join(scratch_dir if scratch_dir else output_dir, "temp_results")
    os.makedirs(temp_output_dir, exist_ok=True)
    
    # Process note keys in chunks
    total_processed = 0
    batch_size = min(chunk_size, len(note_keys))
    num_batches = math.ceil(len(note_keys) / batch_size)
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(note_keys))
        batch_keys = note_keys[start_idx:end_idx]
        
        print(f"Processing batch {batch_idx+1}/{num_batches} with {len(batch_keys)} note keys")
        
        # Load only the notes for this batch
        note_text = dd.read_parquet(
            note_text_dir,
            filters=[('deid_note_key', 'in', batch_keys)]
        )
        
        # Load filtered metadata again for this batch
        batch_filtered_meta = filtered_meta[filtered_meta['deid_note_key'].isin(batch_keys)].persist()
        
        # Merge and save this batch
        batch_result = batch_filtered_meta.merge(note_text, on='deid_note_key')
        batch_result = batch_result.repartition(partition_size="50MB")  # Smaller partitions for better memory management
        
        # Save this batch to a temporary location
        batch_output_dir = os.path.join(temp_output_dir, f"batch_{batch_idx}")
        os.makedirs(batch_output_dir, exist_ok=True)
        print(f"Writing batch {batch_idx+1} to {batch_output_dir}...")
        batch_result.to_parquet(batch_output_dir)
        
        # Count processed rows
        try:
            batch_count = len(batch_result.compute())
            total_processed += batch_count
            print(f"Batch {batch_idx+1} processed: {batch_count} rows")
        except Exception as e:
            print(f"Error counting batch rows: {e}")
            print(f"Continuing with next batch...")
        
        # Clear memory
        del note_text
        del batch_filtered_meta
        del batch_result
        
        # Force garbage collection
        import gc
        gc.collect()
    
    # Step 6: Combine all batches into final output
    print(f"Combining all batches into final output at {output_dir}...")
    try:
        all_batches = dd.read_parquet(os.path.join(temp_output_dir, "batch_*"))
        all_batches = all_batches.repartition(partition_size="100MB")
        all_batches.to_parquet(output_dir)
    # except Exception as e:
    #     print(f"Error combining batches: {e}")
    #     print("Attempting to copy individual batches to output directory...")
    #     # Fallback: Just copy the batch directories to the output
    #     import shutil
    #     for batch_idx in range(num_batches):
    #         batch_dir = os.path.join(temp_output_dir, f"batch_{batch_idx}")
    #         if os.path.exists(batch_dir):
    #             dest_dir = os.path.join(output_dir, f"batch_{batch_idx}")
    #             print(f"Copying {batch_dir} to {dest_dir}")
    #             shutil.copytree(batch_dir, dest_dir, dirs_exist_ok=True)
    
    # Clean up temporary files if using scratch space
    print(f"Cleaning up temporary files in {temp_output_dir}...")
    import shutil
    try:
        shutil.rmtree(temp_output_dir, ignore_errors=True)
    except Exception as e:
        print(f"Error cleaning up temp files: {e}")
    
    elapsed_time = time.time() - start_time
    print(f"Filtered notes saved to {output_dir}")
    print(f"Total rows processed: {total_processed}")
    print(f"Process completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter notes by encounters')
    parser.add_argument('--encounters_dir', required=False, default='/wynton/protected/home/zack/brtan/Stage_2_Staging_Extractor/data/output/filtered_encounters', help='Path to filtered encounters')
    parser.add_argument('--note_meta_dir', required=True, default='/wynton/protected/project/ic/data/parquet/DEID_CDW/note_metadata', help='Path to note metadata')
    parser.add_argument('--note_text_dir', required=True, default='/wynton/protected/project/ic/data/parquet/DEID_CDW/note_text', help='Path to note text')
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
# python filter_notes.py --encounters_dir /wynton/protected/home/zack/brtan/Stage_2_Staging_Extractor/data/output/filtered_encounters --note_meta_dir /wynton/protected/project/ic/data/parquet/DEID_CDW/note_metadata --note_text_dir /wynton/protected/project/ic/data/parquet/DEID_CDW/note_text --output_dir /scratch/brtan/filtered_notes --scratch_dir /scratch/brtan/temp_note_processing


# Run Logs:
# Starting note filtering process...
# Output directory: /scratch/brtan/filtered_notes
# Loading filtered encounters from /wynton/protected/home/zack/brtan/Stage_2_Staging_Extractor/data/output/filtered_encounters...
# Loaded encounters dataframe with 10 partitions
# Total encounters loaded: 2232104 rows
# Loading note metadata from /wynton/protected/project/ic/data/parquet/DEID_CDW/note_metadata...
# Loaded note metadata with 320 partitions
# Filtering note metadata by patient and encounter...
# Computing filtered metadata...
# Total filtered metadata: 2007426 rows
# Found 2007426 unique note keys to extract
# Loading note text from /wynton/protected/project/ic/data/parquet/DEID_CDW/note_text (filtered by keys)...