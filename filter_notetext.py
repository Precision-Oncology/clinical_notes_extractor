#!/usr/bin/env python3
"""
Note Text Extractor

This script extracts note texts from a large dataset based on filtered metadata.
It processes data in chunks to manage memory efficiently and saves the results
as parquet files.
"""

import os
import glob
import logging
from pathlib import Path
import pandas as pd
import dask.dataframe as dd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
FILTERED_METADATA_DIR = "/scratch/brtan/filtered_notes/filtered_metadata"
NOTE_TEXT_DIR = "/wynton/protected/project/ic/data/parquet/DEID_CDW/note_text"
OUTPUT_DIR = "/wynton/protected/home/zack/brtan/Stage_2_Staging_Extractor/data/output/filtered_notes"

def ensure_dir_exists(directory):
    """Ensure the directory exists, create if it doesn't."""
    os.makedirs(directory, exist_ok=True)
    logger.info(f"Ensured directory exists: {directory}")

def load_filtered_metadata():
    """
    Load and combine all filtered metadata parquet files.
    
    Returns:
        pandas.DataFrame: Combined metadata
    """
    metadata_files = glob.glob(os.path.join(FILTERED_METADATA_DIR, "filtered_metadata_chunk_*.parquet"))
    logger.info(f"Found {len(metadata_files)} metadata files to process")
    
    # Load and combine all metadata files
    metadata_dfs = []
    for file_path in tqdm(metadata_files, desc="Loading metadata files"):
        try:
            df = pd.read_parquet(file_path)
            metadata_dfs.append(df)
            logger.debug(f"Loaded metadata file: {file_path}, shape: {df.shape}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    
    combined_metadata = pd.concat(metadata_dfs, ignore_index=True)
    logger.info(f"Combined metadata shape: {combined_metadata.shape}")
    print(f"Loaded and combined {len(metadata_files)} metadata files with {len(combined_metadata)} total records")
    
    return combined_metadata

def extract_note_keys(metadata_df):
    """
    Extract unique note keys from the metadata.
    
    Args:
        metadata_df (pandas.DataFrame): Metadata dataframe
        
    Returns:
        list: List of unique note keys
    """
    note_keys = metadata_df['deid_note_key'].dropna().unique().tolist()
    print(f"Extracted {len(note_keys)} unique note keys to process")
    return note_keys

def process_note_text_in_batches(note_keys, metadata_df, batch_size=3):
    """
    Process note text files in batches, filtering by the extracted note keys.
    
    Args:
        note_keys (list): List of note keys to filter
        metadata_df (pandas.DataFrame): Metadata dataframe for merging
        batch_size (int): Number of parquet files to process in each batch
        
    Returns:
        None: Results are saved to disk
    """
    # Create set of note keys for faster lookups
    note_keys_set = set(note_keys)
    
    # Get all note text parquet files
    note_text_files = glob.glob(os.path.join(NOTE_TEXT_DIR, "*.snappy.parquet"))
    print(f"Found {len(note_text_files)} note text files to process in batches of {batch_size}")
    
    # Process in batches to manage memory
    ensure_dir_exists(OUTPUT_DIR)
    
    # Track which note keys we've found so we can remove them from our search
    found_keys = set()
    batch_num = 0
    
    # Process files in batches
    for i in range(0, len(note_text_files), batch_size):
        batch_files = note_text_files[i:i+batch_size]
        batch_num += 1
        
        print(f"Processing batch {batch_num}/{(len(note_text_files) + batch_size - 1) // batch_size}, files {i+1}-{i+len(batch_files)} of {len(note_text_files)}")
        
        try:
            # Skip if we've found all note keys
            if len(note_keys_set - found_keys) == 0:
                print("All note keys found. Stopping batch processing.")
                break
                
            # Use dask to efficiently load and filter a batch of parquet files
            current_keys = list(note_keys_set - found_keys)
            note_text_batch = dd.read_parquet(batch_files)
            
            # Filter to only rows with keys we need
            filtered_notes = note_text_batch[note_text_batch['deid_note_key'].isin(current_keys)]
            
            # Convert to pandas for easier processing
            batch_df = filtered_notes.compute()
            
            if batch_df.empty:
                logger.info(f"No matching notes found in batch {batch_num}")
                continue
                
            # Update found keys
            batch_found_keys = set(batch_df['deid_note_key'].unique())
            found_keys.update(batch_found_keys)
            print(f"Found {len(batch_found_keys)} new note keys in batch {batch_num}. "
                  f"Progress: {len(found_keys)}/{len(note_keys_set)} ({len(found_keys)/len(note_keys_set)*100:.2f}%)")
            
            # Merge with metadata
            batch_result = pd.merge(
                batch_df,
                metadata_df,
                on='deid_note_key',
                how='inner'
            )
            
            logger.info(f"Merged batch result shape: {batch_result.shape}")
            
            # Save batch result directly to final output directory
            final_output_dir = os.path.join(OUTPUT_DIR, "final")
            ensure_dir_exists(final_output_dir)
            batch_output_file = os.path.join(final_output_dir, f"filtered_notes_batch_{batch_num}.parquet")
            batch_result.to_parquet(batch_output_file, compression='snappy')
            logger.info(f"Saved batch results to {batch_output_file}")
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_num}: {e}", exc_info=True)
    
    print(f"Completed note text extraction. Found {len(found_keys)}/{len(note_keys_set)} note keys "
          f"({len(found_keys)/len(note_keys_set)*100:.2f}%)")

def main():
    """Main function to run the note text extraction pipeline."""
    logger.info("Starting note text extraction pipeline")
    print("Starting note text extraction pipeline")
    
    # Step 1: Load filtered metadata
    metadata_df = load_filtered_metadata()
    
    # Step 2: Extract note keys
    note_keys = extract_note_keys(metadata_df)
    
    # Step 3: Process note text in batches and save directly
    process_note_text_in_batches(note_keys, metadata_df)
    
    print("Note text extraction pipeline completed successfully")

if __name__ == "__main__":
    main()
