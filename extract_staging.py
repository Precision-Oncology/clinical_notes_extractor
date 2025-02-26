import pyarrow.parquet as pq
from pathlib import Path
from staging_utils import StagingExtractor
import pyarrow as pa
import os
import sys
import pandas as pd


def process_batch(batch, extractor):
    """Process a batch of records"""
    print(f"Processing batch of {len(batch)} records...")
    results = []
    
    # Convert PyArrow Table to pandas DataFrame, then iterate through rows
    batch_df = batch.to_pandas()
    for i, (idx, note) in enumerate(batch_df.iterrows(), 1):
        if i % 10 == 0:  # Progress update every 10 records
            print(f"Processed {i}/{len(batch_df)} records...")
            
        findings = extractor.extract_staging(
            note['note_text'],
        )
        for finding in findings:
            results.append({
                'patientdurablekey': note['patientdurablekey'],
                **finding
            })
    print(f"Batch processing complete. Found {len(results)} staging findings")
    return pa.Table.from_pylist(results)

def extract_staging(input_file, output_file, use_llm=True):
    """Stream process notes and extract staging"""
    print(f"\n=== Starting Staging Extraction ===")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}") 
    print(f"Using LLM: {use_llm}")
    
    print("\nInitializing staging extractor...")
    extractor = StagingExtractor()
    extractor.use_llm = use_llm
    
    print("Creating streaming reader...")
    parquet_file = pq.ParquetFile(input_file)
    num_row_groups = parquet_file.num_row_groups
    print(f"Found {num_row_groups} row groups to process")
    
    print("\nStarting batch processing...")
    with pq.ParquetWriter(output_file, schema=pa.schema([
        ('patientdurablekey', pa.string()),
        ('stage', pa.string()),
        ('system', pa.string()),
        ('evidence', pa.string())
    ])) as writer:
        
        total_results = 0
        for i in range(num_row_groups):
            print(f"\nProcessing row group {i+1}/{num_row_groups}")
            batch = parquet_file.read_row_group(i)
            print(f"Row group contains {len(batch)} records")
            
            result_table = process_batch(batch, extractor)
            if result_table.num_rows > 0:
                print(f"Writing {result_table.num_rows} results to output file")
                writer.write_table(result_table)
                total_results += result_table.num_rows
            else:
                print("No results found in this row group")
                
        print(f"\nProcessing complete! Total results written: {total_results}")

if __name__ == "__main__":
    # Get command line arguments
    if len(sys.argv) != 4:
        print("Usage: python extract_staging.py <input_file> <output_file> <use_llm>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    use_llm = sys.argv[3].lower() == "true"
    
    print("\n=== Extract Staging Script ===")
    print("Arguments:")
    print(f"- Input file: {input_file}")
    print(f"- Output file: {output_file}")
    print(f"- Using LLM: {use_llm}")
    
    # Make sure output directory exists
    print("\nCreating output directory if needed...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Run extraction
    extract_staging(input_file, output_file, use_llm)
    
    print(f"\n=== Extraction Complete ===")
    print(f"Results saved to: {output_file}")