import pyarrow.parquet as pq
from pathlib import Path
from staging_utils import StagingExtractor
import pyarrow as pa
import os
import sys

def process_batch(batch, extractor):
    """Process a batch of records"""
    results = []
    for note in batch.to_pylist():
        findings = extractor.extract_staging(
            note['note_text'],
            note['deid_service_date']
        )
        for finding in findings:
            results.append({
                'patientdurablekey': note['patientdurablekey'],
                'note_date': note['deid_service_date'],
                **finding
            })
    return pa.Table.from_pylist(results)

def extract_staging(input_file, output_file, use_llm=True):
    """Stream process notes and extract staging"""
    print(f"Starting extraction with input: {input_file}, output: {output_file}, use_llm: {use_llm}")
    extractor = StagingExtractor()
    extractor.use_llm = use_llm
    
    # Create streaming reader
    reader = pq.ParquetDataset(input_file).get_fragments()
    
    # Process files sequentially
    with pq.ParquetWriter(output_file, schema=pa.schema([
        ('patientdurablekey', pa.string()),
        ('note_date', pa.timestamp('s')),
        ('stage', pa.string()),
        ('system', pa.string()),
        ('confidence', pa.float32()),
        ('evidence', pa.string())
    ])) as writer:
        
        for fragment in reader:
            batch = fragment.to_batches()[0]  # Process 1 file at a time
            print(f"Processing batch with {len(batch)} records")
            result_table = process_batch(batch, extractor)
            if result_table.num_rows > 0:
                print(f"Writing {result_table.num_rows} results to output file")
                writer.write_table(result_table)
            else:
                print("No results found in this batch")

if __name__ == "__main__":
    # Get command line arguments
    if len(sys.argv) != 4:
        print("Usage: python extract_staging.py <input_file> <output_file> <use_llm>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    use_llm = sys.argv[3].lower() == "true"
    
    print(f"Running extract_staging.py with arguments:")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Using LLM: {use_llm}")
    
    # Make sure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Run extraction
    extract_staging(input_file, output_file, use_llm)
    
    print(f"Extraction complete. Results saved to {output_file}")