import pyarrow.parquet as pq
from pathlib import Path
from staging_utils import StagingExtractor
import pyarrow as pa

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

def extract_staging(input_dir: str, output_path: str, use_llm: bool = False):
    """Stream process notes and extract staging"""
    extractor = StagingExtractor()
    extractor.use_llm = use_llm
    
    # Create streaming reader
    reader = pq.ParquetDataset(input_dir).get_fragments()
    
    # Process files sequentially
    with pq.ParquetWriter(output_path, schema=pa.schema([
        ('patientdurablekey', pa.string()),
        ('note_date', pa.timestamp('s')),
        ('stage', pa.string()),
        ('system', pa.string()),
        ('confidence', pa.float32()),
        ('evidence', pa.string())
    ])) as writer:
        
        for fragment in reader:
            batch = fragment.to_batches()[0]  # Process 1 file at a time
            result_table = process_batch(batch, extractor)
            if result_table.num_rows > 0:
                writer.write_table(result_table) 