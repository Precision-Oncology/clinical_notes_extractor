# Staging Extraction Array Job

This directory contains qsub scripts for submitting jobs to the SGE cluster for extracting cancer staging information from clinical notes.

## Scripts

- `qsub_extract_staging_array.sh`: An array job script that processes 100 parquet files and extracts staging information using LLM inference.

## Usage

To submit the array job for staging extraction:

```bash
qsub qsub_script/qsub_extract_staging_array.sh
```

This will:
1. Submit 100 parallel jobs to the SGE cluster
2. Each job processes one parquet file from `/wynton/protected/home/zack/brtan/Stage_2_Staging_Extractor/data/output/filtered_notes/final/filtered_notes_batch_*.parquet`
3. Output files will be saved to `/wynton/protected/home/zack/brtan/Stage_2_Staging_Extractor/data/output/staging_results/staging_results_batch_*.parquet`
4. Logs will be created in the `qsub_logs` directory

## Configuration

The script uses the following configuration:
- Runtime: 29 minutes 50 seconds per job
- Memory: 32GB per job
- GPU Memory: 30GB per job
- Scratch space: 30GB
- Queue: gpu.q

## Output

The output parquet files will contain the following schema:
- `patientdurablekey`: Patient identifier
- `note_date`: Timestamp of the note
- `stage`: Extracted staging information
- `system`: Staging system used
- `confidence`: Confidence score of the extraction
- `evidence`: Evidence text from which the staging was extracted

## Combining Results

After all jobs complete, you may want to combine the individual parquet files into a single file using a tool like `pyarrow`.

Example:
```python
import pyarrow.parquet as pq
import glob
import os

# Get all result files
result_files = glob.glob('/wynton/protected/home/zack/brtan/Stage_2_Staging_Extractor/data/output/staging_results/staging_results_batch_*.parquet')

# Read and combine
tables = [pq.read_table(file) for file in result_files]
combined = pa.concat_tables(tables)

# Write combined table
pq.write_table(combined, '/wynton/protected/home/zack/brtan/Stage_2_Staging_Extractor/data/output/staging_results/combined_staging_results.parquet')
``` 