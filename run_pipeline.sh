#!/bin/bash
# Step 1: Filter encounters
python filter_encounters.py \
  --patient_ids data/input/patient_ids_with_durablekey.csv \
  --input_dir /wynton/protected/project/ic/data/parquet/DEID_CDW/encounterfact \
  --output_dir /wynton/protected/home/zack/brtan/Stage_2_Staging_Extractor/data/output/filtered_encounters

# Step 2: Filter notes
python filter_notes.py \
  --encounters_dir /wynton/protected/home/zack/brtan/Stage_2_Staging_Extractor/data/output/filtered_encounters \
  --note_meta_dir /wynton/protected/project/ic/data/parquet/DEID_CDW/note_metadata \
  --note_text_dir /wynton/protected/project/ic/data/parquet/DEID_CDW/note_text \
  --output_dir /scratch/brtan/filtered_notes

# Step 3: Extract staging
python extract_staging.py \
  --input_dir /scratch/brtan/filtered_notes \
  --output_path data/output/staging_results.parquet \
  --use_llm  # Remove for regex-only

# Optional: Cleanup
rm -rf /scratch/brtan/filtered_encounters
rm -rf /scratch/brtan/filtered_notes

# Primary Data Directories
# /wynton/protected/project/ic/data/parquet/DEID_CDW/
# /wynton/protected/project/ic/data/parquet/DEID_OMOP/
