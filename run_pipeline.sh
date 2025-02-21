#!/bin/bash
# Step 1: Filter encounters
python filter_encounters.py \
  --patient_ids data/patient_ids.csv \
  --input_dir data/encounterfact \
  --output_dir temp/filtered_encounters

# Step 2: Filter notes
python filter_notes.py \
  --encounters_dir temp/filtered_encounters \
  --note_meta_dir data/note_metadata \
  --note_text_dir data/note_text \
  --output_dir temp/filtered_notes

# Step 3: Extract staging
python extract_staging.py \
  --input_dir temp/filtered_notes \
  --output_path final/staging_results.parquet \
  --use_llm  # Remove for regex-only

# Optional: Cleanup
rm -rf temp/ 