#!/usr/bin/env python3
"""
This module extracts the first mention of cancer staging information from clinical notes.
For each note, it returns a structured JSON object with:
note_id: a unique identifier for the note
stage: the earliest staging text found (e.g., "Stage II", "T2N0M0")
date: the associated date (in YYYY-MM-DD) if present in close proximity; otherwise null
Usage:
python extract_staging.py [--input_dir <dir>] [--output_dir <dir>] [--use_llm] [--llm_model <model_name>]
Requirements:
• Python 3
• Standard library modules (re, json, argparse)
• Optional: transformers library for LLM extraction (pip install transformers)
The extraction uses regex to find matches for:
"Stage" followed by a roman numeral (I, II, III, or IV).
TNM descriptors like "T1N0M0"
Dates in the "YYYY-MM-DD" format
For each clinical note, only the first encountered staging reference is used. An optional LLM-based extraction
is provided for cases where regex might be insufficient. This uses a text-generation model (via Hugging Face
transformers) to parse the text.
Example clinical notes and usage are provided in the sample_notes within the code.
"""
import re
import json
import argparse
from datetime import datetime
from dateutil.parser import parse
import sys
from typing import Any
import pandas as pd
import pyarrow.parquet as pq
import os

# Try to import transformers pipeline if available
try:
    from transformers import pipeline
except ImportError:
    pipeline = None

def extract_stage_and_date(text: str) -> tuple[str | None, str | None]:
    """
    Scans the provided text and extracts:
    The first mention of staging information (either "Stage" followed by I/II/III/IV or a TNM descriptor)
    A date (YYYY-MM-DD format) in a window around the match.
    Returns:
    tuple: (stage_string or None, date_string or None)
    """
    # Define regex patterns
    stage_pattern = re.compile(r'\bStage\s+(I|II|III|IV)\b', re.IGNORECASE)
    tnm_pattern = re.compile(r'\bT\d+N\d+M\d+\b', re.IGNORECASE)
    # OMOP CDM compatible date pattern (YYYY-MM-DD with optional time component)
    date_pattern = re.compile(
        r'\d{4}-'          # Year
        r'\d{1,2}-'        # Month (1-2 digits)
        r'\d{1,2}'         # Day (1-2 digits)
        r'(?:[ T]'         # Date/time separator
        r'\d{1,2}:'        # Hours
        r'\d{1,2}:'        # Minutes 
        r'\d{1,2}'         # Seconds
        r'(?:\.\d+)?)?'    # Fractional seconds
        r'(?:Z|[+-]\d{1,2}:?\d{2})?'  # Timezone
    )

    # Find all matches (as iterator objects)
    stage_matches = list(stage_pattern.finditer(text))
    tnm_matches = list(tnm_pattern.finditer(text))

    # Combine matches with their starting index and type
    matches = []
    for m in stage_matches:
        matches.append((m.start(), m, 'stage'))
    for m in tnm_matches:
        matches.append((m.start(), m, 'tnm'))

    # Sort by order of appearance (start index)
    matches.sort(key=lambda x: x[0])

    # If no staging information is found, return (None, None)
    if not matches:
        return None, None

    # Get the first match (earliest mention)
    first_match = matches[0]
    stage_str = first_match[1].group(0)  # text of the staging reference

    # Search for a date within a window around the match:
    context_window = 50  # characters before and after the match
    win_start = max(0, first_match[1].start() - context_window)
    win_end = min(len(text), first_match[1].end() + context_window)
    window_text = text[win_start:win_end]

    date_match = date_pattern.search(window_text)
    if date_match:
        raw_date = date_match.group(0)
        # Convert to OMOP CDM date format (YYYY-MM-DD)
        try:
            # Parse any recognized date format
            dt = parse(raw_date, fuzzy=True)
            date_str = dt.strftime("%Y-%m-%d")
        except (ValueError, OverflowError, TypeError) as e:
            date_str = None
            print(f"Date parsing error: {e}", file=sys.stderr)
    else:
        date_str = None

    return stage_str, date_str

def extract_stage_and_date_llm(text, note_id, llm_pipeline):
    """
    Uses an LLM to extract the first mention of cancer staging information and its date from the text.
    Returns:
    tuple: (stage_string or None, date_string or None)
    The LLM is prompted to return a JSON object with 'note_id', 'stage', and 'date'.
    """
    prompt = (
        f"Extract the first mention of cancer staging information and its associated date from the following clinical note. "
        f"Consider staging information in the form 'Stage I/II/III/IV' or TNM descriptors (e.g., T1N0M0). "
        f"Return a JSON object with the keys 'note_id', 'stage', and 'date' (date in YYYY-MM-DD format). "
        f"If no staging information is found, set 'stage' to null, and if no date is found, set 'date' to null. "
        f"Only return the first staging reference encountered in the text.\n\n"
        f"Clinical note text:\n'''\n{text}\n'''"
    )
    result = llm_pipeline(prompt, max_length=256, truncation=True)
    response_text = result[0]['generated_text']
    try:
        json_start = response_text.find('{')
        json_end = response_text.rfind('}')
        json_str = response_text[json_start:json_end+1]
        json_obj = json.loads(json_str)
        json_obj['note_id'] = note_id
        stage = json_obj.get('stage')
        date = json_obj.get('date')
        return stage, date
    except Exception as e:
        return None, None

def process_note(
    note: dict,
    use_llm: bool = False,
    llm_pipeline_instance: Any | None = None
) -> dict[str, str | None]:
    """
    Processes a single clinical note.
    Arguments:
    note (dict): must include 'note_id' and 'text'
    use_llm (bool): whether to use LLM-based extraction
    llm_pipeline_instance: the initialized LLM pipeline (required if use_llm is True)
    Returns:
    dict: { "note_id": <id>, "stage": <extracted stage string or None>, "date": <extracted date or None> }
    """
    if use_llm and llm_pipeline_instance is not None:
        stage, date = extract_stage_and_date_llm(note['text'], note['note_id'], llm_pipeline_instance)
    else:
        stage, date = extract_stage_and_date(note['text'])
    
    result = {
        "note_id": note['note_id'],
        "stage": stage,
        "date": date
    }
    return result

def load_notes_from_parquet(input_dir):
    """Load clinical notes from Parquet files"""
    notes = []
    for file in os.listdir(input_dir):
        if file.endswith(".parquet"):
            path = os.path.join(input_dir, file)
            df = pd.read_parquet(path)
            notes.extend(df[['deid_note_key', 'note_text']]
                         .rename(columns={'note_text': 'text', 'deid_note_key': 'note_id'})
                         .to_dict('records'))
    return notes

def main():
    """
    Main entry point for processing the clinical notes.
    Run this script directly to see the extraction on sample notes.
    """
    parser = argparse.ArgumentParser(description='Extract cancer staging from clinical notes.')
    parser.add_argument('--input_dir', type=str, 
                       help='Path to directory containing note_text Parquet files')
    parser.add_argument('--output_dir', type=str,
                       help='Output directory for processed Parquet files')
    parser.add_argument('--use_llm', action='store_true', help='Use LLM-based extraction instead of regex.')
    parser.add_argument('--llm_model', type=str, default="gpt-4o", help='LLM model name to use if --use_llm is set.')
    args = parser.parse_args()
    use_llm = args.use_llm
    llm_pipeline_instance = None
    if use_llm:
        if pipeline is None:
            print("Transformers library is not installed. Please install it (pip install transformers) to use LLM extraction.")
            return
        llm_pipeline_instance = pipeline("text2text-generation", model=args.llm_model)
    if args.input_dir:
        notes = load_notes_from_parquet(args.input_dir)
    else:
        notes = [
            {
                "note_id": "note1",
                "text": "Patient presents with Stage II cancer. Biopsy performed on 2022-02-25 and follow-up imaging on 2022-03-01."
            },
            {
                "note_id": "note2",
                "text": "Findings: T2N0M0 identified during diagnostic workup on 2021-09-15. No further staging details were available."
            },
            {
                "note_id": "note3",
                "text": "This note contains no explicit reference to cancer staging. The patient is under routine surveillance."
            },
            {
                "note_id": "note4",
                "text": "Preliminary note: Stage I (dx date: 2022-05-10) but later updated to Stage II; however, only the first reference matters."
            },
            {
                "note_id": "note5",
                "text": ("Additional details: The tumor was assessed, and while T1N1M1 was observed during initial evaluation, "
                         "an alternative note mentioned Stage IV later on. The date 2020-12-31 is provided near the later mention, "
                         "but only the first staging text should be extracted.")
            }
        ]
    results = []
    for note in notes:
        result = process_note(note, use_llm=use_llm, llm_pipeline_instance=llm_pipeline_instance)
        results.append(result)
    if args.output_dir:
        for res in results:
            output_file = f"{args.output_dir}/{res['date']}.parquet"
            df = pd.DataFrame([res])
            pq.write_table(df.to_arrow(), output_file)
        print(f"Results written to {args.output_dir}")
    else:
        for res in results:
            print(json.dumps(res))

if __name__ == "__main__":
    main()
