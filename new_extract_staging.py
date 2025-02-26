#!/usr/bin/env python3
"""
Staging Information Extractor


# TODO: I don't want to be making 20,000 calls.... for each note text. So i either concatenate the note texts for each patient together, or I use a batch process, or I pre-filter the note texts.
- Or both, pre-filter the note texts then batch process by patient!
- Use local model found at /wynton/protected/home/zack/brtan/models/Llama-3.1-8B instead of remote model 

This script processes clinical notes to extract cancer staging information using 
a Language Model (LLM) approach. It processes parquet files containing clinical notes,
extracts staging information using the local Llama-3.1-8B model, and outputs
a filtered dataset containing only notes with staging information.

Usage:
    # First activate the virtual environment:
    source /wynton/protected/home/zack/brtan/Virtual_Environments/dask_distribution_env/bin/activate
    
    # Then run the script:
    python new_extract_staging.py <batch_number>
    
    # For testing the parsing logic:
    python new_extract_staging.py --test
    
    # For benchmarking the parsing performance:
    python new_extract_staging.py --benchmark
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union
import time
import logging
import argparse
from tqdm import tqdm
import pyarrow.parquet as pq
import pyarrow as pa
import re

# Try to load dotenv if available, but continue if not
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("dotenv module not found, skipping .env file loading")
    pass

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StagingExtractor:
    """Class to extract staging information from clinical notes using LLM."""
    
    def __init__(self):
        """Initialize the staging extractor."""
        self.llm_model = None
        self.tokenizer = None
        # Path to local model
        self.model_path = "/wynton/protected/home/zack/brtan/models/Llama-3.1-8B"
        
    def _load_llm(self):
        """Lazy-load LLM model from local path."""
        if not self.llm_model:
            logger.info("Loading local model...")
            try:
                # Load model directly from local path
                try:
                    from transformers import AutoTokenizer, AutoModelForCausalLM
                    import torch
                except ImportError:
                    logger.error("Failed to import transformers or torch. Make sure you've activated the correct environment:")
                    logger.error("source /wynton/protected/home/zack/brtan/Virtual_Environments/dask_distribution_env/bin/activate")
                    raise
                
                # Check if model path exists
                if not os.path.exists(self.model_path):
                    logger.error(f"Model path not found: {self.model_path}")
                    raise FileNotFoundError(f"Model path not found: {self.model_path}")
                
                logger.info(f"Loading model from: {self.model_path}")
                
                # Load tokenizer and model from local path
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    local_files_only=True
                )
                logger.info("✓ Tokenizer loaded successfully")
                
                # Load model with appropriate device placement
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                logger.info(f"Using device: {device}")
                
                # Load model with lower precision for memory efficiency if using GPU
                if device.type == "cuda":
                    self.llm_model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        local_files_only=True,
                        torch_dtype=torch.float16,  # Use half precision for GPU
                        device_map="auto"  # Automatically distribute across available GPUs
                    )
                else:
                    self.llm_model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        local_files_only=True
                    ).to(device)
                
                logger.info("✓ Model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                logger.error("Make sure you've activated the correct environment:")
                logger.error("source /wynton/protected/home/zack/brtan/Virtual_Environments/dask_distribution_env/bin/activate")
                raise
    
    def _llm_extract(self, text: str) -> Dict[str, Optional[str]]:
        """
        Extract staging information from clinical note text using LLM.
        
        Args:
            text: The clinical note text
            
        Returns:
            Dict with extracted staging information or NA if none found
        """
        # Load model if not already loaded
        self._load_llm()
        
        # Construct prompt for staging extraction - adapted for Llama-3.1-8B
        prompt = f"""<|system|>
You are a medical assistant that extracts cancer staging information from clinical notes.
<|user|>
Analyze this clinical note and extract cancer staging information:

{text}

Respond with EXACTLY ONE of these formats (no additional text):
1. "NA" if no staging information exists
2. "Stage: [stage]" for general staging (e.g., "Stage: IIB") 
3. "TNM: [classification]" for TNM classifications (e.g., "TNM: T2N1M0")
<|assistant|>"""

        try:
            # Generate response using the model
            import torch
            
            # Handle very long texts by truncating to fit in context window
            max_length = self.tokenizer.model_max_length
            inputs = self.tokenizer(prompt, truncation=True, max_length=max_length - 100, return_tensors="pt")
            
            # Move inputs to same device as model
            inputs = {k: v.to(self.llm_model.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs, 
                    max_new_tokens=50,      # Reduced since we expect short responses
                    temperature=0.1,        # Low temperature for more focused output
                    do_sample=False,        # Deterministic generation
                    pad_token_id=self.tokenizer.eos_token_id,  # Ensure proper padding
                )
            
            # Decode output
            response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            response = response.strip()
            logger.debug(f"LLM response: {response}")
            
            # Parse the response
            result = self._parse_llm_response(response)
            return result
            
        except Exception as e:
            logger.error(f"Error in LLM extraction: {e}")
            logger.error(f"Text snippet: {text[:100]}...")
            return {"stage": None, "system": None}
    
    def _parse_llm_response(self, response: str) -> Dict[str, Optional[str]]:
        """
        Parse the LLM response to extract structured staging information.
        Simple and fast parsing based on expected LLM output format.
        
        Args:
            response: The raw text response from the LLM
            
        Returns:
            Dict with stage and system information, or None values if not found
        """
        # Default result if no staging information
        result = {"stage": None, "system": None}
        
        # Quick check for NA response
        if "NA" in response:
            return result
            
        # Fast check for TNM format
        if "TNM:" in response:
            # Extract everything after "TNM:"
            tnm_text = response.split("TNM:", 1)[1].strip()
            # Take only the first line if multiple lines
            if "\n" in tnm_text:
                tnm_text = tnm_text.split("\n", 1)[0].strip()
            result["stage"] = tnm_text
            result["system"] = "TNM"
            return result
            
        # Fast check for Stage format
        if "Stage:" in response:
            # Extract everything after "Stage:"
            stage_text = response.split("Stage:", 1)[1].strip()
            # Take only the first line if multiple lines
            if "\n" in stage_text:
                stage_text = stage_text.split("\n", 1)[0].strip()
            result["stage"] = stage_text
            result["system"] = "General"
            return result
            
        # Direct TNM pattern as fallback (e.g., T2N1M0)
        if "T" in response and "N" in response and "M" in response:
            for word in response.split():
                if word.startswith("T") and "N" in word and "M" in word:
                    result["stage"] = word.strip(".,;:()")
                    result["system"] = "TNM"
                    return result
        
        return result

def process_file(file_path: str, extractor: StagingExtractor) -> pd.DataFrame:
    """
    Process a single parquet file to extract staging information.
    
    Args:
        file_path: Path to the parquet file
        extractor: StagingExtractor instance
        
    Returns:
        DataFrame with only rows containing staging information
    """
    try:
        logger.info(f"Reading file: {file_path}")
        df = pd.read_parquet(file_path)
        
        logger.info(f"Processing {len(df)} notes...")
        
        # Create empty columns for staging info
        df['stage'] = None
        df['system'] = None
        
        # Track rows with staging info
        has_staging = []
        
        # Process each note
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting staging info"):
            text = row['note_text']
            if not isinstance(text, str) or not text.strip():
                continue
                
            # Extract staging information
            staging_info = extractor._llm_extract(text)
            
            # Store results in the dataframe
            df.at[idx, 'stage'] = staging_info.get('stage')
            df.at[idx, 'system'] = staging_info.get('system')
            
            # Track if this note has staging info
            has_staging.append(idx if staging_info.get('stage') is not None else None)
        
        # Filter to only rows with staging information
        has_staging = [i for i in has_staging if i is not None]
        if has_staging:
            result_df = df.loc[has_staging].copy()
            logger.info(f"Found {len(result_df)} notes with staging information")
            return result_df
        else:
            logger.info("No staging information found in this file")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return pd.DataFrame()

def test_parsing_logic():
    """
    Test function to verify the parsing logic works with different LLM output formats.
    This function tests only the parsing logic without loading the model.
    """
    # Create a minimal extractor instance without loading the model
    class MinimalExtractor:
        def _parse_llm_response(self, response):
            # Copy of the parsing logic for testing
            result = {"stage": None, "system": None}
            
            # Quick check for NA response
            if "NA" in response:
                return result
                
            # Fast check for TNM format
            if "TNM:" in response:
                # Extract everything after "TNM:"
                tnm_text = response.split("TNM:", 1)[1].strip()
                # Take only the first line if multiple lines
                if "\n" in tnm_text:
                    tnm_text = tnm_text.split("\n", 1)[0].strip()
                result["stage"] = tnm_text
                result["system"] = "TNM"
                return result
                
            # Fast check for Stage format
            if "Stage:" in response:
                # Extract everything after "Stage:"
                stage_text = response.split("Stage:", 1)[1].strip()
                # Take only the first line if multiple lines
                if "\n" in stage_text:
                    stage_text = stage_text.split("\n", 1)[0].strip()
                result["stage"] = stage_text
                result["system"] = "General"
                return result
                
            # Direct TNM pattern as fallback (e.g., T2N1M0)
            if "T" in response and "N" in response and "M" in response:
                for word in response.split():
                    if word.startswith("T") and "N" in word and "M" in word:
                        result["stage"] = word.strip(".,;:()")
                        result["system"] = "TNM"
                        return result
            
            return result
    
    extractor = MinimalExtractor()
    
    test_cases = [
        # Simple format responses - these match our expected LLM output format
        {"response": "NA", "expected": {"stage": None, "system": None}},
        {"response": "Stage: IIB", "expected": {"stage": "IIB", "system": "General"}},
        {"response": "TNM: T2N1M0", "expected": {"stage": "T2N1M0", "system": "TNM"}},
        
        # Variations that our parser should still handle
        {"response": "TNM: T3N2M1\n", "expected": {"stage": "T3N2M1", "system": "TNM"}},
        {"response": "Stage: IV\nAdditional notes", "expected": {"stage": "IV", "system": "General"}},
        {"response": "T3N2M1", "expected": {"stage": "T3N2M1", "system": "TNM"}},
        
        # Llama-3.1-8B specific formats
        {"response": "Based on the clinical note, TNM: T2N0M0", "expected": {"stage": "T2N0M0", "system": "TNM"}},
        {"response": "After reviewing the note, Stage: IIIA", "expected": {"stage": "IIIA", "system": "General"}},
    ]
    
    print("\nTesting simplified parsing logic with different LLM output formats:")
    print("=" * 60)
    
    for i, test in enumerate(test_cases, 1):
        result = extractor._parse_llm_response(test["response"])
        expected = test["expected"]
        
        # Check if result matches expected
        success = (result["stage"] == expected["stage"] and 
                  (result["system"] == expected["system"] or 
                   (result["system"] is None and expected["system"] is None)))
        
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"\nTest {i}: {status}")
        print(f"Input: {test['response']}")
        print(f"Expected: {expected}")
        print(f"Got: {result}")
        
    print("\nTesting complete!")

def benchmark_parsing():
    """
    Benchmark the performance of the parsing logic.
    This function tests only the parsing logic without loading the model.
    """
    import time
    
    # Use the same minimal extractor as in test_parsing_logic
    class MinimalExtractor:
        def _parse_llm_response(self, response):
            # Copy of the parsing logic for testing
            result = {"stage": None, "system": None}
            
            # Quick check for NA response
            if "NA" in response:
                return result
                
            # Fast check for TNM format
            if "TNM:" in response:
                # Extract everything after "TNM:"
                tnm_text = response.split("TNM:", 1)[1].strip()
                # Take only the first line if multiple lines
                if "\n" in tnm_text:
                    tnm_text = tnm_text.split("\n", 1)[0].strip()
                result["stage"] = tnm_text
                result["system"] = "TNM"
                return result
                
            # Fast check for Stage format
            if "Stage:" in response:
                # Extract everything after "Stage:"
                stage_text = response.split("Stage:", 1)[1].strip()
                # Take only the first line if multiple lines
                if "\n" in stage_text:
                    stage_text = stage_text.split("\n", 1)[0].strip()
                result["stage"] = stage_text
                result["system"] = "General"
                return result
                
            # Direct TNM pattern as fallback (e.g., T2N1M0)
            if "T" in response and "N" in response and "M" in response:
                for word in response.split():
                    if word.startswith("T") and "N" in word and "M" in word:
                        result["stage"] = word.strip(".,;:()")
                        result["system"] = "TNM"
                        return result
            
            return result
    
    extractor = MinimalExtractor()
    
    # Create a large list of test responses
    test_responses = []
    for _ in range(1000):
        test_responses.extend([
            "NA",
            "Stage: IIB",
            "TNM: T2N1M0",
            "TNM: T3N2M1\nAdditional notes that should be ignored",
            "Stage: IV\nThis is a stage IV cancer",
            "T3N2M1",
            "Based on the clinical note, TNM: T2N0M0",
            "After reviewing the note, Stage: IIIA"
        ])
    
    # Benchmark the parsing
    start_time = time.time()
    for response in test_responses:
        extractor._parse_llm_response(response)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time_per_parse = total_time / len(test_responses) * 1000  # in milliseconds
    
    print("\nParsing Performance Benchmark:")
    print("=" * 60)
    print(f"Total responses parsed: {len(test_responses)}")
    print(f"Total time: {total_time:.4f} seconds")
    print(f"Average time per parse: {avg_time_per_parse:.4f} milliseconds")
    print(f"Parses per second: {len(test_responses) / total_time:.2f}")
    print("=" * 60)

def main():
    """Main function to run the staging extraction pipeline."""
    # Configure argument parser
    parser = argparse.ArgumentParser(description='Process a single batch of clinical notes')
    parser.add_argument('batch_number', type=int, help='Batch number to process')
    args = parser.parse_args()
    
    # Configure paths
    input_file = f"/wynton/protected/home/zack/brtan/Stage_2_Staging_Extractor/data/output/filtered_notes/final/filtered_notes_batch_{args.batch_number}.parquet"
    output_dir = "/wynton/protected/home/zack/brtan/Stage_2_Staging_Extractor/data/output/staging_results"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Starting staging extraction pipeline")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output directory: {output_dir}")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return
    
    # Initialize the staging extractor
    extractor = StagingExtractor()
    
    # Process the file
    result_df = process_file(input_file, extractor)
    
    if not result_df.empty:
        # Save results
        output_file = os.path.join(output_dir, f"staging_results_batch_{args.batch_number}.parquet")
        result_df.to_parquet(output_file, index=False)
        logger.info(f"Saved results to {output_file}")
        
        # Also save as CSV for easier inspection
        csv_file = os.path.join(output_dir, f"staging_results_batch_{args.batch_number}.csv")
        result_df.to_csv(csv_file, index=False)
        logger.info(f"Also saved as CSV to {csv_file}")
        logger.info(f"Total notes with staging information: {len(result_df)}")
    else:
        logger.info("No staging information found in the file")

if __name__ == "__main__":
    import sys
    
    # Check if we're running tests or benchmarks
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_parsing_logic()
        if len(sys.argv) > 2 and sys.argv[2] == "--benchmark":
            benchmark_parsing()
        sys.exit(0)
    elif len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark_parsing()
        sys.exit(0)
    
    # Normal execution
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    logger.info(f"Pipeline completed in {elapsed_time:.2f} seconds")
