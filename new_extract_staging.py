#!/usr/bin/env python3
"""
Staging Information Extractor


# TODO: I don't want to be making 20,000 calls.... for each note text. So i either concatenate the note texts for each patient together, or I use a batch process, or I pre-filter the note texts.
- Or both, pre-filter the note texts then batch process by patient!

This script processes clinical notes to extract cancer staging information using 
a Language Model (LLM) approach. It processes parquet files containing clinical notes,
extracts staging information using the deepseek-r1-distill-llama-8b model, and outputs
a filtered dataset containing only notes with staging information.
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
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
        
    def _load_llm(self):
        """Lazy-load LLM model."""
        if not self.llm_model:
            logger.info("Loading model...")
            try:
                # Load model directly
                from transformers import AutoTokenizer, AutoModelForCausalLM
                
                # Get token from environment variable
                token = os.environ.get('HUGGINGFACE_TOKEN')
                if not token:
                    logger.warning("No Hugging Face token found in .env file (HUGGINGFACE_TOKEN)")
                
                # Get model name from environment variable or use default
                model_name = os.environ.get('HUGGINGFACE_MODEL', "deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
                logger.info(f"Loading model: {model_name}")
                
                # Load tokenizer and model with token
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
                logger.info("✓ Tokenizer loaded successfully")
                
                self.llm_model = AutoModelForCausalLM.from_pretrained(model_name, token=token)
                logger.info("✓ Model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
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
        
        # Construct prompt for staging extraction
        prompt = f"""Analyze this clinical note and extract cancer staging information:
        {text}

        Output format:
        Most notes will not have staging information, so the output should be NA.
        If there is staging information, output the following:
        - Stage: [detected stage or TNM classification] E.g., TNM staging like T*N*M* OR General staging like Stage I, Stage II, Stage IIA, Stage IIB, Stage III, Stage IV, Stage 1, Stage 2, Stage 3, Stage 4
        - System: [staging system used: TNM or General Staging System]
        """

        try:
            # Generate response using the model
            import torch
            
            # Handle very long texts by truncating to fit in context window
            max_length = self.tokenizer.model_max_length
            inputs = self.tokenizer(prompt, truncation=True, max_length=max_length - 200, return_tensors="pt")
            
            # Move inputs to same device as model
            inputs = {k: v.to(self.llm_model.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs, 
                    max_new_tokens=200,
                    temperature=0.1,  # Low temperature for more focused output
                    do_sample=False,  # Deterministic generation
                )
            
            # Decode output
            response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            logger.debug(f"LLM response: {response}")
            
            # Parse the response
            result = self._parse_llm_response(response)
            return result
            
        except Exception as e:
            logger.error(f"Error in LLM extraction: {e}")
            return {"stage": None, "system": None}
    
    def _parse_llm_response(self, response: str) -> Dict[str, Optional[str]]:
        """
        Parse the LLM response to extract structured staging information.
        
        Args:
            response: The raw text response from the LLM
            
        Returns:
            Dict with stage and system information, or None values if not found
        """
        # Default result if no staging information
        result = {"stage": None, "system": None}
        
        # Check if response indicates no staging info
        if "NA" in response or "No staging information" in response:
            return result
        
        # Look for Stage information
        stage_match = None
        system_match = None
        
        for line in response.split('\n'):
            line = line.strip()
            
            # Look for stage information
            if line.startswith("- Stage:") or line.startswith("Stage:"):
                stage_text = line.split(":", 1)[1].strip()
                if stage_text and stage_text.lower() not in ["na", "none", "not available", "not applicable"]:
                    stage_match = stage_text
            
            # Look for system information
            if line.startswith("- System:") or line.startswith("System:"):
                system_text = line.split(":", 1)[1].strip()
                if system_text and system_text.lower() not in ["na", "none", "not available", "not applicable"]:
                    system_match = system_text
        
        # Update result if we found staging information
        if stage_match:
            result["stage"] = stage_match
            result["system"] = system_match if system_match else "Unknown"
            
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
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    logger.info(f"Pipeline completed in {elapsed_time:.2f} seconds")
