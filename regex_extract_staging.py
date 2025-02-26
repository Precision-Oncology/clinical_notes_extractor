#!/usr/bin/env python3
"""
Regex-Only Staging Information Extractor

This script extracts cancer staging information from clinical notes using only regex patterns.
It's designed to be lightweight, efficient, and have minimal dependencies.

The script processes parquet files containing clinical notes and outputs a filtered dataset
containing only notes with staging information.

Usage:
    python regex_extract_staging.py <input_file> <output_file>
    
    # Process a specific batch:
    python regex_extract_staging.py /path/to/filtered_notes_batch_5.parquet /path/to/output.parquet
    
    # For benchmarking performance:
    python regex_extract_staging.py --benchmark
"""

import os
import re
import time
import argparse
import logging
import pandas as pd
from typing import List, Dict, Optional
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RegexStagingExtractor:
    """Class to extract staging information from clinical notes using regex only."""
    
    def __init__(self, additional_patterns=None):
        """
        Initialize the regex staging extractor with common cancer staging patterns.
        
        Args:
            additional_patterns: Optional dictionary of additional regex patterns to include
        """
        # Core patterns for TNM and general staging
        self.regex_patterns = {
            # Standard TNM pattern (e.g., T2N1M0)
            'tnm_standard': re.compile(r'\b(T[0-4isX]{1,3}N[0-3MX]{1,3}M[0-1X]{1})\b', re.I),
            
            # Clinical and pathological TNM prefixes (e.g., cT2N1M0, pT3N0M0)
            'tnm_prefixed': re.compile(r'\b([cp]T[0-4isX]{1,3}N[0-3MX]{1,3}M[0-1X]{1})\b', re.I),
            
            # Separated TNM components (e.g., T2, N1, M0)
            'tnm_components': re.compile(r'\b(T[0-4isX]{1,3})[,\s.].*\b(N[0-3MX]{1,3})[,\s.].*\b(M[0-1X]{1})\b', re.I),
            
            # Standard stage format (e.g., Stage IIB, Stage IV)
            'stage_standard': re.compile(r'\b(Stage\s+[0IVX]{1,4}[A-C]?)\b', re.I),
            
            # AJCC stage format (e.g., AJCC Stage IIB)
            'stage_ajcc': re.compile(r'\b(AJCC\s+Stage\s+[0IVX]{1,4}[A-C]?)\b', re.I),
            
            # Stage with colon format (e.g., Stage: IIB)
            'stage_colon': re.compile(r'\bStage\s*:\s*([0IVX]{1,4}[A-C]?)\b', re.I),
            
            # Stage without word "stage" (context must contain "stage" or "staging" within 100 chars)
            'roman_numeral_stage': re.compile(r'(?i)(?=.*?\b(?:stage|staging)\b.{0,100})\b([IVX]{1,4}[A-C]?)\b(?!.*?century)', re.I)
        }
        
        # Add any additional patterns provided
        if additional_patterns:
            self.regex_patterns.update(additional_patterns)
    
    def extract_staging(self, text: str) -> List[Dict]:
        """
        Extract staging information from clinical note text using regex patterns.
        
        Args:
            text: The clinical note text
            
        Returns:
            List of dictionaries with extracted staging information
        """
        if not isinstance(text, str) or not text.strip():
            return []
            
        findings = []
        
        # Process TNM standard format
        for match in self.regex_patterns['tnm_standard'].finditer(text):
            findings.append({
                'stage': match.group(1).upper(),
                'system': 'TNM',
                'evidence': self._get_context(text, match.start(), match.end())
            })
        
        # Process TNM with clinical (c) or pathological (p) prefix
        for match in self.regex_patterns['tnm_prefixed'].finditer(text):
            findings.append({
                'stage': match.group(1).upper(),
                'system': 'TNM',
                'evidence': self._get_context(text, match.start(), match.end())
            })
        
        # Process separated TNM components (more complex)
        for match in self.regex_patterns['tnm_components'].finditer(text):
            # Combine the separate components
            combined_tnm = f"{match.group(1)}{match.group(2)}{match.group(3)}"
            findings.append({
                'stage': combined_tnm.upper(),
                'system': 'TNM',
                'evidence': self._get_context(text, match.start(), match.end())
            })
        
        # Process standard stage format
        for match in self.regex_patterns['stage_standard'].finditer(text):
            findings.append({
                'stage': match.group(1).title(),
                'system': 'General',
                'evidence': self._get_context(text, match.start(), match.end())
            })
        
        # Process AJCC stage format
        for match in self.regex_patterns['stage_ajcc'].finditer(text):
            findings.append({
                'stage': match.group(1).title(),
                'system': 'AJCC',
                'evidence': self._get_context(text, match.start(), match.end())
            })
        
        # Process stage with colon format
        for match in self.regex_patterns['stage_colon'].finditer(text):
            findings.append({
                'stage': f"Stage {match.group(1)}".title(),
                'system': 'General',
                'evidence': self._get_context(text, match.start(), match.end())
            })
        
        # Process roman numeral stage (only if "stage" or "staging" appears in context)
        for match in self.regex_patterns['roman_numeral_stage'].finditer(text):
            # This pattern only matches if "stage" or "staging" is in context
            findings.append({
                'stage': f"Stage {match.group(1)}".title(),
                'system': 'General',
                'evidence': self._get_context(text, match.start(), match.end())
            })
        
        return findings
    
    def _get_context(self, text: str, start: int, end: int, context_chars: int = 50) -> str:
        """
        Get surrounding context from the text.
        
        Args:
            text: The full text
            start: Start position of the match
            end: End position of the match
            context_chars: Number of characters to include before and after
            
        Returns:
            String with the match and surrounding context
        """
        context_start = max(0, start - context_chars)
        context_end = min(len(text), end + context_chars)
        return text[context_start:context_end]
    
    def process_text(self, text: str) -> Optional[Dict]:
        """
        Process a single text and return the first staging information found.
        
        Args:
            text: Clinical note text
            
        Returns:
            Dictionary with stage and system, or None if no staging found
        """
        findings = self.extract_staging(text)
        if findings:
            # Return the first finding (with highest confidence)
            return {
                'stage': findings[0]['stage'],
                'system': findings[0]['system']
            }
        return None

def process_file(file_path: str, output_path: str = None, extractor = None):
    """
    Process a single parquet file to extract staging information.
    
    Args:
        file_path: Path to the parquet file
        output_path: Path to save the results (optional)
        extractor: Extractor instance (if None, a new one will be created)
        
    Returns:
        DataFrame with only rows containing staging information
    """
    try:
        logger.info(f"Reading file: {file_path}")
        df = pd.read_parquet(file_path)
        
        # Create extractor if not provided
        if extractor is None:
            extractor = RegexStagingExtractor()
        
        logger.info(f"Processing {len(df)} notes...")
        
        # Create empty columns for staging info
        df['stage'] = None
        df['system'] = None
        
        # Track rows with staging info
        has_staging = []
        
        # Process each note
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting staging info"):
            text = row.get('note_text', '')
            if not isinstance(text, str) or not text.strip():
                continue
                
            # Extract staging information
            staging_info = extractor.process_text(text)
            
            if staging_info:
                # Store results in the dataframe
                df.at[idx, 'stage'] = staging_info.get('stage')
                df.at[idx, 'system'] = staging_info.get('system')
                
                # Track this row as having staging info
                has_staging.append(idx)
        
        # Filter to only rows with staging information
        if has_staging:
            result_df = df.loc[has_staging].copy()
            logger.info(f"Found {len(result_df)} notes with staging information")
            
            # Save results if output path is provided
            if output_path:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                
                # Save as parquet
                result_df.to_parquet(output_path, index=False)
                logger.info(f"Saved results to {output_path}")
                
                # Also save CSV for easy inspection
                csv_path = output_path.replace('.parquet', '.csv')
                result_df.to_csv(csv_path, index=False)
                logger.info(f"Also saved as CSV to {csv_path}")
            
            return result_df
        else:
            logger.info("No staging information found in this file")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return pd.DataFrame()

def process_batch(batch_number: int, input_dir: str, output_dir: str):
    """
    Process a specific batch number.
    
    Args:
        batch_number: The batch number to process
        input_dir: Directory containing input files
        output_dir: Directory to save output files
    """
    input_file = os.path.join(input_dir, f"filtered_notes_batch_{batch_number}.parquet")
    output_file = os.path.join(output_dir, f"staging_results_batch_{batch_number}.parquet")
    
    logger.info(f"Processing batch {batch_number}")
    logger.info(f"Input: {input_file}")
    logger.info(f"Output: {output_file}")
    
    # Create extractor
    extractor = RegexStagingExtractor()
    
    # Process file
    process_file(input_file, output_file, extractor)

def benchmark(iterations: int = 1000):
    """
    Benchmark the performance of the regex extractor.
    
    Args:
        iterations: Number of texts to process
    """
    # Create sample texts
    sample_texts = [
        "Patient has stage IIB breast cancer. Recent studies confirm T2N1M0 classification.",
        "Assessment: cT3N2M0 prostate adenocarcinoma, AJCC Stage III.",
        "The patient was diagnosed with Stage IV lung cancer with metastasis to the liver.",
        "Pathology revealed pT1N0M0, consistent with Stage I disease.",
        "This is a follow-up for the patient's Stage IIIB colon cancer.",
        "TNM classification: T1, N0, M0, indicating early-stage disease.",
        "The patient has no evidence of cancer. All tests were negative.",
        "Follow-up required for medication adjustment. No staging information available.",
        "IMPRESSION: 1. Status post right mastectomy for Stage IIA breast cancer."
    ]
    
    # Create extractor
    extractor = RegexStagingExtractor()
    
    # Benchmark extraction
    logger.info(f"Benchmarking with {iterations} iterations...")
    start_time = time.time()
    
    for _ in tqdm(range(iterations)):
        for text in sample_texts:
            extractor.process_text(text)
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / (iterations * len(sample_texts)) * 1000  # ms
    
    logger.info(f"Benchmark results:")
    logger.info(f"Total time: {total_time:.2f} seconds")
    logger.info(f"Average time per text: {avg_time:.2f} ms")
    logger.info(f"Texts processed per second: {(iterations * len(sample_texts)) / total_time:.2f}")

def main():
    """Main function to run the staging extraction pipeline."""
    # Configure argument parser
    parser = argparse.ArgumentParser(description='Extract staging information using regex only')
    parser.add_argument('input', nargs='?', help='Input parquet file')
    parser.add_argument('output', nargs='?', help='Output parquet file')
    parser.add_argument('--batch', type=int, help='Process a specific batch number')
    parser.add_argument('--input-dir', default='/wynton/protected/home/zack/brtan/Stage_2_Staging_Extractor/data/output/filtered_notes/final',
                        help='Directory containing input files')
    parser.add_argument('--output-dir', default='/wynton/protected/home/zack/brtan/Stage_2_Staging_Extractor/data/output/staging_results',
                        help='Directory to save output files')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark')
    args = parser.parse_args()
    
    # Run benchmark if requested
    if args.benchmark:
        benchmark()
        return
    
    # Handle batch mode
    if args.batch is not None:
        process_batch(args.batch, args.input_dir, args.output_dir)
        return
    
    # Handle direct file processing
    if args.input and args.output:
        process_file(args.input, args.output)
        return
    
    # If no specific mode, show help
    parser.print_help()

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    logger.info(f"Script completed in {elapsed_time:.2f} seconds")
