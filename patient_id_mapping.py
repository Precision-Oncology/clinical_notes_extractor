def map_person_source_value_to_patientdurablekey(patient_ids_path: str, input_dir: str, output_dir: str):
    """
    Maps person_source_value to patientdurablekey.
    
    Args:
        patient_ids_path: Path to the CSV file containing patient IDs
        input_dir: Directory containing patdurabledim parquet files
        output_dir: Directory to save the output file
    """
    import pandas as pd
    import os
    import glob
    
    # 1. Read the patient_ids.csv
    print(f"Reading patient IDs from {patient_ids_path}")
    patient_ids_df = pd.read_csv(patient_ids_path)
    
    # Get the list of person_source_values to match
    person_source_values = set(patient_ids_df['person_source_value'].tolist())
    print(f"Looking for {len(person_source_values)} unique person_source_values")
    
    # 2. Read the patdurabledim parquet files
    print(f"Reading parquet files from {input_dir}")
    parquet_files = glob.glob(os.path.join(input_dir, "*.snappy.parquet"))
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")
    
    # Create an empty DataFrame to store the mapping results
    result_df = patient_ids_df.copy()
    result_df['patientdurablekey'] = None
    
    # Track matches found
    matches_found = 0
    
    # Process each parquet file
    for parquet_file in parquet_files:
        print(f"Processing {parquet_file}")
        df = pd.read_parquet(parquet_file)
        
        # Keep only the columns we need and filter for matching IDs
        if 'patientepicid' in df.columns and 'patientdurablekey' in df.columns:
            # Filter the parquet data to only include rows with matching person_source_values
            filtered_df = df[df['patientepicid'].isin(person_source_values)][['patientepicid', 'patientdurablekey']]
            
            # If we found matches in this file
            if not filtered_df.empty:
                # Update the result dataframe with matches from this file
                for _, row in filtered_df.iterrows():
                    mask = result_df['person_source_value'] == row['patientepicid']
                    result_df.loc[mask, 'patientdurablekey'] = row['patientdurablekey']
                    matches_found += mask.sum()
                
                # Remove matched IDs from our search set to speed up future iterations
                matched_ids = set(filtered_df['patientepicid'].tolist())
                person_source_values -= matched_ids
                
                print(f"Found {len(matched_ids)} matches in this file. {len(person_source_values)} IDs remaining to match.")
                
                # If we've found all matches, we can stop processing files
                if not person_source_values:
                    print("All person_source_values matched. Stopping file processing.")
                    break
        else:
            print(f"Warning: Required columns not found in {parquet_file}")
    
    # Report on match results
    total_ids = len(patient_ids_df)
    match_percentage = (matches_found / total_ids) * 100 if total_ids > 0 else 0
    print(f"Matched {matches_found} out of {total_ids} IDs ({match_percentage:.2f}%)")
    
    # Print row counts from original file
    print(f"\nOriginal file: {len(patient_ids_df)} rows")
    
    # 4. Save the result
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    print(f"Saving results to {output_dir}")
    result_df.to_csv(output_dir, index=False)
    
    # Print row counts from result file
    print(f"Result file: {len(result_df)} rows")
    print("Mapping complete!")
    return output_dir


if __name__ == "__main__":
    # Hardcoded paths instead of using command line arguments
    patient_ids_path = "data/input/person_id_mapping.csv"
    input_dir = "/wynton/protected/project/ic/data/parquet/DEID_CDW/patdurabledim"
    output_dir = "data/input/patient_ids_with_durablekey.csv"
    
    # Run the mapping function with hardcoded paths
    map_person_source_value_to_patientdurablekey(
        patient_ids_path=patient_ids_path,
        input_dir=input_dir,
        output_dir=output_dir
    )
