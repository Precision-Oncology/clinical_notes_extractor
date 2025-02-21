import dask.dataframe as dd
from pathlib import Path

def filter_notes(encounters_dir: str, note_meta_dir: str, 
                note_text_dir: str, output_dir: str):
    """Join notes with filtered encounters using Dask"""
    # Load filtered encounters
    encounters = dd.read_parquet(
        Path(encounters_dir)/"*.parquet",
        columns=['patientdurablekey', 'encounterkey']
    )
    
    # Load note metadata and text
    note_meta = dd.read_parquet(note_meta_dir)
    note_text = dd.read_parquet(note_text_dir)
    
    # Merge and process in partitions
    (note_meta.merge(note_text, on='deid_note_key')
              .merge(encounters, on=['patientdurablekey', 'encounterkey'])
              .repartition(partition_size="100MB")
              .to_parquet(output_dir)
    )
    print(f"Filtered notes saved to {output_dir}") 