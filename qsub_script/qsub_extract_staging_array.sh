#!/bin/bash
#$ -cwd
#$ -l h_rt=00:15:00
#$ -l mem_free=32G
#$ -l gpu_mem=30G
#$ -l scratch=10G 
#$ -pe smp 1
#$ -q gpu.q 
#$ -t 1-2 # Edit to do all 100 files. For now, test on 2 files!
#$ -o qsub_logs/extract_staging_${SGE_TASK_ID}.out
#$ -e qsub_logs/extract_staging_${SGE_TASK_ID}.err

# --- Validation Checks ---
module purge || { echo "Module purge failed"; exit 1; }
module load cuda

# --- Environment Setup ---
export XDG_CACHE_HOME=/wynton/scratch/$USER/.cache
mkdir -p $XDG_CACHE_HOME

# Create logs directory if it doesn't exist
mkdir -p qsub_logs

source /wynton/protected/home/zack/brtan/Virtual_Environments/dask_distribution_env/bin/activate

# --- GPU Configuration ---
export CUDA_VISIBLE_DEVICES=$SGE_GPU
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib

# Set input and output directories
INPUT_DIR="/wynton/protected/home/zack/brtan/Stage_2_Staging_Extractor/data/output/filtered_notes/final"
OUTPUT_DIR="/wynton/protected/home/zack/brtan/Stage_2_Staging_Extractor/data/output/staging_results"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Construct file paths based on task ID
INPUT_FILE="${INPUT_DIR}/filtered_notes_batch_${SGE_TASK_ID}.parquet"
OUTPUT_FILE="${OUTPUT_DIR}/staging_results_batch_${SGE_TASK_ID}.parquet"

# Print job information for debugging
echo "Starting job $JOB_ID, task $SGE_TASK_ID on GPU $CUDA_VISIBLE_DEVICES at $(date)"
echo "Current working directory: $(pwd)"
echo "Processing file: $INPUT_FILE"
echo "Output will be saved to: $OUTPUT_FILE"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file $INPUT_FILE does not exist"
    exit 1
fi

# Create a Python script to call the extract_staging function
cat > run_extract_staging.py << 'EOF'
import sys
import resource
import time
from extract_staging import extract_staging

def get_max_memory_mb():
    """Get the maximum memory usage in MB"""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    use_llm = sys.argv[3].lower() == "true"
    
    start_time = time.time()
    start_mem = get_max_memory_mb()
    print(f"Processing {input_file} with LLM={use_llm}")
    print(f"Initial memory usage: {start_mem:.2f} MB")
    
    extract_staging(input_file, output_file, use_llm)
    
    end_time = time.time()
    max_mem = get_max_memory_mb()
    print(f"Results saved to {output_file}")
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    print(f"Maximum memory usage: {max_mem:.2f} MB")
    print(f"Memory increase: {max_mem - start_mem:.2f} MB")
EOF

# Run the Python script with time command to track system resources
/usr/bin/time -v python $(pwd)/run_extract_staging.py "$INPUT_FILE" "$OUTPUT_FILE" "true" 2>&1 | tee -a qsub_logs/extract_staging_${SGE_TASK_ID}.log

# --- Exit Status Check ---
if [ $? -eq 0 ]; then
    echo "Job completed successfully at $(date)"
else
    echo "Job failed with exit code $? at $(date)"
    exit 1
fi 