#!/bin/bash
#$ -cwd
#$ -l h_rt=00:29:50
#$ -l mem_free=32G
#$ -l gpu_mem=30G
#$ -l scratch=200G 
#$ -pe smp 1
#$ -q gpu.q 
#$ -o qsub_logs/filter_notes_${SGE_TASK_ID}.out
#$ -e qsub_logs/filter_notes_${SGE_TASK_ID}.err

# --- Validation Checks ---
module purge || { echo "Module purge failed"; exit 1; }
module load cuda

# --- Environment Setup ---
# export HF_HOME=/wynton/scratch/$USER/huggingface_cache
export XDG_CACHE_HOME=/wynton/scratch/$USER/.cache
mkdir -p $HF_HOME $XDG_CACHE_HOME

# Create logs directory if it doesn't exist
mkdir -p qsub_logs

source /wynton/protected/home/zack/brtan/Virtual_Environments/dask_distribution_env/bin/activate

# --- GPU Configuration ---
export CUDA_VISIBLE_DEVICES=$SGE_GPU
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib

# Process single parquet file based on job array ID
echo "Starting job $JOB_ID on GPU $CUDA_VISIBLE_DEVICES at $(date)"

# Print current working directory for debugging
echo "Current working directory: $(pwd)"

# Run the Python script with full path
python $(pwd)/filter_notes.py --encounters_dir /wynton/protected/home/zack/brtan/Stage_2_Staging_Extractor/data/output/filtered_encounters --note_meta_dir /wynton/protected/project/ic/data/parquet/DEID_CDW/note_metadata --note_text_dir /wynton/protected/project/ic/data/parquet/DEID_CDW/note_text --output_dir /scratch/brtan/filtered_notes 2>&1 | tee -a qsub_logs/notes_$JOB_ID.log

# --- Exit Status Check ---
if [ $? -eq 0 ]; then
    echo "Job completed successfully at $(date)"
else
    echo "Job failed with exit code $? at $(date)"
    exit 1
fi 