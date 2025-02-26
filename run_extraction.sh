#!/bin/bash
# Wrapper script to run the staging extraction with the correct virtual environment
# Uses the local Llama-3.1-8B model at /wynton/protected/home/zack/brtan/models/Llama-3.1-8B

# Display usage information
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Usage: ./run_extraction.sh [OPTION] BATCH_NUMBER"
    echo ""
    echo "Options:"
    echo "  --test         Run the parsing logic tests"
    echo "  --benchmark    Run the parsing performance benchmark"
    echo "  --help, -h     Display this help message"
    echo ""
    echo "Example:"
    echo "  ./run_extraction.sh 5         # Process batch 5"
    echo "  ./run_extraction.sh --test    # Run tests"
    exit 0
fi

# Check if CUDA is available
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "Warning: CUDA_VISIBLE_DEVICES is not set. Using default GPU configuration."
    # Uncomment the line below to set specific GPUs if needed
    # export CUDA_VISIBLE_DEVICES=0,1
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source /wynton/protected/home/zack/brtan/Virtual_Environments/dask_distribution_env/bin/activate

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment"
    echo "Path: /wynton/protected/home/zack/brtan/Virtual_Environments/dask_distribution_env/bin/activate"
    exit 1
fi

# Verify model path exists
MODEL_PATH="/wynton/protected/home/zack/brtan/models/Llama-3.1-8B"
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path not found: $MODEL_PATH"
    deactivate
    exit 1
fi
echo "Using local model at: $MODEL_PATH"

# Run the extraction script with all arguments passed to this script
echo "Running extraction script with arguments: $@"
python new_extract_staging.py "$@"

# Capture the exit code
EXIT_CODE=$?

# Deactivate the virtual environment
deactivate

# Exit with the same code as the Python script
exit $EXIT_CODE 