#!/bin/bash
# Script to create necessary directories for staging extraction jobs

# Create logs directory
mkdir -p qsub_logs

# Create output directory in scratch
mkdir -p /wynton/protected/home/zack/brtan/Stage_2_Staging_Extractor/data/output/staging_results

echo "Created required directories for staging extraction jobs."
echo "Ready to submit: qsub qsub_script/qsub_extract_staging_array.sh" 