#!/bin/bash
# Process synthetic tools in batches of 10
# Adjust BATCH_SIZE, START_OFFSET, and MAX_OFFSET as needed

BATCH_SIZE=10
START_OFFSET=${1:-0}      # Start from this offset (default 0, or pass as argument)
MAX_OFFSET=460            # 462 tools needing synthetic / 10 per batch = offset up to 460

echo "Processing tools in batches of $BATCH_SIZE..."
echo "Starting from offset: $START_OFFSET"
echo "Max offset: $MAX_OFFSET"
echo "Total batches remaining: $(((MAX_OFFSET - START_OFFSET) / BATCH_SIZE + 1))"

for offset in $(seq $START_OFFSET $BATCH_SIZE $MAX_OFFSET); do
  echo ""
  echo "========================================"
  echo "Batch: offset=$offset, max=$BATCH_SIZE"
  echo "========================================"
  /usr/bin/python3 generate_synthetic_tools.py \
    --max-tools-to-process $BATCH_SIZE \
    --offset $offset \
    --model llama-3.3-70b-versatile
  
  if [ $? -ne 0 ]; then
    echo "ERROR: Batch failed at offset=$offset"
    exit 1
  fi
  
  # Rate limit sleep (adjust as needed for your Groq limits)
  echo "Sleeping 20 seconds before next batch..."
  sleep 20
done

echo ""
echo "All batches completed successfully!"
echo "Check analysis_embeddings/tools_en_synthetic_candidates.jsonl"

echo ""
echo "All batches completed!"
echo "Check analysis_embeddings/tools_en_synthetic_candidates.jsonl"
