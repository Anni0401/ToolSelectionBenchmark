#!/bin/bash

# Configuration parameters
DEST_MACHINE=()  # Can be empty or contain machine names: ("deeplearn2" "faretra" "moro2332")
SCRIPT_TYPE="" # "mnrl", "ports", or "replug"
GPU_TYPE=("nvidia_geforce_rtx_3090")
GPU_COUNT=(1)

# Default hyperparameters as arrays (can be overridden by command line)
LR=(2e-5)
BATCH_SIZE=(2)
EPOCHS=(1)
DATASET=(toolbench)
RETRIEVAL_MODEL=("BAAI/bge-base-en-v1.5")  # Default retrieval/encoder model
INFERENCE_MODEL=("llama3-8B")  # Default LLM/inference model
WANDB_RUN_NAME=()  # Empty by default, will be auto-generated if not provided
WANDB_PROJECT_NAME=("PORTS_EMNLP_Dashboard")  # Default project name
ADDITIONAL_PARAMS=()
SAVE_CHECKPOINTS=false  # Add save_checkpoints with default to false
INFERENCE_MAX_SEQ_LEN=(1024)  # Default inference max seq length

# New parameters
BETA=(1.0)  # Default beta value
GAMMA=(1.0)  # Default gamma value
LAMBDA_LOSS=(0.2)  # Default lambda loss value
PREFERENCE_WEIGHT=(0.1)  # Default preference weight
WEIGHT_DECAY=(0.01)  # Default weight decay
N_REEMBEDDING_STEPS=(50)  # Default n_reembedding_steps

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --machine=*) 
      IFS=',' read -ra DEST_MACHINE <<< "${1#*=}"
      ;;
    --script=*) 
      SCRIPT_TYPE="${1#*=}" 
      ;;
    --gpu_type=*) 
      IFS=',' read -ra GPU_TYPE <<< "${1#*=}"
      ;;
    --gpu_count=*) 
      IFS=',' read -ra GPU_COUNT <<< "${1#*=}"
      ;;
    --lr=*) 
      IFS=',' read -ra LR <<< "${1#*=}"
      ;;
    --batch_size=*) 
      IFS=',' read -ra BATCH_SIZE <<< "${1#*=}"
      ;;
    --epochs=*) 
      IFS=',' read -ra EPOCHS <<< "${1#*=}"
      ;;
    --dataset=*) 
      IFS=',' read -ra DATASET <<< "${1#*=}"
      ;;
    --retrieval_model=*) 
      IFS=',' read -ra RETRIEVAL_MODEL <<< "${1#*=}"
      ;;
    --inference_model=*) 
      IFS=',' read -ra INFERENCE_MODEL <<< "${1#*=}"
      ;;
    --wandb_run_name=*) 
      IFS=',' read -ra WANDB_RUN_NAME <<< "${1#*=}"
      ;;
    --wandb_project_name=*) 
      IFS=',' read -ra WANDB_PROJECT_NAME <<< "${1#*=}"
      ;;
    --save_checkpoints=*) 
      SAVE_CHECKPOINTS="${1#*=}"
      ;;
    --params=*) 
      # For additional params, we just keep the entire string
      ADDITIONAL_PARAMS=("${1#*=}")
      ;;
    --inference_max_seq_len=*)
      IFS=',' read -ra INFERENCE_MAX_SEQ_LEN <<< "${1#*=}"
      ;;
    # New parameter options
    --beta=*)
      IFS=',' read -ra BETA <<< "${1#*=}"
      ;;
    --gamma=*)
      IFS=',' read -ra GAMMA <<< "${1#*=}"
      ;;
    --lambda_loss=*)
      IFS=',' read -ra LAMBDA_LOSS <<< "${1#*=}"
      ;;
    --preference_weight=*)
      IFS=',' read -ra PREFERENCE_WEIGHT <<< "${1#*=}"
      ;;
    --weight_decay=*)
      IFS=',' read -ra WEIGHT_DECAY <<< "${1#*=}"
      ;;
    --embedding_update_steps=*)
      # For backward compatibility, use this as n_reembedding_steps
      IFS=',' read -ra N_REEMBEDDING_STEPS <<< "${1#*=}"
      ;;
    --n_reembedding_steps=*)
      IFS=',' read -ra N_REEMBEDDING_STEPS <<< "${1#*=}"
      ;;
    *) 
      echo "Unknown parameter: $1"
      exit 1 
      ;;
  esac
  shift
done

# Check if required parameters are provided
if [ -z "$SCRIPT_TYPE" ]; then
  echo "Error: Script type is required. Use --script=mnrl|ports|replug"
  exit 1
fi

# Choose the appropriate script based on type
case $SCRIPT_TYPE in
  "mnrl") SCRIPT="scripts/train_mnrl_encoder.sh" ;;
  "ports") SCRIPT="scripts/train_ports_encoder.sh" ;;
  "replug") SCRIPT="scripts/train_replug_encoder.sh" ;;
  *) echo "Error: Invalid script type. Use --script=mnrl|ports|replug"; exit 1 ;;
esac

# Set default values for empty arrays
[[ ${#LR[@]} -eq 0 ]] && LR=("")
[[ ${#BATCH_SIZE[@]} -eq 0 ]] && BATCH_SIZE=("")
[[ ${#EPOCHS[@]} -eq 0 ]] && EPOCHS=("")
[[ ${#DATASET[@]} -eq 0 ]] && DATASET=("")
[[ ${#RETRIEVAL_MODEL[@]} -eq 0 ]] && RETRIEVAL_MODEL=("BAAI/bge-base-en-v1.5")
[[ ${#INFERENCE_MODEL[@]} -eq 0 ]] && INFERENCE_MODEL=("llama3-8B")
[[ ${#WANDB_RUN_NAME[@]} -eq 0 ]] && WANDB_RUN_NAME=("")
[[ ${#WANDB_PROJECT_NAME[@]} -eq 0 ]] && WANDB_PROJECT_NAME=("PORTS_AAAI-EMNLP")
[[ ${#ADDITIONAL_PARAMS[@]} -eq 0 ]] && ADDITIONAL_PARAMS=("")
[[ ${#INFERENCE_MAX_SEQ_LEN[@]} -eq 0 ]] && INFERENCE_MAX_SEQ_LEN=(1024)
# Default values for new parameters if empty
[[ ${#BETA[@]} -eq 0 ]] && BETA=(1.0)
[[ ${#GAMMA[@]} -eq 0 ]] && GAMMA=(1.0)
[[ ${#LAMBDA_LOSS[@]} -eq 0 ]] && LAMBDA_LOSS=(0.2)
[[ ${#PREFERENCE_WEIGHT[@]} -eq 0 ]] && PREFERENCE_WEIGHT=(0.1)
[[ ${#WEIGHT_DECAY[@]} -eq 0 ]] && WEIGHT_DECAY=(0.01)
[[ ${#N_REEMBEDDING_STEPS[@]} -eq 0 ]] && N_REEMBEDDING_STEPS=(50)

# If no machines specified, set a single empty value to iterate once without machine specification
[[ ${#DEST_MACHINE[@]} -eq 0 ]] && DEST_MACHINE=("")

# Job counter
job_count=0

# Iterate over all parameter combinations
for gpu_type in "${GPU_TYPE[@]}"; do
  for gpu_count in "${GPU_COUNT[@]}"; do
    for lr in "${LR[@]}"; do
      for bs in "${BATCH_SIZE[@]}"; do
        for ep in "${EPOCHS[@]}"; do
          for ds in "${DATASET[@]}"; do
            for rm in "${RETRIEVAL_MODEL[@]}"; do
              for im in "${INFERENCE_MODEL[@]}"; do
                for max_seq_len in "${INFERENCE_MAX_SEQ_LEN[@]}"; do
                  for beta in "${BETA[@]}"; do
                    for gamma in "${GAMMA[@]}"; do
                      for lambda_loss in "${LAMBDA_LOSS[@]}"; do
                        for pref_weight in "${PREFERENCE_WEIGHT[@]}"; do
                          for weight_decay in "${WEIGHT_DECAY[@]}"; do
                            for n_reembedding_steps in "${N_REEMBEDDING_STEPS[@]}"; do
                              for wr in "${WANDB_RUN_NAME[@]}"; do
                                for wp in "${WANDB_PROJECT_NAME[@]}"; do
                                  for add_params in "${ADDITIONAL_PARAMS[@]}"; do
                                    # Construct parameter string for this iteration
                                    PARAMS=""
                                    [[ -n "$lr" ]] && PARAMS="$PARAMS --lr=$lr"
                                    [[ -n "$bs" ]] && PARAMS="$PARAMS --batch_size=$bs"
                                    [[ -n "$ep" ]] && PARAMS="$PARAMS --epochs=$ep"
                                    [[ -n "$ds" ]] && PARAMS="$PARAMS --dataset=$ds"
                                    [[ -n "$rm" ]] && PARAMS="$PARAMS --retrieval_model=$rm"
                                    [[ -n "$im" ]] && PARAMS="$PARAMS --inference_model=$im"
                                    [[ -n "$max_seq_len" ]] && PARAMS="$PARAMS --inference_max_seq_len=$max_seq_len"
                                    
                                    # Add parameters with appropriate names based on script type
                                    [[ -n "$beta" ]] && PARAMS="$PARAMS --beta=$beta"
                                    [[ -n "$gamma" ]] && PARAMS="$PARAMS --gamma=$gamma"
                                    
                                    # Handle parameters that have different names in different scripts
                                    if [[ "$SCRIPT_TYPE" == "ports" ]]; then
                                        [[ -n "$lambda_loss" ]] && PARAMS="$PARAMS --lambda_weight=$lambda_loss"
                                        [[ -n "$pref_weight" ]] && PARAMS="$PARAMS --preference_weight=$pref_weight"
                                    else
                                        # For replug and mnrl, still pass them for compatibility
                                        [[ -n "$lambda_loss" ]] && PARAMS="$PARAMS --lambda_loss=$lambda_loss"
                                        [[ -n "$pref_weight" ]] && PARAMS="$PARAMS --preference_weight=$pref_weight"
                                    fi
                                    
                                    # Common parameters for all script types
                                    [[ -n "$weight_decay" ]] && PARAMS="$PARAMS --weight_decay=$weight_decay"
                                    
                                    # Add reembedding steps parameter
                                    [[ -n "$n_reembedding_steps" ]] && PARAMS="$PARAMS --n_reembedding_steps=$n_reembedding_steps"
                                    
                                    # Add save_checkpoints if true
                                    [[ "$SAVE_CHECKPOINTS" == "true" ]] && PARAMS="$PARAMS --save_checkpoints=true"
                                    
                                    # Simplified WANDB run name: <method>_<dataset>
                                    # If custom wandb_run_name was provided, use it instead
                                    if [[ -n "$wr" ]]; then
                                      PARAMS="$PARAMS --wandb_run_name=$wr"
                                    else
                                      # Auto-generate name based on method and dataset
                                      method_name=$(echo "$SCRIPT_TYPE" | tr '[:lower:]' '[:upper:]')
                                      PARAMS="$PARAMS --wandb_run_name=${method_name}_${ds}"
                                    fi
                                    
                                    # Add W&B project name
                                    [[ -n "$wp" ]] && PARAMS="$PARAMS --wandb_project_name=$wp"
                                    
                                    [[ -n "$add_params" ]] && PARAMS="$PARAMS $add_params"
                                    
                                    # Create a unique job name with model info and all parameters
                                    job_id="${SCRIPT_TYPE}_${ds}_${im##*/}_${rm##*/}_lr${lr}_bs${bs}_ep${ep}"
                                    
                                    # Add new parameters to job name for better tracking
                                    if [[ "$SCRIPT_TYPE" == "ports" || "$SCRIPT_TYPE" == "replug" ]]; then
                                      job_id="${job_id}_b${beta}_g${gamma}"
                                    fi
                                    
                                    if [[ "$SCRIPT_TYPE" == "ports" ]]; then
                                      job_id="${job_id}_lam${lambda_loss}_pref${pref_weight}"
                                      # Add reembedding steps to job name for ports
                                      job_id="${job_id}_nrs${n_reembedding_steps}"
                                    fi
                                    
                                    # Add weight decay to all job types
                                    job_id="${job_id}_wd${weight_decay}"
                                    
                                    # Sanitize for filename safety
                                    job_id=$(echo "$job_id" | tr '/' '-' | tr '.' '_')
                                    
                                    echo "Submitting job: $job_id"
                                    echo "Parameters: $PARAMS"
                                    
                                    #  Output file name (config params separated by underscores)
                                    OUTPUT_DIR="output/sbatch_output"
                                    mkdir -p "$OUTPUT_DIR"
                                    OUTPUT_FILE="$OUTPUT_DIR/${job_id}.out"
    
                                    # Prepare sbatch command
                                    sbatch_cmd="sbatch -N 1 --gpus=$gpu_type:$gpu_count --output=$OUTPUT_FILE --error=$OUTPUT_FILE"
                                    
                                    # Only add machine parameter if specified
                                    machine=$DEST_MACHINE
                                    if [[ -n "$machine" ]]; then
                                      sbatch_cmd="$sbatch_cmd --nodelist faretra,moro232"
                                      echo "Running on: $machine with $gpu_count $gpu_type GPU(s)"
                                    else
                                      echo "Running on: any available machine with $gpu_count $gpu_type GPU(s)"
                                    fi
                                    
                                    # Add job name and script
                                    sbatch_cmd="$sbatch_cmd -J \"$job_id\" --wrap=\"$SCRIPT $PARAMS\""

                                    echo "***************************"
                                    echo $sbatch_cmd
                                    echo "***************************"
                                    
                                    # Submit job with specified parameters
                                    eval $sbatch_cmd
                                    
                                    # Increment job counter
                                    ((job_count++))
                                    
                                    # Add a small delay between submissions to prevent overwhelming the scheduler
                                    sleep 0.5
                                  done
                                done
                              done
                            done
                          done
                        done
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done

echo "Total jobs submitted: $job_count"