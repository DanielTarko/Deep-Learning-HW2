#!/bin/bash

# SLURM parameters
NUM_CORES=2
NUM_GPUS=1
OUTPUT_DIR="slurm"
LOG_FILE="experiment_slots.log"
MAX_CONCURRENT_JOBS=5
USERNAME="hillahhassan"

# Experiment configurations
EXP1_NAME="exp1_1"
EXP1_K_VALUES=(32 64)
EXP1_L_VALUES=(2 4 8 16)

EXP2_NAME="exp1_2"
EXP2_L_VALUES=(2 4 8)
EXP2_K_VALUES=(32 64 128)

EXP3_NAME="exp1_3"
EXP3_K_VALUES=("64-128")
EXP3_L_VALUES=(2 3 4)

EXP4_NAME="exp1_4"
EXP4_K1_VALUES=("32")            # Single-element array
EXP4_L1_VALUES=(8 16 32)


EXP4_K2_VALUES=("64-128-256")    # Three-element array
EXP4_L2_VALUES=(2 4 8)

# Ensure the output and log directories exist
mkdir -p $OUTPUT_DIR
: > $LOG_FILE  # Clear the log file

# Function to check running jobs and log slots
check_running_jobs() {
    local slot_counter=0
    local running_jobs=$(squeue -u $USERNAME -o "%j" | grep -E "exp1" | wc -l)
    echo "Currently running jobs: $running_jobs"
    while [ "$running_jobs" -ge "$MAX_CONCURRENT_JOBS" ]; do
        echo "Waiting for available slots... already the $slot_counter time"
        sleep 10
        running_jobs=$(squeue -u $USERNAME -o "%j" | grep -E "exp1" | wc -l)
        slot_counter=$((slot_counter + 1))
    done

}

# Submit jobs and log experiment assignment
submit_job() {
    local exp_name="$1"
    local run_name="$2"
    local output_file="$3"
    local job_name="$4"
    local command="$5"

    # Wait for available slots if needed
    check_running_jobs

    # Submit the job
    sbatch -c $NUM_CORES \
           --gres=gpu:$NUM_GPUS \
           -o $output_file \
           --job-name $job_name \
           --wrap "$command"

    echo "Slot assigned: Experiment: $exp_name | Run: $run_name" >> $LOG_FILE
    echo "Submitted job: $run_name with output to $output_file"
}

## Submit jobs for Experiment 1
#for K in "${EXP1_K_VALUES[@]}"; do
#    for L in "${EXP1_L_VALUES[@]}"; do
#        RUN_NAME="${EXP1_NAME}"
#        OUTPUT_FILE="${OUTPUT_DIR}/slurm-${RUN_NAME}.out"
#        JOB_NAME="${RUN_NAME}"
#        COMMAND="python -m hw2.experiments run-exp -n $RUN_NAME -K $K -L $L -P 4 -H 100"
#
#        submit_job "$EXP1_NAME" "$RUN_NAME" "$OUTPUT_FILE" "$JOB_NAME" "$COMMAND"
#    done
#done
#
## Submit jobs for Experiment 2
#for L in "${EXP2_L_VALUES[@]}"; do
#    for K in "${EXP2_K_VALUES[@]}"; do
#        RUN_NAME="${EXP2_NAME}"
#        OUTPUT_FILE="${OUTPUT_DIR}/slurm-${RUN_NAME}.out"
#        JOB_NAME="${RUN_NAME}"
#        COMMAND="python -m hw2.experiments run-exp -n $RUN_NAME -K $K -L $L -P 4 -H 100"
#
#        submit_job "$EXP2_NAME" "$RUN_NAME" "$OUTPUT_FILE" "$JOB_NAME" "$COMMAND"
#    done
#done

## Submit jobs for Experiment 3
for L in "${EXP3_L_VALUES[@]}"; do
    RUN_NAME="${EXP3_NAME}"
    OUTPUT_FILE="${OUTPUT_DIR}/slurm-${RUN_NAME}.out"
    JOB_NAME="${RUN_NAME}"
    COMMAND="python -m hw2.experiments run-exp -n $RUN_NAME -K 64 128 -L $L -P 4 -H 100"

    submit_job "$EXP3_NAME" "$RUN_NAME" "$OUTPUT_FILE" "$JOB_NAME" "$COMMAND"
done


## Submit jobs for Experiment 4 - Configuration set 1
#for L in "${EXP4_L1_VALUES[@]}"; do
#    RUN_NAME="${EXP4_NAME}"
#    OUTPUT_FILE="${OUTPUT_DIR}/slurm-${RUN_NAME}.out"
#    JOB_NAME="${RUN_NAME}"
#    COMMAND="python -m hw2.experiments run-exp -n $RUN_NAME -K 32 -L $L -P 8 -H 100 --model-type resnet"
#
#    submit_job "$EXP4_NAME" "$RUN_NAME" "$OUTPUT_FILE" "$JOB_NAME" "$COMMAND"
#done
#
## Submit jobs for Experiment 4 - Configuration set 2
#for L in "${EXP4_L2_VALUES[@]}"; do
#    RUN_NAME="${EXP4_NAME}"
#    OUTPUT_FILE="${OUTPUT_DIR}/slurm-${RUN_NAME}.out"
#    JOB_NAME="${RUN_NAME}"
#    COMMAND="python -m hw2.experiments run-exp -n $RUN_NAME -K 64 128 256 -L $L -P 8 -H 100 --model-type resnet"
#
#    submit_job "$EXP4_NAME" "$RUN_NAME" "$OUTPUT_FILE" "$JOB_NAME" "$COMMAND"
#done