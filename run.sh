#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=0
# Script function: execute text keyword extraction or image region extraction based on parameters
# Usage: /root/run_extractor.sh [txt|img|pimg|vimg|eimg] [-config CONFIG_NAME] [-s]

# Project directory path
PROJECT_DIR="/root/CABA-main"

# Default: do not shut down
SHUTDOWN=false

# Default config file name (without extension)
CONFIG_NAME="dscmr"

# Parse arguments
while [ $# -gt 0 ]; do
    case "$1" in
        -s)
            SHUTDOWN=true
            shift
            ;;
        -config)
            if [ -n "$2" ] && [ "${2#-}" = "$2" ]; then
                CONFIG_NAME="$2"
                shift 2
            else
                echo "Error: -config requires an argument value"
                exit 1
            fi
            ;;
        txt|img|pimg|vimg|eimg)
            EXEC_TYPE="$1"
            shift
            ;;
        *)
            echo "Error: invalid argument '$1'"
            echo "Usage: $0 [txt|img|pimg|vimg|eimg] [-config CONFIG_NAME] [-s]"
            echo "  txt        - Execute text keyword extraction"
            echo "  img        - Execute image region extraction"
            echo "  pimg       - Execute image poisoning attack"
            echo "  vimg       - Execute model validation"
            echo "  eimg       - Execute evaluate target image"
            echo "  -config    - Specify config file name (default: dscmr)"
            echo "  -s         - Optional, automatically shut down after execution"
            exit 1
            ;;
    esac
done

# Check if execution type is provided
if [ -z "$EXEC_TYPE" ]; then
    echo "Error: execution type argument is required"
    echo "Usage: $0 [txt|img|pimg|vimg|eimg] [-config CONFIG_NAME] [-s]"
    echo "  txt        - Execute text keyword extraction"
    echo "  img        - Execute image region extraction"
    echo "  pimg       - Execute image poisoning attack"
    echo "  vimg       - Execute model validation"
    echo "  eimg       - Execute evaluate target image"
    echo "  -config    - Specify config file name (default: dscmr)"
    echo "  -s         - Optional, automatically shut down after execution"
    exit 1
fi

# Check if project directory exists
if [ ! -d "$PROJECT_DIR" ]; then
    echo "Error: project directory does not exist: $PROJECT_DIR"
    exit 1
fi

# Switch to project directory
cd "$PROJECT_DIR" || {
    echo "Error: cannot switch to project directory: $PROJECT_DIR"
    exit 1
}

# EXEC_TYPE already set in argument parsing loop

# Generate timestamp and date
TIMESTAMP=$(date +"%Y_%m_%d_%H_%M_%S")
DATE_DIR=$(date +"%Y-%-m-%-d")

# Run ID
RUN_ID="${EXEC_TYPE}_${TIMESTAMP}"

# Create base log directory
BASE_LOG_DIR="$PROJECT_DIR/log/run_log"
mkdir -p "$BASE_LOG_DIR"

# Execute corresponding command based on argument
case $EXEC_TYPE in
    txt)
        LOG_TYPE_DIR="$BASE_LOG_DIR/text_log/$DATE_DIR"
        mkdir -p "$LOG_TYPE_DIR"
        LOG_FILE="$LOG_TYPE_DIR/txt_${TIMESTAMP}.txt"
        echo "========================================" | tee "$LOG_FILE"
        echo "Executing text keyword extraction..." | tee -a "$LOG_FILE"
        echo "Unique run ID for this run: $RUN_ID" | tee -a "$LOG_FILE"
        echo "Working directory: $(pwd)" | tee -a "$LOG_FILE"
        echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
        echo "Start time: $(date)" | tee -a "$LOG_FILE"
        echo "========================================" | tee -a "$LOG_FILE"
        START_TS=$(date +%s)
        python -m badcm.text_keyword_extractor 2>&1 | tee -a "$LOG_FILE"
        EXIT_CODE=${PIPESTATUS[0]}
        END_TS=$(date +%s)
        ELAPSED=$((END_TS - START_TS))
        ELAPSED_FMT=$(printf '%02d:%02d:%02d' $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60)))
        echo "========================================" | tee -a "$LOG_FILE"
        echo "End time: $(date)" | tee -a "$LOG_FILE"
        echo "Total elapsed: ${ELAPSED}s (${ELAPSED_FMT})" | tee -a "$LOG_FILE"
        echo "Exit code: $EXIT_CODE" | tee -a "$LOG_FILE"

        # Check if shutdown is needed
        if [ "$SHUTDOWN" = true ]; then
            echo "========================================" | tee -a "$LOG_FILE"
            echo "Execution completed, shutting down..." | tee -a "$LOG_FILE"
            shutdown -h now
        fi

        exit $EXIT_CODE
        ;;
    img)
        LOG_TYPE_DIR="$BASE_LOG_DIR/image_log/$DATE_DIR"
        mkdir -p "$LOG_TYPE_DIR"
        LOG_FILE="$LOG_TYPE_DIR/img_${TIMESTAMP}.txt"
        echo "========================================" | tee "$LOG_FILE"
        echo "Executing image region extraction..." | tee -a "$LOG_FILE"
        echo "Unique run ID for this run: $RUN_ID" | tee -a "$LOG_FILE"
        echo "Working directory: $(pwd)" | tee -a "$LOG_FILE"
        echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
        echo "Start time: $(date)" | tee -a "$LOG_FILE"
        echo "========================================" | tee -a "$LOG_FILE"
        START_TS=$(date +%s)
        python -m badcm.image_region_extractor 2>&1 | tee -a "$LOG_FILE"
        EXIT_CODE=${PIPESTATUS[0]}
        END_TS=$(date +%s)
        ELAPSED=$((END_TS - START_TS))
        ELAPSED_FMT=$(printf '%02d:%02d:%02d' $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60)))
        echo "========================================" | tee -a "$LOG_FILE"
        echo "End time: $(date)" | tee -a "$LOG_FILE"
        echo "Total elapsed: ${ELAPSED}s (${ELAPSED_FMT})" | tee -a "$LOG_FILE"
        echo "Exit code: $EXIT_CODE" | tee -a "$LOG_FILE"

        # Check if shutdown is needed
        if [ "$SHUTDOWN" = true ]; then
            echo "========================================" | tee -a "$LOG_FILE"
            echo "Execution completed, shutting down..." | tee -a "$LOG_FILE"
            shutdown -h now
        fi

        exit $EXIT_CODE
        ;;
    pimg)
        LOG_TYPE_DIR="$BASE_LOG_DIR/poison_image_log/$DATE_DIR"
        mkdir -p "$LOG_TYPE_DIR"
        LOG_FILE="$LOG_TYPE_DIR/pimg_${TIMESTAMP}.txt"
        echo "========================================" | tee "$LOG_FILE"
        echo "Executing image poisoning attack..." | tee -a "$LOG_FILE"
        echo "Unique run ID for this run: $RUN_ID" | tee -a "$LOG_FILE"
        echo "Working directory: $(pwd)" | tee -a "$LOG_FILE"
        echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
        echo "Start time: $(date)" | tee -a "$LOG_FILE"
        echo "========================================" | tee -a "$LOG_FILE"
        START_TS=$(date +%s)
        python -m badcm.poison_image_regions 2>&1 | tee -a "$LOG_FILE"
        EXIT_CODE=${PIPESTATUS[0]}
        END_TS=$(date +%s)
        ELAPSED=$((END_TS - START_TS))
        ELAPSED_FMT=$(printf '%02d:%02d:%02d' $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60)))
        echo "========================================" | tee -a "$LOG_FILE"
        echo "End time: $(date)" | tee -a "$LOG_FILE"
        echo "Total elapsed: ${ELAPSED}s (${ELAPSED_FMT})" | tee -a "$LOG_FILE"
        echo "Exit code: $EXIT_CODE" | tee -a "$LOG_FILE"

        # Check if shutdown is needed
        if [ "$SHUTDOWN" = true ]; then
            echo "========================================" | tee -a "$LOG_FILE"
            echo "Execution completed, shutting down..." | tee -a "$LOG_FILE"
            shutdown -h now
        fi

        exit $EXIT_CODE
        ;;
    vimg)
        LOG_TYPE_DIR="$BASE_LOG_DIR/validate_log/$DATE_DIR"
        mkdir -p "$LOG_TYPE_DIR"

        # CONFIG_NAME directly used as module name

        # Read config file to get backbone (first value)
        BACKBONE=$(python3 -c "
import yaml
import sys
config_path = 'config/${CONFIG_NAME}.yaml'
try:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    backbones = config.get('backbones', [])
    print(backbones[0] if backbones else 'Unknown')
except Exception as e:
    print('Unknown')
")

        LOG_FILE="$LOG_TYPE_DIR/vimg_${CONFIG_NAME}_${BACKBONE}_${TIMESTAMP}.txt"
        echo "========================================" | tee "$LOG_FILE"
        echo "Executing model validation..." | tee -a "$LOG_FILE"
        echo "Unique run ID for this run: $RUN_ID" | tee -a "$LOG_FILE"
        echo "Working directory: $(pwd)" | tee -a "$LOG_FILE"
        echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
        echo "Start time: $(date)" | tee -a "$LOG_FILE"
        echo "========================================" | tee -a "$LOG_FILE"
        START_TS=$(date +%s)
        python main.py --config_name "${CONFIG_NAME}.yaml" 2>&1 | tee -a "$LOG_FILE"
        EXIT_CODE=${PIPESTATUS[0]}
        END_TS=$(date +%s)
        ELAPSED=$((END_TS - START_TS))
        ELAPSED_FMT=$(printf '%02d:%02d:%02d' $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60)))
        echo "========================================" | tee -a "$LOG_FILE"
        echo "End time: $(date)" | tee -a "$LOG_FILE"
        echo "Total elapsed: ${ELAPSED}s (${ELAPSED_FMT})" | tee -a "$LOG_FILE"
        echo "Exit code: $EXIT_CODE" | tee -a "$LOG_FILE"

        # Check if shutdown is needed
        if [ "$SHUTDOWN" = true ]; then
            echo "========================================" | tee -a "$LOG_FILE"
            echo "Execution completed, shutting down..." | tee -a "$LOG_FILE"
            shutdown -h now
        fi

        exit $EXIT_CODE
        ;;
    eimg)
        LOG_TYPE_DIR="$BASE_LOG_DIR/evaluate_target_image_log/$DATE_DIR"
        mkdir -p "$LOG_TYPE_DIR"
        LOG_FILE="$LOG_TYPE_DIR/eimg_${TIMESTAMP}.txt"
        echo "========================================" | tee "$LOG_FILE"
        echo "Executing evaluate target image..." | tee -a "$LOG_FILE"
        echo "Unique run ID for this run: $RUN_ID" | tee -a "$LOG_FILE"
        echo "Working directory: $(pwd)" | tee -a "$LOG_FILE"
        echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
        echo "Start time: $(date)" | tee -a "$LOG_FILE"
        echo "========================================" | tee -a "$LOG_FILE"
        START_TS=$(date +%s)
        python -m badcm.evaluate_target 2>&1 | tee -a "$LOG_FILE"
        EXIT_CODE=${PIPESTATUS[0]}
        END_TS=$(date +%s)
        ELAPSED=$((END_TS - START_TS))
        ELAPSED_FMT=$(printf '%02d:%02d:%02d' $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60)))
        echo "========================================" | tee -a "$LOG_FILE"
        echo "End time: $(date)" | tee -a "$LOG_FILE"
        echo "Total elapsed: ${ELAPSED}s (${ELAPSED_FMT})" | tee -a "$LOG_FILE"
        echo "Exit code: $EXIT_CODE" | tee -a "$LOG_FILE"

        # Check if shutdown is needed
        if [ "$SHUTDOWN" = true ]; then
            echo "========================================" | tee -a "$LOG_FILE"
            echo "Execution completed, shutting down..." | tee -a "$LOG_FILE"
            shutdown -h now
        fi

        exit $EXIT_CODE
        ;;
    *)
        echo "Error: invalid argument '$EXEC_TYPE'"
        echo "Usage: $0 [txt|img|pimg|vimg|eimg] [-config CONFIG_NAME] [-s]"
        echo "  txt        - Execute text keyword extraction"
        echo "  img        - Execute image region extraction"
        echo "  pimg       - Execute image poisoning attack"
        echo "  vimg       - Execute model validation"
        echo "  eimg       - Execute evaluate target image"
        echo "  -config    - Specify config file name (default: dscmr)"
        echo "  -s         - Optional, automatically shut down after execution"
        exit 1
        ;;
esac