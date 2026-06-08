#!/bin/bash

# Script to train 10 DQN agents and 10 DQN_RND agents iteratively
# on both sanitized and non-sanitized versions of the SQLiMicrobenchmark

# Configuration
BASE_URL="http://localhost:8000"
LOG_FILE="./SQLiMicrobenchmark/mysql/general.log"
NUM_AGENTS=20
VERBOSE=2
LOSS_CRITERIA=200
WIN_CRITERIA=14
EPISODES=200
MAX_TIMESTAMP=30

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to check if required services are running
check_services() {
    print_status "Checking if services are running..."

    # Check if docker container is running
    if [ -d "SQLiMicrobenchmark" ]; then
        cd SQLiMicrobenchmark
        docker compose up -d 2>/dev/null || docker-compose up -d 2>/dev/null
        sleep 5
        cd ..
    else
        print_warning "SQLiMicrobenchmark directory not found. Please ensure the container is running manually."
    fi


    curl -X GET "http://localhost:8000/functions_external/sqli1.php?id=test&difficulty=None" > /dev/null 2>&1


    # Check if log file exists
    if [ ! -f "$LOG_FILE" ]; then
        print_error "Log file not found: $LOG_FILE"
        print_status "Please ensure the SQLiMicrobenchmark is running and the log file exists"
        exit 1
    fi

    # Check if log file is writable
    if [ ! -w "$LOG_FILE" ]; then
        print_warning "Log file is not writable. Attempting to fix permissions..."
        sudo chmod 777 "$LOG_FILE" 2>/dev/null || {
            print_error "Cannot fix permissions for log file. Please run: chmod +rw $LOG_FILE"
            exit 1
        }
    fi

    print_success "Services check complete"
}

# Function to train a single agent
train_agent() {
    local agent_type=$1
    local agent_num=$2
    local url=$3
    local agent_name=$4

    print_status "Training $agent_name #$agent_num (Agent Type: $agent_type) on $url"

    python3 sqirl.py \
        -u "$url" \
        -v $VERBOSE \
        --log_file "$LOG_FILE" \
        --loss_criteria $LOSS_CRITERIA \
        --win_criteria $WIN_CRITERIA \
        --agent $agent_type \
        --learning True \
        --episodes $EPISODES \
        --max_timestamp $MAX_TIMESTAMP

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        print_success "Completed training $agent_name #$agent_num"
    else
        print_error "Failed training $agent_name #$agent_num (exit code: $exit_code)"
        return $exit_code
    fi
}

# Main training function
main() {
    print_status "Starting training script for $NUM_AGENTS agents of each type"
    print_status "Training on both sanitized and non-sanitized versions"
    echo ""

    # Check services before starting
    check_services

    print_status "Results will be saved in: $RESULTS_DIR"
    echo ""

    # Training counters
    TOTAL_TRAININGS=$((NUM_AGENTS * 2))  # 20 DQN_RND sanitized + 20 DQN_RND non-sanitized
    CURRENT_TRAINING=0

    mkdir stats_logs

    # Train 20 DQN_RND agents on sanitized version (training.php)
    print_status "=========================================="
    print_status "Phase 1: Training DQN_RND agents (Agent Type 2) on SANITIZED version"
    print_status "=========================================="
    for i in $(seq 1 $NUM_AGENTS); do
        CURRENT_TRAINING=$((CURRENT_TRAINING + 1))
        print_status "[$CURRENT_TRAINING/$TOTAL_TRAININGS] Starting DQN_RND agent #$i (sanitized)"
        train_agent 2 $i "${BASE_URL}/training.php" "DQN_RND (sanitized)"
        echo ""
        sleep 2  # Small delay between trainings
    done
    mkdir -p ./stats_logs_sanitized/
    mv ./stats_logs/* ./stats_logs_sanitized/
    mkdir ./stats_logs
    # Train 20 DQN_RND agents on non-sanitized version (training_no_san.php)
    print_status "=========================================="
    print_status "Phase 2: Training DQN_RND agents (Agent Type 2) on NON-SANITIZED version"
    print_status "=========================================="
    for i in $(seq 1 $NUM_AGENTS); do
        CURRENT_TRAINING=$((CURRENT_TRAINING + 1))
        print_status "[$CURRENT_TRAINING/$TOTAL_TRAININGS] Starting DQN_RND agent #$i (non-sanitized)"
        train_agent 2 $i "${BASE_URL}/training_no_san.php" "DQN_RND (non-sanitized)"
        echo ""
        sleep 2  # Small delay between trainings
    done
    mv ./stats_logs/* ./stats_logs_non_san
    print_status "=========================================="
    print_success "All training completed!"
    print_status "=========================================="
    print_status "Total agents trained: $TOTAL_TRAININGS"
    print_status "- $NUM_AGENTS DQN agents on sanitized version"
    print_status "- $NUM_AGENTS DQN agents on non-sanitized version"
    print_status "- $NUM_AGENTS DQN_RND agents on sanitized version"
    print_status "- $NUM_AGENTS DQN_RND agents on non-sanitized version"
    print_status ""
    print_status "Training logs and model checkpoints are saved in:"
    print_status "  - stats_logs/ directory"
    print_status ""
    print_status "To view the latest training results, check:"
    print_status "  ls -lt stats_logs/ | head -20"
    print_status ""
    print_status "Each training run creates a directory with timestamp containing:"
    print_status "  - Model checkpoints (DQN_Agent_Checkpoint/ or DQN_RND_Agent_Checkpoint/)"
    print_status "  - Training statistics (result_stats_*.stats)"
    print_status "  - Action logs (actions_Applied_*.stats)"
    print_status "  - SQL statements (sql_statment_*.stats)"
    print_status "  - Rewards (rewards_*.stats)"
}

# Run main function
main "$@"

