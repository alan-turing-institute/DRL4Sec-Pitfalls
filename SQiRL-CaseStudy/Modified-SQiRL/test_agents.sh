#!/bin/bash

# Script to test all trained agents on the no_feedback.php endpoint
# Tests agents from both stats_logs_non_san and stats_logs_sanitized directories

# Configuration
BASE_URL="http://localhost:8000"
TEST_ENDPOINT="${BASE_URL}/no_feedback.php"
LOG_FILE="./SQLiMicrobenchmark/mysql/general.log"
VERBOSE=2
LOSS_CRITERIA=10
WIN_CRITERIA=1
EPISODES=10
MAX_TIMESTAMP=10

# Output directory base
OUTPUT_BASE="./test_results"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
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

print_header() {
    echo -e "${CYAN}=========================================="
    echo -e "$1"
    echo -e "==========================================${NC}"
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

    # Test endpoint
    curl -s -o /dev/null -w "%{http_code}" "$TEST_ENDPOINT" > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        print_error "Cannot reach $TEST_ENDPOINT. Please ensure the server is running."
        exit 1
    fi

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

# Function to find model checkpoint directory within a stats_log directory
find_model_checkpoint() {
    local stats_dir=$1
    
    # Look for checkpoint directories (DQN_Agent_Checkpoint, DQN_RND_Agent_Checkpoint, etc.)
    for checkpoint_type in "DQN_RND_Agent_Checkpoint"; do
        if [ -d "${stats_dir}/${checkpoint_type}" ]; then
            echo "${stats_dir}/${checkpoint_type}"
            return 0
        fi
    done
    
    return 1
}

# Function to test a single agent
test_agent() {
    local model_dir=$1
    local source_type=$2  # "non_san" or "sanitized"
    local stats_dirname=$3

    # Create descriptive output directory name
    local output_dir="${OUTPUT_BASE}/TEST_${source_type}_${stats_dirname}_on_no_feedback_${TIMESTAMP}"
    
    print_status "Testing agent from: $model_dir"
    print_status "Source: stats_logs_${source_type}/${stats_dirname}"
    print_status "Output will be saved to: $output_dir"

    python3 sqirl.py \
        -u "$TEST_ENDPOINT" \
        -v $VERBOSE \
        --log_file "$LOG_FILE" \
        --loss_criteria $LOSS_CRITERIA \
        --win_criteria $WIN_CRITERIA \
        --model_dir "$model_dir" \
        --learning False \
        --episodes $EPISODES \
        --max_timestamp $MAX_TIMESTAMP

    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        # Move results from stats_logs to output directory
        mkdir -p "$output_dir"
        
        # Find the most recent stats_logs directory (the one just created)
        latest_result=$(ls -td ./stats_logs/*/ 2>/dev/null | head -1)
        if [ -n "$latest_result" ] && [ -d "$latest_result" ]; then
            cp -r "$latest_result"/* "$output_dir/"
            rm -rf "$latest_result"
            
            # Create a metadata file to track the source
            cat > "${output_dir}/test_metadata.txt" <<EOF
Test Metadata
=============
Test Timestamp: $TIMESTAMP
Source Stats Log: stats_logs_${source_type}/${stats_dirname}
Model Directory: $model_dir
Test Endpoint: $TEST_ENDPOINT
Source Type: $source_type
EOF
        fi
        
        print_success "Completed testing agent from ${source_type}/${stats_dirname}"
    else
        print_error "Failed testing agent from ${source_type}/${stats_dirname} (exit code: $exit_code)"
        return $exit_code
    fi
}

# Function to test all agents in a stats_logs directory
test_agents_from_dir() {
    local stats_logs_dir=$1
    local source_type=$2

    print_header "Testing agents from: $stats_logs_dir ($source_type)"
    
    if [ ! -d "$stats_logs_dir" ]; then
        print_warning "Directory not found: $stats_logs_dir"
        return 1
    fi

    local agent_count=0
    local success_count=0
    local fail_count=0

    # Iterate through each subdirectory in stats_logs_*
    for stats_subdir in "$stats_logs_dir"/*; do
        if [ -d "$stats_subdir" ]; then
            local stats_dirname=$(basename "$stats_subdir")
            
            # Find the model checkpoint in this directory
            model_checkpoint=$(find_model_checkpoint "$stats_subdir")
            
            if [ -n "$model_checkpoint" ]; then
                agent_count=$((agent_count + 1))
                print_status "[$agent_count] Found model: $model_checkpoint"
                
                test_agent "$model_checkpoint" "$source_type" "$stats_dirname"
                
                if [ $? -eq 0 ]; then
                    success_count=$((success_count + 1))
                else
                    fail_count=$((fail_count + 1))
                fi
                
                echo ""
            else
                print_warning "No model checkpoint found in: $stats_subdir"
            fi
        fi
    done

    print_status "Completed testing from $stats_logs_dir"
    print_status "  Total: $agent_count, Success: $success_count, Failed: $fail_count"
    echo ""
}

# Main function
main() {
    print_header "SQIRL Agent Testing Script"
    print_status "Testing all agents on: $TEST_ENDPOINT"
    print_status "Test run timestamp: $TIMESTAMP"
    echo ""

    # Check services before starting
    check_services

    # Create output base directory
    mkdir -p "$OUTPUT_BASE"
    mkdir -p "./stats_logs"

    # Count total agents to test
    NON_SAN_COUNT=$(find ./stats_logs_non_san -maxdepth 2 -type d -name "*Checkpoint*" 2>/dev/null | wc -l)
    SANITIZED_COUNT=$(find ./stats_logs_sanitized -maxdepth 2 -type d -name "*Checkpoint*" 2>/dev/null | wc -l)
    TOTAL_AGENTS=$((NON_SAN_COUNT + SANITIZED_COUNT))

    print_status "Found $NON_SAN_COUNT agents in stats_logs_non_san"
    print_status "Found $SANITIZED_COUNT agents in stats_logs_sanitized"
    print_status "Total agents to test: $TOTAL_AGENTS"
    echo ""

    # Test agents from stats_logs_non_san
    test_agents_from_dir "./stats_logs_non_san" "non_san"

    # Test agents from stats_logs_sanitized
    test_agents_from_dir "./stats_logs_sanitized" "sanitized"

    print_header "Testing Complete!"
    print_status "All test results saved in: $OUTPUT_BASE"
    print_status ""
    print_status "Output directory naming convention:"
    print_status "  TEST_<source_type>_<original_stats_dir>_on_no_feedback_<timestamp>"
    print_status ""
    print_status "Each test directory contains:"
    print_status "  - test_metadata.txt (source tracking info)"
    print_status "  - result_stats_*.stats (test results)"
    print_status "  - actions_Applied_*.stats (action logs)"
    print_status "  - sql_statment_*.stats (SQL statements)"
    print_status "  - rewards_*.stats (reward logs)"
    echo ""
    print_status "To view all test results:"
    print_status "  ls -la $OUTPUT_BASE/"
}

# Run main function
main "$@"




