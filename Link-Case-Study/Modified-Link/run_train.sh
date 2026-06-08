#!/bin/bash

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Anaconda or Miniconda."
    echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Initialize conda for bash (if not already initialized)
eval "$(conda shell.bash hook)"

# Check if conda environment "link" exists (from environment.yaml)
CONDA_ENV_EXISTS=false
if conda env list | grep -qE "^\s*link\s"; then
    CONDA_ENV_EXISTS=true
elif [ -d "venv" ] && [ -f "venv/pyvenv.cfg" ]; then
    # Check if local venv directory exists (from environment.yaml prefix)
    CONDA_ENV_EXISTS=true
fi

# Create or update conda environment
if [ "$CONDA_ENV_EXISTS" = false ]; then
    echo "Creating conda environment 'link' from environment.yaml..."
    conda env create -f environment.yaml -n link
else
    echo "Conda environment 'link' exists. Updating from environment.yaml..."
    conda env update -f environment.yaml -n link --prune
fi


PYTHON_CMD="conda run --live-stream -n link python"

cd wavsep-docker
echo "Building wavsep-train container..."
docker build -t wavsep-train .
echo "Running wavsep-train container..."
docker run -d --rm --name wavsep-train -p 8080:8080 wavsep-train
echo "Wavsep-train container running..."

cd ..
#$PYTHON_CMD train_A2C.py -u 'http://localhost:8080/wavsep/active/index-xss.jsp' -t 3500000 -d

for i in {1..20}; do
    #run with deterministic state-action space
    $PYTHON_CMD train_A2C.py -u 'http://localhost:8080/wavsep/active/Reflected-XSS/RXSS-Detection-Evaluation-GET/index.jsp' -t 3500000 -da

    #run with non-deterministic state-action space
    $PYTHON_CMD train_A2C.py -u 'http://localhost:8080/wavsep/active/Reflected-XSS/RXSS-Detection-Evaluation-GET/index.jsp' -t 3500000 -ds

    #run with deterministic state-action space
    $PYTHON_CMD train_A2C.py -u 'http://localhost:8080/wavsep/active/Reflected-XSS/RXSS-Detection-Evaluation-GET/index.jsp' -t 3500000 -d

    #run with non-deterministic state-action space
    $PYTHON_CMD train_A2C.py -u 'http://localhost:8080/wavsep/active/Reflected-XSS/RXSS-Detection-Evaluation-GET/index.jsp' -t 3500000

done


# Stop container if it exists
if docker ps -a --format '{{.Names}}' | grep -q "^wavsep-train$"; then
    docker stop wavsep-train
fi
