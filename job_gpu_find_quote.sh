#!/bin/bash
#SBATCH --job-name=Embedding               # Job name  
#SBATCH -N1 --ntasks-per-node=1          # Number of nodes and cores per node required 
#SBATCH --gres=gpu:H100:1                # GPU type (H100) and number of GPUs 
#SBATCH --mem-per-gpu=64GB              # Memory per CPU core, 8 CPUs/GPU 
#SBATCH --time=2:00:00                        # Duration of the job (Ex: 1 hour) 
#SBATCH -o Report-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL       # Mail preferences

# Create or activate virtual environment
module load python/3.10
if [ ! -d "embedding_env" ]; then
    echo "Creating virtual environment..."
    python3.10 -m venv embedding_env
    source embedding_env/bin/activate
    echo "Installing dependencies..."
    pip install -r requirements.txt
else
    echo "Using existing virtual environment..."
    source embedding_env/bin/activate
fi

python -u find_quote.py
