#!/bin/bash
#SBATCH --job-name=paper_id_check
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/gpfs/DANQIC/jz4391/bargain/slurm/paper_id_check_%j.out
#SBATCH --error=/scratch/gpfs/DANQIC/jz4391/bargain/slurm/paper_id_check_%j.err

set -euo pipefail

cd /scratch/gpfs/DANQIC/jz4391/bargain
exec .venv/bin/python scripts/build_paper_experiment_data_manifest.py --check --check-identifiers
