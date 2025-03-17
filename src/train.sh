#!/bin/bash
#SBATCH --output=output.txt      # Standard output and error log (%j will be replaced by job ID)
#SBATCH --error=error.txt        # Error log
#SBATCH --time=4-12:00:00        # Time limit (4 days)
#SBATCH --partition=gpunodes     # Partition to run the job (use appropriate partition for your cluster)
#SBATCH --gres=gpu:1             # Request one GPU (adjust as needed)
#SBATCH --nodelist=gpunode23
python3 -u train.py --dir_image="data/aotgan/JPEGImages" --dir_mask="data/aotgan/masks" --batch_size=4
