#!/bin/bash
#SBATCH --job-name=BO
#SBATCH --partition=gpuq
#SBATCH --nodes=1                
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=30
#SBATCH --hint=nomultithread
#SBATCH --time=120:00:00
#SBATCH --output=BO_%j.out
#SBATCH --error=BO_%j.out

# before running:
# first create env provided by Max
# pip install transformers
# pip install accelerate
# pip install tiktoken
# pip install sentencepiece
# pip install scikit-optimize
# download pretrained model in ./from_pretrained (by changing 'AutoModel.from_pretrained()')

source /home/echen/anaconda3/etc/profile.d/conda.sh # CHANGE THIS
conda activate BO

set -x


python main.py --dataset shields --seed 1337 --mc_runs 20 --batch_size 1 --n_iter 50 --switch_after 5


python main.py --dataset buchwald_hartwig --seed 1337 --mc_runs 20 --batch_size 1 --n_iter 50 --impute_mode ignore --switch_after 5


python main.py --dataset suzuki --seed 1337 --mc_runs 20 --batch_size 1 --n_iter 50 --switch_after 5


python main.py --dataset ni_catalyzed_1 --seed 1337 --mc_runs 20 --batch_size 1 --n_iter 100 --switch_after 5

python main.py --dataset ni_catalyzed_2 --seed 1337 --mc_runs 20 --batch_size 1 --n_iter 100 --switch_after 5

python main.py --dataset ni_catalyzed_3 --seed 1337 --mc_runs 20 --batch_size 1 --n_iter 100 --switch_after 5

python main.py --dataset ni_catalyzed_4 --seed 1337 --mc_runs 20 --batch_size 1 --n_iter 100 --switch_after 5

