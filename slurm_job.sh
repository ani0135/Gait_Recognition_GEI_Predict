#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4-00:00:00
#SBATCH --job-name=Gait_4
#SBATCH --error=error_train_4
#SBATCH --output=output_train_4
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=1

cd $SLURM_SUBMIT_DIR
source /home/aniketk.scee.iitmandi/exit/bin/activate python37
export PATH=$PATH:/usr/local/cuda/bin/
echo $PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/aniketk.scee.iitmandi/exit/lib/
export LD_LIBRARY_PATH="/usr/local/lib64/:$LD_LIBRARY_PATH"
echo $LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --rdzv_backend c10d --rdzv_endpoint localhost:29501 --nproc_per_node=1 opengait/main.py --cfgs ./configs/gei_predict/gei_predict_GREW.yaml --phase test |& tee -a log_test.txt