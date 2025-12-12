# Examples:
# bash scripts/train_policy_image.sh flowpolicy adroit_hammer 0129 0 3

DEBUG=FALSE 
save_ckpt=True

alg_name=${1}
task_name=${2}
# Automatically append _image to config and task names
config_name=${alg_name}_image
task_config_name=${task_name}_image

addition_info=${3}
seed=${4}
exp_name=${task_name}-image-${alg_name}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"

CURRENT_DIR=$(pwd)
# gpu_id=$(bash scripts/find_gpu.sh)
gpu_id=${5}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


if [ $DEBUG = True ]; then
    wandb_mode=offline
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi

cd $(pwd)/FlowPolicy

# Use the original task name for dataset path (assuming data is shared)
dataset_path="${CURRENT_DIR}/data/${task_name}_expert.zarr"

echo "Config Name: ${config_name}"
echo "Task Config: ${task_config_name}"
echo "Dataset Path: ${dataset_path}"

export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}
python train.py --config-name=${config_name}.yaml \
                            task=${task_config_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0"\
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt}  \
                            task.dataset.zarr_path="${dataset_path}" \
