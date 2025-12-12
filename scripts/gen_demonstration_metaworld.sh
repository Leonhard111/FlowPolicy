# bash scripts/gen_demonstration_metaworld.sh basketball



#cd third_party/Metaworld
#"${CURRENT_DIR}/data/code/FlowPolicy/data/" \
task_name=${1}
CURRENT_DIR=$(pwd)
export CUDA_VISIBLE_DEVICES=0,1
python third_party/Metaworld/gen_demonstration_expert.py --env_name=${task_name} \
            --num_episodes 10 \
            --root_dir "${CURRENT_DIR}/data/" 
