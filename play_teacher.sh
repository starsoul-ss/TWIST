
# bash eval_teacher.sh g1 0403_amass_omomo_improve_knee

# bash eval_teacher.sh g1 0402_15k_new_arch

# bash eval_teacher.sh g1 0330_amass_omomo

# bash eval_teacher.sh toddy 0412_amass_omomo

# bash eval_teacher.sh t1 0331_amass_omomo

export CUDA_VISIBLE_DEVICES=6

robot_name=$1
task_name="${robot_name}_priv_mimic"
proj_name="${robot_name}_priv_mimic"
exptid=$2

cd legged_gym/legged_gym/scripts

# Run the eval script
python play.py --task "${task_name}" \
                --proj_name "${proj_name}" \
                --exptid "${exptid}" \
                --num_envs 1 \
                --record_video \
                # --headless \
                # --checkpoint 13000 \
                # --record_log \
                # --use_jit \
                # --teleop_mode
