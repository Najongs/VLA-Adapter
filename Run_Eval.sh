CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/data/public/NAS/VLA-Adapter/LIBERO:$PYTHONPATH python experiments/robot/libero/run_libero_eval.py \
    --use_proprio True \
    --num_images_in_input 2 \
    --use_film False \
    --pretrained_checkpoint outputs/LIBERO-Spatial-Pro \
    --task_suite_name libero_spatial \
    --use_pro_version True