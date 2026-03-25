CUDA_VISIBLE_DEVICES=1 python custom_dataset/convert_h5_to_rlds.py \
    --input_dir /data/public/NAS/VLANeXt/dataset/New2/collected_data_merged \
    --output_dir /data/public/NAS/LIBERO_modified/needle_insertion2 \
    --dataset_name needle_insertion \
    --max_episodes 0 \
    --num_shards 16