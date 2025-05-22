CUDA_VISIBLE_DEVICES=0

python3 -m roboannotatorx/inference \
    --model RoboAnnotatorX-vicuna-v1 \
    --model-path RoboAnnotatorX \
    --data_path  \
    --image_folder  \
    --video_folder  \
    --video_fps 0 \
    --video_stride 2 \
    --output_dir /8T/klx/kisa-v2/dataset/Output/