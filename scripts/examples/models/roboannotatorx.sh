CUDA_VISIBLE_DEVICES=0

python3 -m roboannotatorx/inference \
    --model RoboAnnotatorX-vicuna-v1 \
    --model-path /8T/klx/kisa-v2/work_dirs/roboannotatex-v3/roboannotatorx-7b-2x4x-grid8-interval32-v3-stage2-image-video-epoch-1/full_model/ \
    --video_dir /8T/klx/kisa-v2/dataset/Finetune/ \
    --video_fps 0 \
    --video_stride 2 \
    --gt_file_question /8T/klx/kisa-v2/dataset/Finetune/mixing_v2_stage3_55K.json \
    --output_dir /8T/klx/kisa-v2/dataset/Output/ \
    --batch_size 1 \
    --test_num 10