import argparse
import json
import os
import numpy as np
import random
import torch

from tqdm import tqdm
from decord import VideoReader, cpu
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria, process_images

from roboannotatorx.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from roboannotatorx.conversation import conv_templates, SeparatorStyle
from roboannotatorx.model.builder import load_roboannotator


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument("--model", type=str, default="RoboAnnotatorX", help="Model name")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--model-max-length", type=int, default=None)

    parser.add_argument('--video_dir', help='Directory containing video files.', default=None)
    parser.add_argument('--gt_file_question', help='Path to the ground truth file containing question.', default=None)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', default=None)
    
    # test
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--test_num', type=int, default=200)
    parser.add_argument('--interval', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


def load_video(video_path):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    frame_idx = [i for i in range(0, len(vr))]
    if len(frame_idx) > 2000: # 4x downsampling
        frame_idx = frame_idx[::4]
    elif len(frame_idx) > 500: # 2x downsampling
        frame_idx = frame_idx[::2]
    total_frames = vr.get_batch(frame_idx).asnumpy()
    return total_frames, total_frame_num
    

def get_dataset_name_from_path(model_path, info='mix'):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1] != 'full_model':
        if model_paths[-1] != 'Finetune':
            return model_paths[-1]
        else:
            return info.strip("/").split("/")[-1].split('.')[0]
    else:
        return '_'.join(model_paths[-2:])


def run_inference(args):
    random.seed(args.seed)
    print(f"seed {args.seed}")

    model_name = args.model if args.model else get_dataset_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_roboannotator(
        model_path = args.model_path, 
        model_base = args.model_base, 
    )

    conv_mode = args.conv_mode
        
    # Load both ground truth file containing questions and answers
    with open(args.gt_file_question) as file:
        gt_questions_answers = json.load(file)

    # Create the output directory if it doesn't exist
    dataset_name = get_dataset_name_from_path(args.video_dir, args.gt_file_question)
    output_dir = os.path.join(args.output_dir, dataset_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    test_num = args.test_num if args.test_num < len(gt_questions_answers) else len(gt_questions_answers)
    answers_file = os.path.join(output_dir, f"{model_name}_{test_num}.json")
    ans_file = open(answers_file, "w")

    test_questions_answers = random.sample(gt_questions_answers, test_num)
    total_frames = []
    index = 0

    for sample in tqdm(test_questions_answers):
        video_name = sample['video']
        question = sample['conversations'][0]['value']
        answer = sample['conversations'][1]['value']
        question_type = sample['question_type']

        sample_set = {'id': video_name, 'question': question, 'answer': answer, 'question_type': question_type}
        index += 1

        # Load the video file
        video_path = os.path.join(args.video_dir, video_name)

        # Check if the video exists
        image_tensor = []
        if os.path.exists(video_path):
            video, total_frame = load_video(video_path)
            total_frames.append(total_frame)
            video = image_processor.preprocess(video, return_tensors='pt')['pixel_values'].half().cuda()
            image_tensor.append(video)
        
        qs = question
        
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            if DEFAULT_IMAGE_TOKEN not in qs:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
                
        # w/o FLOPS
        with torch.inference_mode():
            model.update_prompt([[question]])
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        sample_set['pred'] = outputs
        ans_file.write(json.dumps(sample_set) + "\n")
        ans_file.flush()

    ans_file.close()
    print(f'avg frame number{np.mean(total_frames)}, max frame number{np.max(total_frames)}')

if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
