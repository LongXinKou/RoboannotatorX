import argparse
import torch
import random
import numpy as np
import json
import yaml
import os

from tqdm import tqdm

from roboannotatorx.utils import disable_torch_init
from roboannotatorx.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria, process_video_with_decord
from roboannotatorx.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from roboannotatorx.conversation import conv_templates, SeparatorStyle
from roboannotatorx.model.builder import load_roboannotator


def parse_args():
    parser = argparse.ArgumentParser()

    # ModelArguments
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--model-max-length", type=int, default=None)
    parser.add_argument('--interval', type=int, default=None)

    # DataArguments
    parser.add_argument('--data_path', help='Path to the ground truth file containing question.', default=None)
    parser.add_argument('--image_folder', help='Directory containing image files.', default=None)
    parser.add_argument('--video_folder', help='Directory containing video files.', default=None)
    parser.add_argument("--video_fps", type=int, default=0)
    parser.add_argument("--video_stride", type=int, default=2)

    # TestArguments
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


def get_dataset_name_from_path(dataset_path):
    dataset_path = dataset_path.strip("/")
    dataset_paths = dataset_path.split("/")
    return dataset_paths[-1]


def run_inference(args):
    random.seed(args.seed)
    print(f"seed {args.seed}")

    # Model
    disable_torch_init()

    model_name = args.model
    tokenizer, model, image_processor, context_len = load_roboannotator(
        model_path=args.model_path,
        model_base=args.model_base,
    )

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    # Dataset
    test_dataset = []
    if args.data_path.endswith(".json"):
        with open(args.data_path, "r") as file:
            test_dataset = json.load(file)
            print(f"Testing {args.data_path}")
    elif args.data_path.endswith(".yaml"):
        with open(args.data_path, "r") as file:
            yaml_data = yaml.safe_load(file)
            datasets = yaml_data.get("datasets", [])
            dataset_paths = [dataset.get("json_path") for dataset in datasets]
            print(f"Testing {dataset_paths}")
            for dataset in datasets:
                json_path = dataset.get("json_path")
                sampling_strategy = dataset.get("sampling_strategy", "all")
                sampling_number = None

                if json_path.endswith(".jsonl"):
                    cur_data_dict = []
                    with open(json_path, "r") as json_file:
                        for line in json_file:
                            cur_data_dict.append(json.loads(line.strip()))
                elif json_path.endswith(".json"):
                    with open(json_path, "r") as json_file:
                        cur_data_dict = json.load(json_file)
                else:
                    raise ValueError(f"Unsupported file type: {json_path}")

                if ":" in sampling_strategy:
                    sampling_strategy, sampling_number = sampling_strategy.split(":")
                    if "%" in sampling_number:
                        sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                    else:
                        sampling_number = int(sampling_number)

                # Apply the sampling strategy
                if sampling_strategy == "first" and sampling_number is not None:
                    cur_data_dict = cur_data_dict[:sampling_number]
                elif sampling_strategy == "end" and sampling_number is not None:
                    cur_data_dict = cur_data_dict[-sampling_number:]
                elif sampling_strategy == "random" and sampling_number is not None:
                    random.shuffle(cur_data_dict)
                    cur_data_dict = cur_data_dict[:sampling_number]
                test_dataset.extend(cur_data_dict)
    else:
        raise ValueError(f"Unsupported file type: {args.data_path}")

    # Test
    dataset_name = get_dataset_name_from_path(args.data_path)
    output_dir = os.path.join(args.output_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    test_num = len(test_dataset)
    answers_file = os.path.join(output_dir, f"{model_name}_{test_num}.json")
    ans_file = open(answers_file, "w")

    total_frames = []
    for sample in tqdm(test_dataset):
        if 'image' in sample:
            pass
        elif 'video' in sample:
        video_name = sample['video']
        question = sample['conversations'][0]['value']
        answer = sample['conversations'][1]['value']
        question_type = sample['question_type']

        sample_set = {'id': video_name, 'question': question, 'answer': answer, 'question_type': question_type}


        # Load the video file
        video_path = os.path.join(args.video_dir, video_name)

        # Check if the video exists
        if os.path.exists(video_path):
            video, total_frame_num = process_video_with_decord(
                video_path = video_path,
                image_processor=image_processor,
                video_fps = args.video_fps,
                video_stride = args.video_stride,
            )
            total_frames.append(total_frame_num)
            video = image_processor.preprocess(video, return_tensors='pt')['pixel_values'].half().cuda()
            video = [video]

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

        cur_prompt = question
        with torch.inference_mode():
            model.update_prompt([[cur_prompt]])
            output_ids = model.generate(
                input_ids,
                images=video,
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
