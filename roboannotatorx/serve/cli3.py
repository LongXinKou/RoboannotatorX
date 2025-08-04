import argparse
import torch

from roboannotatorx.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from roboannotatorx.conversation import conv_templates, SeparatorStyle
from roboannotatorx.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
from decord import VideoReader, cpu

#---------#
import json
import random
import os
#---------#

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def load_video(video_path, fps=1):
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames

def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit)

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower() or "vid" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    if args.image_file is not None:
        if '.mp4' in args.image_file:
            image = load_video(args.image_file)
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
            image_tensor = [image_tensor]
        else:
            image = load_image(args.image_file)
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
    else:
        image_tensor = None
        
        
        
    task_list = [
        "move-block-in-x-area-to-y-area",
        "pick-and-place-primitive",
        "pick-and-place-primitive-with-size",
        "put-block-in-matching-bowl",
        "put-block-in-mismatching-bowl",
        "stack-block-of-same-color",
        "stack-block-of-same-size",
        "stack-smaller-over-bigger-with-same-color",
        "stack-smaller-over-bigger-with-same-color-in-same-color-zone"
    ]
    file_path=args.image_file
    dirs = file_path.split('/')[-2:] 
    base_dir = '/'.join(file_path.split('/')[:-3]) 
    task_dir = os.path.join(base_dir, dirs[0])
    index = task_list.index(dirs[0])
    cnt = 0
    
    while True:
        if cnt==11:
            break
        sub_dir =  os.listdir(task_dir)[cnt]
        cnt = cnt + 1
        full_sub_dir_path = os.path.join(task_dir, sub_dir)
        json_file_path = os.path.join(full_sub_dir_path, dirs[1][:2] + '.json')
        output_path = '/home/ubuntu/LLaMA-VID/for_test_lohoravens/'
        output_path = os.path.join(output_path, sub_dir)
        output_path_prompt = os.path.join(output_path , f"random_prompt_lohoravens_{index:01d}.txt") 
        output_path_outputs = os.path.join(output_path , f"output_lohoravens_{index:01d}.txt") 
        
        try:
            if os.path.exists(json_file_path):
                with open(json_file_path, 'r', encoding='utf-8') as file1:
                    data = json.load(file1)
                           
                caption = data['prompt']
                print(f"Selected caption: {caption}")
                inp = caption
                        
                with open(output_path_prompt, 'a', encoding='utf-8') as file2:
                    file2.write(inp+"\n")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"{roles[1]}: ", end="")

        model.update_prompt([[inp]])

        if args.image_file is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        conv.messages[-1][-1] = outputs
        conv.messages[-2][-1] = conv.messages[-2][-1].replace(DEFAULT_IMAGE_TOKEN+'\n','')
        #---------#
        target=outputs.replace('</s>', '\n')
        with open(output_path_outputs, 'a', encoding='utf-8') as file3:
            file3.write(target)
        #---------#

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default=None)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.5) # set to 0.5 for video, 0.2 for image
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
