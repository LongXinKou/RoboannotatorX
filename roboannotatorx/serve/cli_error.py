import argparse
import torch

from roboannotatorx.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from roboannotatorx.conversation import conv_templates, SeparatorStyle
from roboannotatorx.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
from decord import VideoReader, cpu

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

import json
import os
import random


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
        
    #---------#
    cnt=0
    json_file_path = '/data/ubuntu/lohoravens_error/test/llamavid_output/output_llamavid.json'
    json_file_path_annotation = "/data/ubuntu/lohoravens_error/test/llamavid_output/annotation.json"
    number = int(os.path.splitext(os.path.basename(args.image_file))[0])
    print(number)
    '''
    if number < 100:
        lang = "put the blocks in the bowls with matching colors."
    elif number >=100 and number <200:
        lang = "put the blocks in the bowls with mismatched colors."
    elif number >=200 and number <300:
        lang = "stack all the smaller blocks over the bigger blocks of the same color."
    elif number >=300 and number<400:
        lang = "stack all the smaller blocks over the bigger blocks of the same color in the zone same color."
    '''
    with open(json_file_path_annotation, 'r', encoding='utf-8') as file:
        data_annotation = json.load(file)
    #---------#
    while True:
        try:
            #------------------------------#
            if cnt==1:
                break
            '''
            inp="Background information: When a robot receives a language instruction, it needs to break down the language instruction into many sub-goals that need to be completed, and complete the initial language instruction by completing each sub-goal one by one. For example, if the language instruction is: Put the plate in the refrigerator on the table, then the sub-goals should be: 1. Approach the refrigerator 2. Open the refrigerator door 3. Take out the plate 4. Close the refrigerator door 5. Approach the table 6. Put down the plate. One on-site image is taken after each sub-goal is completed. However, sometimes even if all sub-goals have been completed, the original language instruction have not been implemented, which is due to the unreasonable formulation of sub-goals. Now all sub-goals have been completed, but the language instruction have not been implemented yet due to the unreasonable formulation of sub-goals. As an expert in error detection, you need to determine which images are unreasonable (i.e. which sub-goals are unreasonable) based on the language instruction and a set of images .It should be noted that 1. Each image is taken after each sub-goal is completed, so each image represents one sub-goal. 2.Unreasonable images refer to the sub-goals they represent are unreasonable. 3. Unreasonable images may be one or multiple. 4. The information displayed in each image not only shows the completion of the current new sub goals, but also displays the completion of the sub goals completed before, as the sub-goals are executed in an orderly manner. If the current sub-goal does not violate the language instruction, but the previously completed sub-goal violates the language instruction, then the current image should be considered reasonable, because the sub-goal it represents matches the language instruction. If the sub-goal the current image represents violates the language instruction, it is considered that this image is incorrect, because the sub-goal it represents doesn't match the language instruction. 5.All images start counting from 1. The on-site images have been provided.Language instruction:" + lang +"Your output format is as follows:Unreasonable image: [],Reason:[],Correct sub goals: []"
            '''
            inp = data_annotation[number]["conversations"][0]["value"].rstrip("\n<image>")
            print(inp)
            cnt = cnt +1
            #------------------------------#
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
        # print(conv.messages[-1][-1])
        target=outputs.replace('</s>', '')
        #with open('/data/ubuntu/lohoravens_error/test/llamavid_output/output_lohoravens.txt', 'a', encoding='utf-8') as file3:
            #file3.write(target)
        # 检查文件是否存在
        if os.path.exists(json_file_path):
            # 如果文件存在，读取现有内容
            with open(json_file_path, 'r', encoding='utf-8') as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError:
                    data = []
        else:
            # 如果文件不存在，初始化为空列表
            data = []
        target = {
            "video":  f"{number}.mp4",
            "conversations": [
                {
                    "from": "gpt",
                    "value": target
                }
            ]
        }
        data.append(target)
        # 将更新后的数据写回 JSON 文件
        with open(json_file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
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
