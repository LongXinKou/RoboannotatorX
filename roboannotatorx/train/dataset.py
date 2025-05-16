import transformers
import json
import copy
import os
import torch
import random

from typing import Dict, Optional, Sequence, List
from dataclasses import dataclass, field
from torch.utils.data import Dataset
from PIL import Image
from decord import VideoReader, cpu

from config import DataArguments

local_rank = None
def rank0_print(*args):
    if local_rank == 0:
        print(*args)

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if ('image' in sample) or ('video' in sample) else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        attempt, max_attempt = 0, 10
        while attempt < max_attempt:
            try:
                sources = self.list_data_dict[i]
                suffix = None
                if isinstance(i, int):
                    sources = [sources]
                assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

                if 'image' in sources[0]:
                    image_file = self.list_data_dict[i]['image']
                    image_folder = self.data_args.image_folder
                    processor = self.data_args.image_processor
                    image_file = os.path.join(image_folder, image_file)
                    image = Image.open(image_file).convert('RGB')
                    if self.data_args.image_aspect_ratio == 'pad':
                        def expand2square(pil_img, background_color):
                            width, height = pil_img.size
                            if width == height:
                                return pil_img
                            elif width > height:
                                result = Image.new(pil_img.mode, (width, width), background_color)
                                result.paste(pil_img, (0, (width - height) // 2))
                                return result
                            else:
                                result = Image.new(pil_img.mode, (height, height), background_color)
                                result.paste(pil_img, ((height - width) // 2, 0))
                                return result

                        image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
                        image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                    else:
                        image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                    sources = preprocess_multimodal(
                        copy.deepcopy([e["conversations"] for e in sources]),
                        self.data_args)
                elif 'video' in sources[0]:
                    video_file = self.list_data_dict[i]['video']
                    video_folder = self.data_args.video_folder
                    video_file = os.path.join(video_folder, video_file)
                    suffix = video_file.split('.')[-1]

                    vr = VideoReader(video_file, ctx=cpu(0))
                    total_frames = len(vr)
                    if total_frames == 1:
                        raise ValueError("Single frame video detected")

                    if self.data_args.video_fps != 0:
                        sample_fps = round(vr.get_avg_fps() / self.data_args.video_fps)
                        frame_idx = [i for i in range(0, total_frames, sample_fps)]
                    else:
                        frame_idx = [i for i in range(0, total_frames)]
                        # 4x downsampling
                        stride = 4 if total_frames > 1000 else 2
                        frame_idx = list(range(0, total_frames, stride))
                        if frame_idx[-1] != total_frames - 1:
                            frame_idx.append(total_frames - 1)

                    video = vr.get_batch(frame_idx).asnumpy()
                    processor = self.data_args.image_processor
                    image = processor.preprocess(video, return_tensors='pt')['pixel_values']
                    sources = preprocess_multimodal(
                        copy.deepcopy([e["conversations"] for e in sources]),
                        self.data_args)
                else:
                    sources = copy.deepcopy([e["conversations"] for e in sources])
                break
            except Exception as e:
                attempt += 1
                print(f"Error: {e} in loading {i}, retrying...")
                print(sources)
                i = random.randint(0, len(self.list_data_dict) - 1)