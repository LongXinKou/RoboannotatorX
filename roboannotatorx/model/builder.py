import os

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch

from roboannotatorx.model import *
from roboannotatorx.model.llava_utils.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def load_roboannotator(model_path, model_base=None, load_8bit=False, load_4bit=False, device_map="auto", device="cuda",
                       args=None):
    kwargs = {"device_map": device_map}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    # Load base model + LoRA adapter if model_base is specified
    if model_base is not None:
        print('Loading LLM...')
        print('Loading Tokenizer...')
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        cfg_pretrained = AutoConfig.from_pretrained(model_path)
        model = LlavaLlamaRobotForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained,
                                                           **kwargs)
        print('Loading LoRA Adapter...')
        from peft import PeftModel, PeftConfig
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
        model.to(torch.float16)
    # If only a single (merged) model is provided, load directly
    else:
        print('Loading LLM...')
        print('Loading Tokenizer...')
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = LlavaLlamaRobotForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        model.config.model_path = model_path

    print('Initialize Tokenizer...')
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    print('Loading Vision Module...')
    model.get_model().initialize_vision_modules(model_args=model.config, for_eval=True)
    image_processor = model.get_vision_tower().image_processor

    print('Loading Attention Module...')
    model.get_model().initialize_attention_modules(model.config, for_eval=True)

    print('Loading additional weights if exist...')
    if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
        non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in
                               non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}

        model.load_state_dict(non_lora_trainables, strict=False)

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len