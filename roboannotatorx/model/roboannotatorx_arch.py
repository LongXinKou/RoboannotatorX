#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# ------------------------------------------------------------------------
# Modified from LLaVA (https://github.com/haotian-liu/LLaVA)
# Copyright 2023 Yanwei Li
# ------------------------------------------------------------------------

from abc import ABC, abstractmethod
import os
import json
import einops

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import BertLMHeadModel as BertLMHeadModelRaw

from .qformer import BertConfig
from .qformer import BertLMHeadModel as BertLMHeadModelQF
from .temporal_encoder import TransformerEncoder

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from roboannotatorx.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN


class RoboAnnotatorMetaModel:

    def __init__(self, config):
        super(RoboAnnotatorMetaModel, self).__init__(config)

        # For evaluation mode: if vision tower is defined in config, build it with delayed loading
        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None, max_token=2048, for_eval=False):
        def get_w(weights, keyword):
            return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

        model_save_path = getattr(model_args, 'model_name_or_path', None) or getattr(model_args, 'model_path', None)

        vision_tower = getattr(model_args, "vision_tower", None)
        image_processor = getattr(model_args, 'image_processor', None)
        mm_vision_select_layer = getattr(model_args, "mm_vision_select_layer", None)
        mm_vision_select_feature = getattr(model_args, "mm_vision_select_feature", None)
        mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        pretrain_mm_mlp_adapter = getattr(model_args, "pretrain_mm_mlp_adapter", None)
        max_frame_pos = getattr(model_args, "max_frame_pos", None)

        self.config.mm_vision_tower = vision_tower
        self.config.image_processor = image_processor

        if getattr(self, 'vision_tower', None) is None:
            vision_tower = build_vision_tower(model_args)
            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        elif not self.vision_tower.is_loaded:
            self.vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = mm_projector_type
        self.config.mm_hidden_size = self.vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.max_token = max_token
        self.max_frame_pos = max_frame_pos

        if getattr(self, 'mm_projector', None) is None: # Pretraining
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
        else:
            trainable_module = ["mm_projector"]
            weight_file = json.load(open(os.path.join(model_save_path, 'pytorch_model.bin.index.json'), 'r'))[
                'weight_map']
            model_path = set([weight_file[_key] for _key in weight_file if
                              any([_module in _key for _module in trainable_module])])
            mm_projector_weights = {}
            for _model in model_path:
                mm_projector_weights.update(torch.load(os.path.join(model_save_path, _model), map_location='cpu'))

        # If no projector weights found, skip loading
        if len(mm_projector_weights) == 0:
            return
        self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

        if for_eval:
            weight_type = torch.float16
            device_type = self.mm_projector[0].weight.device
            self.vision_tower = self.vision_tower.to(device=device_type, dtype=weight_type)

    def initialize_attention_modules(self, model_args, for_eval=False):
        def get_w(weights, keyword):
            return {k.split(keyword + '.')[1]: v for k, v in weights.items() if
                    keyword in k and len(k.split(keyword + '.')) > 1}

        model_save_path = getattr(model_args, 'model_name_or_path', None) or getattr(model_args, 'model_path', None)

        pretrain_mm_mlp_adapter = getattr(model_args, "pretrain_mm_mlp_adapter", None)
        pretrain_qformer = getattr(model_args, "pretrain_qformer", None)
        self.config.bert_type = getattr(model_args, "bert_type", "qformer")
        self.config.num_query = getattr(model_args, "num_query", 32)
        self.config.compress_type = getattr(model_args, "compress_type", None)

        if 'pretrain' in self.config.bert_type:
            # for qformer that use evaclip for prtrain
            att_feat_size = 1408
        else:
            att_feat_size = self.config.mm_hidden_size

        # print('Loading Q-Former')
        self.vlm_att_tokenlizer, self.vlm_att_encoder, self.vlm_att_query = self.init_Qformer(att_feat_size,
                                                                                              truncation_side="left")
        self.vlm_att_projector = torch.nn.Linear(self.vlm_att_encoder.config.hidden_size, self.config.mm_hidden_size)
        self.vlm_att_key_projector = torch.nn.Linear(self.config.mm_hidden_size, self.config.mm_hidden_size)
        self.vlm_att_val_projector = torch.nn.Linear(self.config.mm_hidden_size, self.config.hidden_size)

        # print('Loading Clip Q-Former')
        _, self.clip_qformer, self.clip_qformer_query = self.init_Qformer(att_feat_size)
        self.clip_qformer_projector = torch.nn.Linear(self.clip_qformer.config.hidden_size, self.config.hidden_size)

        # print('Loading Motion Encoder')
        self.motion_encoder = self.init_motion_encoder(self.config.mm_hidden_size)

        # Parameter sharing
        if "pretrain" in self.config.bert_type and self.config.mm_hidden_size != att_feat_size:
            self.vlm_att_bert_proj = torch.nn.Linear(self.config.mm_hidden_size, att_feat_size)
        else:
            self.vlm_att_bert_proj = None
        if 'qformer_pretrain' in self.config.bert_type:
            self.vlm_att_ln = torch.nn.LayerNorm(att_feat_size)
            self.clip_qformer_ln = torch.nn.LayerNorm(att_feat_size)

        if pretrain_qformer is not None:  # stage 1
            print("Loading pretrained qformer weights...")
            qformer_weight = torch.load(pretrain_qformer, map_location='cpu')['model']
            bert_weight = {_key: qformer_weight[_key] for _key in qformer_weight if 'bert' in _key}
            self.vlm_att_encoder.load_state_dict(get_w(bert_weight, 'Qformer'))
            self.vlm_att_ln.load_state_dict(get_w(qformer_weight, 'ln_vision'))
            self.vlm_att_query.data = qformer_weight['query_tokens']

            self.clip_qformer.load_state_dict(get_w(bert_weight, 'Qformer'))
            self.clip_qformer_ln.load_state_dict(get_w(qformer_weight, 'ln_vision'))
            self.clip_qformer_query.data = qformer_weight['query_tokens']

        if 'freeze_all' in self.config.bert_type:
            print("Freezing all qformer weights...")
            self.vlm_att_encoder.requires_grad_(False)
            self.vlm_att_ln.requires_grad_(False)
            self.vlm_att_query.requires_grad_(False)
            self.vlm_att_projector.requires_grad_(False)
            self.vlm_att_key_projector.requires_grad_(False)
            self.vlm_att_val_projector.requires_grad_(False)
            self.clip_qformer.requires_grad_(False)
            self.clip_qformer_ln.requires_grad_(False)
            self.clip_qformer_query.requires_grad_(False)
            self.clip_qformer_projector.requires_grad_(False)

            # self.motion_encoder.requires_grad_(False)
        elif 'freeze' in self.config.bert_type:  # stage 1
            print("Freezing pretrained qformer weights...")
            self.vlm_att_encoder.requires_grad_(False)
            self.vlm_att_ln.requires_grad_(False)
            self.vlm_att_query.requires_grad_(False)
            self.clip_qformer.requires_grad_(False)
            self.clip_qformer_ln.requires_grad_(False)
            self.clip_qformer_query.requires_grad_(False)

            self.motion_encoder.requires_grad_(False)

        print('Loading pretrained weights...')
        if pretrain_mm_mlp_adapter is not None:  # stage 2
            att_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
        elif model_save_path is not None:  # stage 3/full model
            trainable_module = ['vlm_att_encoder', 'vlm_att_projector', 'vlm_att_key_projector',
                                'vlm_att_val_projector', 'vlm_att_query', 'vlm_att_visual_proj',
                                'vlm_att_ln',
                                'clip_qformer', 'clip_qformer_query', 'clip_qformer_projector',
                                'clip_qformer_ln',
                                'motion_encoder']
            model_idx_path = model_save_path
            weight_file = json.load(open(os.path.join(model_idx_path, 'pytorch_model.bin.index.json'), 'r'))[
                'weight_map']
            model_path = set(
                [weight_file[_key] for _key in weight_file if any([_module in _key for _module in trainable_module])])
            att_projector_weights = {}
            for _model in model_path:
                att_projector_weights.update(torch.load(os.path.join(model_idx_path, _model), map_location='cpu'))
            if len(att_projector_weights) == 0:
                return
        else:
            att_projector_weights = {}

        if len(att_projector_weights) > 0:
            bert_dict = get_w(att_projector_weights, 'vlm_att_encoder')
            if "bert.embeddings.position_ids" not in bert_dict and "raw_bert" not in self.config.bert_type:
                bert_dict["bert.embeddings.position_ids"] = self.vlm_att_encoder.bert.embeddings.position_ids
            self.vlm_att_encoder.load_state_dict(bert_dict)
            self.vlm_att_projector.load_state_dict(get_w(att_projector_weights, 'vlm_att_projector'))
            self.vlm_att_key_projector.load_state_dict(get_w(att_projector_weights, 'vlm_att_key_projector'))
            self.vlm_att_val_projector.load_state_dict(get_w(att_projector_weights, 'vlm_att_val_projector'))
            self.vlm_att_query.data = att_projector_weights['model.vlm_att_query']
            if "pretrain" in self.config.bert_type:
                self.vlm_att_ln.load_state_dict(get_w(att_projector_weights, 'vlm_att_ln'))
            if self.vlm_att_bert_proj is not None:
                self.vlm_att_bert_proj.load_state_dict(get_w(att_projector_weights, 'vlm_att_bert_proj'))

            bert_dict = get_w(att_projector_weights, 'clip_qformer')
            if "bert.embeddings.position_ids" not in bert_dict:
                bert_dict["bert.embeddings.position_ids"] = self.clip_qformer.bert.embeddings.position_ids
            self.clip_qformer.load_state_dict(bert_dict)
            self.clip_qformer_projector.load_state_dict(get_w(att_projector_weights, 'clip_qformer_projector'))
            self.clip_qformer_query.data = att_projector_weights['model.clip_qformer_query']
            if "pretrain" in self.config.bert_type:
                self.clip_qformer_ln.load_state_dict(get_w(att_projector_weights, 'clip_qformer_ln'))

            motion_weights = get_w(att_projector_weights, 'motion_encoder')
            if hasattr(self.motion_encoder, 'pos_encoder') and 'pos_encoder.pe' not in motion_weights:
                motion_weights['pos_encoder.pe'] = self.motion_encoder.pos_encoder.pe
            self.motion_encoder.load_state_dict(motion_weights)

        if for_eval:
            weight_type = torch.float16
            device_type = self.mm_projector[0].weight.device
            self.vlm_att_encoder = self.vlm_att_encoder.to(device=device_type, dtype=weight_type)
            self.vlm_att_projector = self.vlm_att_projector.to(device=device_type, dtype=weight_type)
            self.vlm_att_key_projector = self.vlm_att_key_projector.to(device=device_type, dtype=weight_type)
            self.vlm_att_val_projector = self.vlm_att_val_projector.to(device=device_type, dtype=weight_type)

            self.vlm_att_query.data = self.vlm_att_query.data.to(device=device_type, dtype=weight_type)
            if "pretrain" in self.config.bert_type:
                self.vlm_att_ln = self.vlm_att_ln.to(device=device_type, dtype=weight_type)

            if self.vlm_att_bert_proj is not None:
                self.vlm_att_bert_proj = self.vlm_att_bert_proj.to(device=device_type, dtype=weight_type)

            self.clip_qformer = self.clip_qformer.to(device=device_type, dtype=weight_type)
            self.clip_qformer_projector = self.clip_qformer_projector.to(device=device_type, dtype=weight_type)
            self.clip_qformer_query.data = self.clip_qformer_query.data.to(device=device_type, dtype=weight_type)
            if "pretrain" in self.config.bert_type:
                self.clip_qformer_ln = self.clip_qformer_ln.to(device=device_type, dtype=weight_type)

            self.motion_encoder = self.motion_encoder.to(device=device_type, dtype=weight_type)

    def init_Qformer(self, vision_width, cross_attention_freq=2, truncation_side="right"):
        '''
        Initialize Q-former with text
        '''

        # initialize BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side=truncation_side)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})

        # initialize BERT
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq

        # Load BERT-based Q-former model with the modified config
        Qformer = BertLMHeadModelQF.from_pretrained(
            "bert-base-uncased", config=encoder_config
        )
        query_tokens = nn.Parameter(
            torch.zeros(1, self.config.num_query, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        # With text input
        Qformer.resize_token_embeddings(len(tokenizer))
        Qformer.cls = None

        return tokenizer, Qformer, query_tokens

    def init_motion_encoder(self, hidden_size, nhead=8, num_layers=4, dropout=0.1):
        motion_encoder = TransformerEncoder(
            hidden_size=hidden_size,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout
        )
        return motion_encoder


class RoboAnnotatorMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images, prompts=None, image_counts=None):
        image_features = self.get_model().get_vision_tower()(images)

        image_features = self.hierachical_encoder(image_features,
                                                  prompts=prompts,
                                                  image_counts=image_counts)

        return image_features

    def hierachical_encoder(self, image_features, prompts=None, image_counts=None):
        img_feat_lst = []
        if image_counts is None:
            assert len(image_features) == len(
                prompts), f"Size mismatch! image_features: {len(image_features)}, prompts: {len(prompts)}"
        else:
            assert len(prompts) == len(
                image_counts), f"Size mismatch! prompts: {len(prompts)}, image_counts: {len(image_counts)}"

        image_atts = torch.ones(image_features.size()[:-1], dtype=torch.long).to(image_features.device)

        total_count = 0
        for _idx in range(len(prompts)):
            if not image_counts or image_counts[_idx] == 1:  # image-only
                # 1. Visual embedding
                img_feat_prompt = image_features[_idx, None].expand(len(prompts[_idx]), -1,
                                                                    -1)  # shape: [prompt_num, image_shape, feat_dim]
                img_att_prompt = image_atts[_idx, None].expand(len(prompts[_idx]), -1)

                # remove cls embedding
                if self.config.mm_vision_select_feature == 'patch':
                    if img_feat_prompt.shape[1] % 2 == 1:
                        img_feat_prompt = img_feat_prompt[:, 1:]

                # 2. Input LLaMA
                final_token = self.token_generation(vis_embed=img_feat_prompt)

            else:  # video-only/video+image
                frame_list = [f"Frame {i}" for i in range(1, image_counts[_idx] + 1)]
                input_token = self.get_model().vlm_att_tokenlizer(
                    frame_list,
                    padding='longest',
                    truncation=True,
                    max_length=256,
                    return_tensors="pt"
                ).to(image_features.device)

                input_ids = input_token.input_ids
                attention_masks = input_token.attention_mask

                # 1. Visual embedding
                img_feat_prompt = image_features[total_count:total_count + image_counts[_idx]]
                img_feat_prompt = img_feat_prompt[None].expand(len(prompts[_idx]), -1, -1, -1)  # (prompt_num * frame_num, image_shape, hidden_size)
                img_att_prompt = image_atts[total_count:total_count + image_counts[_idx]]
                img_att_prompt = img_att_prompt[None].expand(len(prompts[_idx]), -1, -1).flatten(0, 1)

                input_ids = input_ids[None].expand(len(prompts[_idx]), -1, -1).flatten(0, 1)
                attention_masks = attention_masks[None].expand(len(prompts[_idx]), -1, -1).flatten(0, 1)

                # keyframe anchoring
                keyframe_index = self.keyframe_anchoring(total_frame_num=image_counts[_idx],
                                                         interval=self.config.interval)
                total_count += image_counts[_idx]

                if "pretrain" in self.config.bert_type and self.get_model().vlm_att_bert_proj is not None:
                    bert_feat = self.get_model().vlm_att_bert_proj(img_feat_prompt)
                else:
                    bert_feat = img_feat_prompt.clone()

                # remove cls embedding
                if self.config.mm_vision_select_feature == 'patch':
                    if img_feat_prompt.shape[1] % 2 == 1:
                        img_feat_prompt = img_feat_prompt[:, 1:]

                # Query generator
                query_tokens = self.get_model().vlm_att_query.expand(bert_feat.shape[0], -1,
                                                                     -1)  # (prompt_num * frame_num, num_query, att_feat)
                query_atts = torch.cat([torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(bert_feat.device),
                                        attention_masks], dim=1)
                if 'pretrain' in self.config.bert_type:
                    mm_img_in = self.get_model().vlm_att_ln(bert_feat)
                else:
                    mm_img_in = bert_feat

                query_output = self.get_model().vlm_att_encoder.bert(
                    input_ids,
                    query_embeds=query_tokens,
                    attention_mask=query_atts,
                    encoder_hidden_states=mm_img_in,
                    encoder_attention_mask=img_att_prompt,
                    return_dict=True,
                )
                query_output = query_output.last_hidden_state[:, :query_tokens.shape[1]]
                text_q = self.get_model().vlm_att_projector(query_output)  # Shape: (T, num_query, mm_hidden_size)

                if image_counts[_idx] == 1:  # image
                    scene_features = img_feat_prompt[keyframe_index]  # Shape: (N, image_shape, mm_hidden_size)
                    clip_features = None
                    motion_features = None
                else:  # video
                    # 1. Scene features
                    scene_features = img_feat_prompt[keyframe_index]  # Shape: (N, image_shape, mm_hidden_size)

                    # 2. Clip features
                    clip_features = []
                    prev_keyframe = 0
                    for i, curr_keyframe in enumerate(keyframe_index):
                        if i == 0:
                            continue
                        clip = text_q[prev_keyframe:curr_keyframe + 1]
                        clip_query_tokens = self.get_model().clip_qformer_query.expand(clip.shape[0], -1, -1)
                        clip_att = torch.ones(clip.shape[:-1], dtype=torch.long, device=clip.device)
                        if 'pretrain' in self.config.bert_type:
                            mm_clip_in = self.get_model().vlm_att_ln(clip)
                        else:
                            mm_clip_in = clip

                        clip_output = self.get_model().clip_qformer.bert(
                            query_embeds=clip_query_tokens,
                            encoder_hidden_states=mm_clip_in,
                            encoder_attention_mask=clip_att,
                            return_dict=True,
                        )
                        clip_feature = clip_output.last_hidden_state
                        clip_features.append(clip_feature.mean(0))
                        prev_keyframe = curr_keyframe
                    clip_features = torch.stack(clip_features)  # Shape: (N-1, num_query, mm_hidden_size)

                    # 3. Motion features
                    motion_features = text_q.mean(1, keepdim=True)  # Shape: (T, 1, mm_hidden_size)
                    motion_features = self.get_model().motion_encoder(motion_features)

                # 4. Input LLaMA
                final_token = self.token_generation(img_feat_prompt, scene_feature=scene_features,
                                                    clip_feature=clip_features, motion_feature=motion_features)
                final_token = final_token.reshape(len(prompts[_idx]), -1, final_token.shape[
                    -1])  # shape: [prompt_num, visual_token_number, hidden_feature_dim]

            img_feat_lst.append(final_token)
        return img_feat_lst

    def token_generation(self, vis_embed, scene_feature=None, clip_feature=None, motion_feature=None):
        '''
        vis_embed: (frame_num, image_shape, hidden_feature_dim)
        '''
        is_video = clip_feature is not None and scene_feature is not None

        if is_video:
            # Key part 1 : Calculate Scene Feature
            if self.config.compress_type is None:
                self.config.compress_type = 'full'
            elif 'grid' in self.config.compress_type:
                grid_size = int(self.config.compress_type.split('grid:')[-1])
                cur_shape = int(scene_feature.shape[1] ** 0.5)
                scene_feature = scene_feature.reshape(scene_feature.shape[0], cur_shape, cur_shape, -1)
                grid_stride = cur_shape // grid_size
                scene_feature = F.avg_pool2d(scene_feature.permute(0, 3, 1, 2),
                                             padding=0,
                                             kernel_size=grid_stride,
                                             stride=grid_stride)
                scene_feature = scene_feature.permute(0, 2, 3, 1).flatten(1, 2)  # Shape(frame_num, 4, hidden_size)
            scene_feature = self.get_model().mm_projector(scene_feature)
            scene_feature = einops.rearrange(scene_feature, 'f n h -> (f n) h')

            # Key part 2 : Calculate Clip Feature
            clip_feature = self.get_model().clip_qformer_projector(clip_feature)
            clip_feature = einops.rearrange(clip_feature, 'c n h -> (c n) h')

            # Key part 3 : Calculate Motion Feature
            motion_feature = self.get_model().vlm_att_val_projector(motion_feature)
            motion_feature = einops.rearrange(motion_feature, 't n h -> (t n) h')

            # concat token in shape (t+32*(n-1)+size*n, C)
            final_token = torch.cat([scene_feature, clip_feature, motion_feature], dim=0)
        else:
            # key part 1 : Scene feature
            if self.config.compress_type is None:
                self.config.compress_type = 'full'
            else:
                grid_size = int(self.config.compress_type.split('grid:')[-1])
                cur_shape = int(vis_embed.shape[1] ** 0.5)
                vis_embed = vis_embed.reshape(vis_embed.shape[0], cur_shape, cur_shape, -1)
                grid_stride = cur_shape // grid_size
                vis_embed = F.avg_pool2d(vis_embed.permute(0, 3, 1, 2),
                                         padding=0,
                                         kernel_size=grid_stride,
                                         stride=grid_stride)
                vis_embed = vis_embed.permute(0, 2, 3, 1).flatten(1, 2)

                # concat token in shape (n+1, C)
            vis_embed = self.get_model().mm_projector(vis_embed)  # Shape(1, 4, hidden_feature_dim)
            final_token = vis_embed

        return final_token

    def keyframe_anchoring(self, total_frame_num, interval=30):
        keyframe_index = [0]
        current_frame = interval
        while current_frame < (total_frame_num - 1):
            if (current_frame + interval) <= (total_frame_num - 1):
                keyframe_index.append(current_frame)
                current_frame += interval
            else:
                break
        if keyframe_index[-1] != (total_frame_num - 1):
            keyframe_index.append(total_frame_num - 1)

        return keyframe_index

    def update_prompt(self, prompts=None):
        self.prompts = prompts

    def prepare_inputs_labels_for_multimodal(
            self, input_ids, attention_mask, past_key_values, labels, images, prompts=None
    ):
        if prompts is None and hasattr(self, 'prompts'):
            prompts = self.prompts

        # Visual Embedding
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[
                1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                                            dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            images = [image if len(image.shape) == 4 else image.unsqueeze(0) for image in images]
            image_counts = [image.shape[0] for image in images]
            concat_images = torch.cat(images, dim=0) # shape: (total_num_frames, C, H, W)
            image_features = self.encode_images(concat_images, prompts, image_counts)
        else:  # image-only
            image_features = self.encode_images(images, prompts)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                half_len = cur_input_ids.shape[0] // 2
                if isinstance(image_features, list):
                    cur_image_features = image_features[cur_image_idx][0]
                else:
                    cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape

            token_idx = 0
            while image_token_indices.numel() > 0:
                if isinstance(image_features, list):
                    cur_image_features = image_features[cur_image_idx][token_idx]
                else:
                    cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]

                # If using <im_start> and <im_end> tokens
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end',
                                                                                  False):
                    # Add embeddings before <im_start>
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[:image_token_start - 1]).detach())
                    # Add <im_start> token embedding
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[image_token_start - 1:image_token_start]))
                    # Add image features
                    cur_new_input_embeds.append(cur_image_features)
                    # Add <im_end> token embedding
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[image_token_start + 1:image_token_start + 2]))
                    if labels is not None:
                        # Copy labels before <im_start>
                        cur_new_labels.append(cur_labels[:image_token_start])
                        # Ignore labels for image feature tokens
                        cur_new_labels.append(
                            torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device,
                                       dtype=labels.dtype))
                        # Copy label for <im_end>
                        cur_new_labels.append(cur_labels[image_token_start:image_token_start + 1])
                        # Trim used labels
                        cur_labels = cur_labels[image_token_start + 2:]

                # If not using <im_start> and <im_end>
                else:
                    # Add embeddings before <image>
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                    # Add image features
                    cur_new_input_embeds.append(cur_image_features)
                    if labels is not None:
                        # Copy labels before <image>
                        cur_new_labels.append(cur_labels[:image_token_start])
                        # Ignore labels for image feature tokens
                        cur_new_labels.append(
                            torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device,
                                       dtype=labels.dtype))
                        # Trim used labels
                        cur_labels = cur_labels[image_token_start + 1:]

                # Trim used input ids
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end',
                                                                                  False):
                    cur_input_ids = cur_input_ids[image_token_start + 2:]
                else:
                    cur_input_ids = cur_input_ids[image_token_start + 1:]
                # Recompute image token positions
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
                token_idx += 1

            # change image idx after processing one sample
            cur_image_idx += 1

            # If there are remaining tokens after all image tokens are processed
            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end',
                                                                                  False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)

            # Move all embeddings to the correct device and concatenate
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        # Check if all new input embeddings have the same shape
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            # Pad all embeddings to the maximum length with zeros
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed,
                                           torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]),
                                                       dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                # Pad labels to max length with IGNORE_INDEX
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label,
                                               torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX,
                                                          dtype=cur_new_label.dtype, device=cur_new_label.device)),
                                              dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            # only used for right padding in tokenlizer
            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels,
                                                                                    new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True,
                                                        dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],),
                                                         False, dtype=attention_mask.dtype,
                                                         device=attention_mask.device)
                    cur_new_attention_mask = torch.cat(
                        (new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            # only used for right padding in tokenlizer
            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full(
                    (attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True,
                    dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            # If tuning the multi-modal adapter, only make input embeddings trainable
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            # If using a pretrained multi-modal adapter, load the pretrained weights
            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
