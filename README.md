# RoboAnnotatorX: A Comprehensive and Universal Annotation Framework for Accurate Understanding of Long-horizon Robot Demonstration

<a href='https://roboannotatex.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://arxiv.org/abs/2311.17043'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
<a href='https://huggingface.co/koulx/roboannotatorx'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue'></a>
<a href='https://huggingface.co/datasets/koulx/RoboX-VQA'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Data-green'></a>

A reliable annotation tool that enhances multimodal large language model to generate high-quality, 
context-rich annotations for complex long-horizon robotics demonstrations.

## 🚀 News

- [25/07/01] 🔥 Our work has been accepted to ICCV 2025!)
- [25/05/09] 🔥 RoboannotatorX is comming!

## 📅 TODO

- [ ] ⭐ Release scripts for model training and inference.
- [ ] ⭐ Release evaluation scripts for Benchmarks.

## 🛠️ Setup
Please follow the instructions below to install the required packages.
1. Clone this repository
```bash
git clone https://github.com/LongXinKou/RoboannotatorX.git
```

2. Install Package
```bash
conda create -n roboannotatorx python=3.10 -y
conda activate roboannotatorx
cd RoboannotatorX
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases
```bash
pip install ninja
pip install flash-attn --no-build-isolation

# [Option] install the per-commit wheel built by that PR, "https://github.com/Dao-AILab/flash-attention/releases"
pip install flash_attn-2.6.3+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

## 🎯 Training

### Preparation
We organize the data in the format of LLaVA, the folder structure should be organized as follows before training.

```
data/
├── Pretrain/
│   ├── images
│   ├── bc_z
│   ├── droid
│   ├── ...
│   ├── blip_laion_cc_sbu_558k.json
│   └── mixing_pretrain_510k.json
├── Finetune/
│   ├── images
│   ├── bc_z
│   ├── droid
│   ├── ...
│   ├── complex_reasoning_77k.json
    ├── llava_instruct_150k.json
│   └── mixing_fintune_stage2_886k.json
│   └── mixing_fintune_stage3_86k.json
```

#### Pretrain

We first establish fundamental visual-language alignments through captioning-based pretraining.

- For image-based dataset, we use 558K image-caption pairs from [LLaVA-filtered CC3M](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain).
- For video-based dataset, we use 510K video-caption pairs from [RoboX-VQA-Pretrain](https://huggingface.co/datasets/koulx/RoboX-VQA-Pretraining).

#### Finetune-Stage2

Based on general visual understanding foundation, we conduct short-horizon instruction fine-tuning.

- For image-based dataset, we use 227K image QA pairs(complex_reasoning_77k + llava_instruct_150k) from [LLaVA-Instruct](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K).
- For video-based dataset, we use 886K video QA pairs from [RoboX-VQA-Stage2](https://huggingface.co/datasets/koulx/RoboX-VQA-Stage2).

#### Finetune-Stage3

We conduct long-horizon instruction fine-tuning with complex robotic demonstrations.

- For video-based dataset, we use 86K video QA pairs from [RoboX-VQA-Stage3](https://huggingface.co/datasets/koulx/RoboX-VQA-Stage3).

### Training Script

Please make sure you download and organize the data following Preparation before training. 
If you are interested in training the model, you can run the following command.

#### Pretraining

```bash
# 7B model
bash scripts/train/pretrain_7b.sh

# 13B model
bash scripts/train/pretrain_13b.sh
```

#### Finetune-Stage2

```bash
# 7B model
bash scripts/train/finetune_stage2_7b.sh

# 13B model
bash scripts/train/finetune_stage2_13b.sh
```

#### Finetune-Stage3

```bash
# 7B model
bash scripts/train/finetune_stage3_7b.sh
# 13B model
bash scripts/train/finetune_stage3_13b.sh
```

## 📊 Inference
To run inference with the trained model, you can use [`embodied_eval`](https://github.com/LongXinKou/embodied-eval),
and the command is as follows:

```bash
bash example/vqa/roboannoatorx.sh
```

or you can just run `pred.py` directly to generate predictions.


## 🎁 Acknowledgement
We would like to thank the following repos for their great work:

- This work is built upon the [LLaMA-VID](https://github.com/dvlab-research/LLaMA-VID).
- This work utilize pre-collected training dataset from [LLaVA](https://github.com/haotian-liu/LLaVA).
- This work utilizes LLM from [Vicuna](https://github.com/lm-sys/FastChat)
- This work utilizes pretrained weights from [InstructBLIP](https://github.com/salesforce/LAVIS).

