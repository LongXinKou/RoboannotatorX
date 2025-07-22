# RoboAnnotatorX: A Comprehensive and Universal Annotation Framework for Accurate Understanding of Long-horizon Robot Demonstration

<a href='https://roboannotatex.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://arxiv.org/abs/2311.17043'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
<a href='https://huggingface.co/koulx/roboannotatorx'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue'></a>
<a href='https://huggingface.co/datasets/koulx/RoboX-VQA'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Data-green'></a>

A reliable annotation tool that enhances multimodal large language model to generate high-quality, context-rich annotations for complex long-horizon demonstrations.

## 🚀 News

[//]: # (- [24/07/04] 🔥 Our work has been accepted to ECCV 2024!)
[//]: # (- [23/12/05] 🔥 We release the full training and evalution [model]&#40;https://huggingface.co/YanweiLi/llama-vid-7b-full-224-long-video&#41;, [data]&#40;https://huggingface.co/datasets/YanweiLi/LLaMA-VID-Data&#41;, and scripts to support movie chating! )
[//]: # (- [25/05/09] 🔥 LLaMA-VID is comming! We release the [paper]&#40;https://arxiv.org/abs/2311.17043&#41;, [code]&#40;https://github.com/dvlab-research/LLaMA-VID&#41;, [data]&#40;https://huggingface.co/datasets/YanweiLi/LLaMA-VID-Data&#41;, [models]&#40;https://huggingface.co/YanweiLi&#41;, and [demo]&#40;https://llama-vid.github.io/&#41; for LLaMA-VID!)
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
```

## 🎯 Training

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

### Pretrain

We first establish fundamental visual-language alignments through captioning-based pretraining.

- For image-based dataset, we use 558K image-caption pairs from [LLaVA-filtered CC3M](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain).
- For video-based dataset, we use 510K video-caption pairs from [RoboX-VQA-Pretrain](https://huggingface.co/datasets/koulx/RoboX-VQA-Pretraining).

### Finetune-Stage2

Based on general visual understanding foundation, we conduct short-horizon instruction fine-tuning.

- For image-based dataset, we use 227K image QA pairs(complex_reasoning_77k + llava_instruct_150k) from [LLaVA-Instruct](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K).
- For video-based dataset, we use 886K video QA pairs from [RoboX-VQA-Stage2](https://huggingface.co/datasets/koulx/RoboX-VQA-Stage2).

### Finetune-Stage3s.

We conduct long-horizon instruction fine-tuning with complex robotic demonstrations.

- For video-based dataset, we use 86K video QA pairs from [RoboX-VQA-Stage3](https://huggingface.co/datasets/koulx/RoboX-VQA-Stage3).

