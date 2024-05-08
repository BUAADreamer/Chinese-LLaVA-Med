# Chinese-LLaVA-Med

## Training

### Install [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)

```shell
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers==0.0.25
pip install .[bitsandbytes]
```

### Finetuning with [llava-med-zh-instruct-60k](https://huggingface.co/datasets/BUAADreamer/llava-med-zh-instruct-60k)

```shell
git clone https://github.com/BUAADreamer/Chinese-LLaVA-Med.git
cd Chinese-LLaVA-Med
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train config/llava1_5_lora_sft.yaml
```

## Evaluation

```shell
python3 evaluation/test.py
```

