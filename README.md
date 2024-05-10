# Chinese-LLaVA-Med

## Benchmark

|                                        Method                                        | llava-med-zh-eval Qwen Score |
|:------------------------------------------------------------------------------------:| :--------------------------: |
|                                  GPT4 Ground Truth                                   |            68.26             |
|           [LLaVA-1.5-7B](https://huggingface.co/llava-hf/llava-1.5-7b-hf)            |            53.13             |
| [Chinese-LLaVA-Med-1.5-7B](https://huggingface.co/BUAADreamer/Chinese-LLaVA-1.5-Med) |          **58.78**           |

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

# finetuning
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train config/llava1_5_lora_sft.yaml

# export
# modify your own export_hub_model_id and hf_hub_token in the config/llava1_5_lora_sft_export.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli export config/llava1_5_lora_sft_export.yaml
```

## Evaluation

```shell
# generate output results
python3 evaluation/generate_eval_content.py --model_name_or_path models/llava1_5-7b-med

# eval by qwen-1.5-14b-chat
python3 evaluation/eval_qwen_score.py --input_path outputs/llava_med_zh_eval_llava1_5-7b-med.json
```

## Inference

```shell
# with final model
llamafactory-cli webchat config/llava1_5_lora_sft_infer.yaml

# with lora
llamafactory-cli webchat config/llava1_5_lora_sft_infer_lora.yaml
```

