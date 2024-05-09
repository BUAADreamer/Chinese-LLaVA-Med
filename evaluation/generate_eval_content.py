import json
import os

import fire
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq, LlavaProcessor


def test_one(example, model, processor: LlavaProcessor):
    messages = [
        {'role': 'user', 'content': '<image>' + example['question']}
    ]
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    image = example['images'][0]
    inputs = processor(prompt, image, return_tensors='pt').to(0, torch.float16)
    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    output = processor.decode(outputs[0], skip_special_tokens=True)
    output = output[len(prompt) - 7:]
    return output


def main(
        model_name_or_path: str = "llava-hf/llava-1.5-7b-hf",
        data_name_or_path: str = "BUAADreamer/llava-med-zh-eval",
        output_path: str = "outputs",
):
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    if processor.tokenizer.chat_template is None:
        processor.tokenizer.chat_template = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. {% for message in messages %}{% if message['role'] == 'user' %}USER: {{ message['content'] }} ASSISTANT: {% else %}{{ message['content'] }}{% endif %} {% if message['role'] == 'user' %} {% else %}{{eos_token}}{% endif %}{% endfor %}"""
    model = AutoModelForVision2Seq.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cuda",
    )
    dataset = load_dataset(data_name_or_path)['test']
    N = len(dataset)
    output_ls = []
    for i in tqdm(range(N)):
        try:
            example = dataset[i]
            output = test_one(example, model, processor)
            example['output'] = output
            del example['images']
            output_ls.append(example)
        except Exception as e:
            print(e)
            print(f"{i}th data has problem")
    os.makedirs(output_path, exist_ok=True)
    model_name = model_name_or_path.split("/")[-1]
    output_path = os.path.join(output_path, f"llava_med_zh_eval_{model_name}.json")
    print(output_path)
    with open(output_path, 'w') as f:
        f.write(json.dumps(output_ls, ensure_ascii=False, indent=4))


if __name__ == '__main__':
    fire.Fire(main)
