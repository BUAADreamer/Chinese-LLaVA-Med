import fire
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq
import evaluate
from rouge import Rouge

rouge = Rouge()

em_tool = evaluate.load('evaluation/exact_match.py')


def test_one(example, model, processor):
    messages = example['messages']
    messages[0]['content'] = '<image>' + messages[0]['content']
    processor.tokenizer.chat_template = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. {% for message in messages %}{% if message['role'] == 'user' %}USER: {{ message['content'] }} ASSISTANT: {% else %}{{ message['content'] }}{% endif %} {% if message['role'] == 'user' %} {% else %}{{eos_token}}{% endif %}{% endfor %}"""
    prompt = processor.tokenizer.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=False)
    label = example['messages'][1]['content']
    image = example['images'][0]
    inputs = processor(prompt, image, return_tensors='pt').to(0, torch.float16)
    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    output = processor.decode(outputs[0][2:], skip_special_tokens=True)[len(prompt):]
    scores = rouge.get_scores(output, label)[0]
    res = {
        "em": em_tool.compute(references=[label], predictions=[output])['exact_match'],
        "rouge-1": scores['rouge-1']['f'],  # f p r
        "rouge-2": scores['rouge-2']['f'],
        "rouge-l": scores['rouge-l']['f'],
    }
    return res


def main(
        model_name_or_path: str = "llava-hf/llava-1.5-7b-hf",
        data_name_or_path: str = "BUAADreamer/llava-med-zh-eval"
):
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cuda",
    )
    dataset = load_dataset(data_name_or_path)['test']
    N = len(dataset)
    score_dict = {
        "em": 0,
        "rouge-1": 0,
        "rouge-2": 0,
        "rouge-l": 0,
    }
    for i in tqdm(range(N)):
        try:
            example = dataset[i]
            per_score_dict = test_one(example, model, processor)
            for key in per_score_dict:
                score_dict[key] += per_score_dict[key]
        except Exception as e:
            print(e)
            print(f"{i}th data has problem")
    for key in score_dict:
        score_dict[key] /= N
        print(f"{key}:{score_dict[key]}")


if __name__ == '__main__':
    fire.Fire(main)
