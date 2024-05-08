import fire
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq
import evaluate

em_tool = evaluate.load('exact_match.py')
bleu_tool = evaluate.load('bleu')


def test_one(example, model, processor):
    messages = example['messages']
    messages[0]['content'] = '<image>' + messages[0]['content']
    prompt = processor.tokenizer.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=False)
    label = example['messages'][1]['content']
    image = example['images'][0]
    inputs = processor(prompt, image, return_tensors='pt').to(0, torch.float16)
    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    output = processor.decode(outputs[0][2:], skip_special_tokens=True)
    res = {
        "em": em_tool.compute(references=[label], predictions=[output]),
        "bleu": bleu_tool.compute(references=[label], predictions=[output])
    }
    return res


def main(
        model_name_or_path: str,
        data_name_or_path: str
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
        "bleu": 0,
        "em": 0,
    }
    for i in tqdm(range(N)):
        example = dataset[i]
        per_score_dict = test_one(example, model, processor)
        for key in per_score_dict:
            score_dict[key] += per_score_dict[key]
    for key in score_dict:
        score_dict[key] /= N
        print(f"{key}:{score_dict[key]}")


if __name__ == '__main__':
    fire.Fire(main)
