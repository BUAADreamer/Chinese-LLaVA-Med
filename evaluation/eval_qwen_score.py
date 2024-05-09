import json

import fire
from tqdm import tqdm
import torch

torch.backends.cuda.enable_mem_efficient_sdp(False)
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"  # the device to load the model onto
model_path = 'Qwen/Qwen1.5-14B-Chat'
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
system_prompt = """We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image. For your reference, the visual content in the image is represented with caption describing the same image.
  Please rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.
  Please first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."""
role = 'Assistant'
tokenizer = AutoTokenizer.from_pretrained(model_path)


def conv_to_str(context, output):
    return (context + f'[{role} 2]\n{output}\n\n[End of {role} 2]\n\n')


def main(input_path):
    with open(input_path) as f:
        data_ls = json.loads(f.read())
    score = 0
    abs_score = 0
    N = len(data_ls)
    for i, data in enumerate(tqdm(data_ls)):
        for _ in range(5):
            try:
                message = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": conv_to_str(data['context'], data['output'])}
                ]
                text = tokenizer.apply_chat_template(
                    message,
                    tokenize=False,
                    add_generation_prompt=True
                )
                model_inputs = tokenizer([text], return_tensors="pt").to(device)

                generated_ids = model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=512
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]

                response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                score_str = response.split("\n")[0].strip()
                score_gpt4, score_model = map(int, score_str.split())
                score += score_model / score_gpt4
                abs_score += score_model
                break
            except Exception as e:
                print(response)
    score /= N
    abs_score /= N
    print("average relative score:", score)
    print("average absolute score:", abs_score)


if __name__ == '__main__':
    fire.Fire(main)
