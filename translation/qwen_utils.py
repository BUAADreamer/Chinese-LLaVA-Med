from tqdm import tqdm
import torch
torch.backends.cuda.enable_mem_efficient_sdp(False)
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"  # the device to load the model onto
model_path = 'qwen1_5/qwen1_5-14b-chat'
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
prompt = "Please translate the following English text to Chinese: {}"
system_prompt = "You are a helpful assistant."


def test_translate():
    message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt.format("The image is a CT scan of the abdomen and pelvis, focusing on the adrenal glands.")}
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

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print(response)


def translate_messages(messages):
    for raw_message in tqdm(messages):
        content = raw_message['content']
        if '<image>' in content:
            content = content.replace("<image>", "")
            content = content.replace("\n", "")
        message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt.format(content)}
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
        raw_message['content'] = response
    return messages


def batch_translate_messages(messages):
    texts = []
    for raw_message in tqdm(messages):
        content = raw_message['content']
        if '<image>' in content:
            content = content.replace("<image>", "")
            content = content.replace("\n", "")
        message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt.format(content)}
        ]
        text = tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )
        texts.append(text)
    bs = 8
    N = len(texts)
    res_list = []
    for i in tqdm(range(0, N, bs)):
        max_idx = min(N, i + bs)
        cur_texts = texts[i:max_idx]
        model_inputs = tokenizer(cur_texts, return_tensors="pt", padding=True).to(device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for res in response:
            res_list.append(res)
    for message, res in zip(messages, res_list):
        message['content'] = res
    return messages


if __name__ == '__main__':
    test_translate()
