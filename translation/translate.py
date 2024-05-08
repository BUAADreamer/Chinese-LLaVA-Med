import json

import fire

"""
mkdir logs

CUDA_VISIBLE_DEVICES=0 python3 translate.py --input_path llava-med/llava_med_instruct_60k_inline_mention.json \
--output_path llava-med-zh/llava_med_zh_instruct_60k_inline_mention.json \
--idx idx >logs/translate_idx.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 python3 translate.py --input_path llava-med/llava_med_eval_qa50_qa.jsonl \
--output_path llava-med-zh/llava_med_zh_eval_qa50_qa.jsonl

CUDA_VISIBLE_DEVICES=0 python3 translate.py \
--output_path llava-med-zh/llava_med_zh_instruct_60k_inline_mention.json \
--do_merge
"""


def translate(data_ls):
    from qwen_utils import translate_messages
    messages = []
    for i, data in enumerate(data_ls):
        for j, conversation in enumerate(data['conversations']):
            messages.append(
                {
                    "role": "user" if conversation['from'] == 'human' else 'assistant',
                    "content": conversation['value'],
                    "idx_1": i,
                    "idx_2": j,
                }
            )
    messages = translate_messages(messages)
    for message in messages:
        i = message["idx_1"]
        j = message["idx_2"]
        data_ls[i]['conversations'][j]['value'] = message['content']
    return data_ls


def translate_eval(data_ls):
    from qwen_utils import translate_messages
    messages = []
    for i, data in enumerate(data_ls):
        messages.append(
            {
                "role": "user",
                "content": data['text'],
                "key": "text",
                "idx": i,
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": data["gpt4_answer"],
                "key": "gpt4_answer",
                "idx": i,
            }
        )
    messages = translate_messages(messages)
    for message in messages:
        idx = message['idx']
        key = message['key']
        data_ls[idx][key] = message['content']
    return data_ls


def main(
        input_path: str = '',
        output_path: str = '',
        idx: int = -1,
        num: int = 10,
        do_merge: bool = False
):
    input_file_list = input_path.split(",")
    output_file_list = output_path.split(",")
    if do_merge:
        for output_file in output_file_list:
            data_ls = []
            for i in range(num):
                with open(output_file.replace(".json", f"_{i}.json")) as f:
                    data_ls.extend(json.loads(f.read()))
            with open(output_file, 'w') as f:
                f.write(json.dumps(data_ls, ensure_ascii=False, indent=4))
    else:
        for input_file, output_file in zip(input_file_list, output_file_list):
            print(f"translate from {input_file} to {output_file}")
            if input_file.endswith("json"):
                with open(input_file) as f:
                    raw_data_ls = json.loads(f.read())
                    if idx >= 0:
                        N = len(raw_data_ls)
                        bs = N // num + 1
                        max_idx = min(idx * bs + bs, N)
                        raw_data_ls = raw_data_ls[idx * bs:max_idx]
                        output_file = output_file.replace(".json", f"_{idx}.json")
                    data_ls = translate(raw_data_ls)
            elif input_file.endswith("jsonl"):
                with open(input_file) as f:
                    raw_data_ls = []
                    for line in f:
                        raw_data_ls.append(json.loads(line))
                    data_ls = translate_eval(raw_data_ls)
            with open(output_file, 'w') as f:
                f.write(json.dumps(data_ls, ensure_ascii=False, indent=4))


if __name__ == '__main__':
    fire.Fire(main)
