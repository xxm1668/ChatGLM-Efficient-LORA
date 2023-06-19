# coding=utf-8
# Implements several parameter-efficient supervised fine-tuning method for ChatGLM.
# This code is inspired by https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py
import json
import random
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModel
from peft import PeftModel
from utils import prepare_args


def get_attention_masks(tokenizer, input_ids: torch.Tensor) -> torch.Tensor:
    r"""
    Generates attention masks for left-padded sequences.

    Note that ChatGLM assigns False on token to be attended in attention mask. In general settings, it should be True.

    According to: https://huggingface.co/THUDM/chatglm-6b/blob/v1.1.0/modeling_chatglm.py#L680
    """
    batch_size, seq_length = input_ids.size()
    attention_mask = torch.ones((batch_size, seq_length, seq_length))
    attention_mask.tril_()
    for i, seq in enumerate(input_ids):
        attention_mask[i, :, :(seq == tokenizer.bos_token_id).nonzero()[0].item()] = 1  # context
        attention_mask[i, :, :(seq != tokenizer.pad_token_id).nonzero()[0].item()] = 0  # padding
    attention_mask.unsqueeze_(1)
    attention_mask = (attention_mask < 0.5).bool()
    return attention_mask


def get_position_ids(model, tokenizer, input_ids: torch.Tensor):
    r"""
    Generates position ids for left-padded sequenes.

    According to: https://huggingface.co/THUDM/chatglm-6b/blob/v1.1.0/modeling_chatglm.py#L692
    """
    batch_size, seq_length = input_ids.size()
    mask: int = model.config.mask_token_id
    gmask: int = model.config.gmask_token_id
    position_ids = torch.zeros((batch_size, seq_length), dtype=torch.long)
    block_position_ids = torch.zeros((batch_size, seq_length), dtype=torch.long)
    for i, seq in enumerate(input_ids):
        mask_token = gmask if gmask in seq else mask
        context_length = (seq == tokenizer.bos_token_id).nonzero()[0].item()
        padding_length = (seq != tokenizer.pad_token_id).nonzero()[0].item()
        position_ids[i, padding_length:] = torch.arange(seq_length - padding_length, dtype=torch.long)
        if model.position_encoding_2d or (mask_token != gmask):  # 2d position encoding or not gMASK
            position_ids[i, context_length:] = (seq == mask_token).nonzero()[0].item() - padding_length  # mask position
        block_position_ids[i, context_length:] = torch.arange(seq_length - context_length, dtype=torch.long) + 1
    if model.position_encoding_2d:
        position_ids = torch.stack((position_ids, block_position_ids), dim=1)
    return position_ids


# 创建自定义回调函数，用于集成wandb

def main():
    # Prepare pretrained model and dataset
    model_args, data_args, training_args, finetuning_args = prepare_args()
    config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        padding_side="left",
        **config_kwargs
    )

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        **config_kwargs
    )
    model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, **config_kwargs)
    model.half()
    model = PeftModel.from_pretrained(model,
                                      "/home/house365ai/xxm/ChatGLM-Efficient-Tuning/output/lora_estate_qa8/global_step-17550")
    model.cuda()
    model.eval()
    filename = r'data/estate_qa.json'
    target_filename = r'output/prediction.json'
    target_w = open(target_filename, 'a+', encoding='utf-8')
    samples = []
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            data = json.loads(line)
            samples.append(data)
    samples = random.sample(samples, 500)
    for i, line in enumerate(tqdm(samples)):
        with torch.no_grad():
            prefix = ''
            input_text = ''
            answer = ''
            if line["instruction"] and line["output"]:
                query, answer = line["instruction"], line["output"]
                if line["input"]:
                    query += line["input"]
                if line["history"]:
                    prompt = ""
                    history = line["history"]
                    for j, (old_query, response) in enumerate(history):
                        prompt += "[Round {}]\n问：{}\n答：{}\n".format(j, old_query, response)
                    prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
                else:
                    prompt = query
                prompt = prefix + prompt
                input_text = prompt
            input_ids = tokenizer.encode(text=input_text, add_special_tokens=False)
            input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
            input_ids = [torch.tensor(input_ids).flip(0)]
            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True,
                                                        padding_value=tokenizer.pad_token_id).flip(-1)
            attention_mask = get_attention_masks(tokenizer, input_ids)
            position_ids = get_position_ids(model, tokenizer, input_ids)
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            position_ids = position_ids.cuda()

            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                max_length=256,
                do_sample=False,
                temperature=0.1,
                top_p=0.1)
            out_text = tokenizer.decode(out[0])
            tmp = {}
            tmp['question'] = input_text
            tmp['ori_answer'] = answer
            tmp['pre_answer'] = out_text.split(' ')[1]
            target_w.write(json.dumps(tmp, ensure_ascii=False) + '\n')


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
