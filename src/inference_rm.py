# coding=utf-8
# Implements several parameter-efficient supervised fine-tuning method for ChatGLM.
# This code is inspired by https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py
import json
from utils import DataCollatorForChatGLM
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModel
from peft import PeftModel
from utils import prepare_args
from utils.other import AverageMeter
import random
from transformers import DataCollatorWithPadding, BatchEncoding
from trl import AutoModelForCausalLMWithValueHead


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


def get_position_ids(model, tokenizer, input_ids: torch.Tensor) -> torch.Tensor:
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


def main():
    # Prepare pretrained model and dataset
    model_args, data_args, training_args, finetuning_args = prepare_args(stage='sft')
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
    print(model_args.checkpoint_dir)
    model = PeftModel.from_pretrained(model, model_args.checkpoint_dir[0], is_trainable=True)

    model2 = AutoModelForCausalLMWithValueHead.from_pretrained(model)
    model2.cuda()
    model2.eval()
    filename = r'output/prediction_sft.json'
    target_filename = r'output/rm_score.json'
    target_w = open(target_filename, 'a+', encoding='utf-8')
    samples = []
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            data = json.loads(line)
            if len(data['history']) > 0:
                continue
            samples.append(data)
    samples = random.sample(samples, 200)
    for i, line in enumerate(tqdm(samples)):
        with torch.no_grad():
            prefix = ''
            input_text = ''
            answer = ''
            if line["instruction"] and line["output"][-1]:
                query, answer = line["instruction"], line["output"][-1]
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

            instruction = line["instruction"]
            queries = [torch.tensor(tokenizer.encode(instruction)).cuda()]
            responses = [torch.tensor(tokenizer.encode(answer)).cuda()]
            input_ids = [torch.cat([q, r]) for q, r in zip(queries, responses)]
            input_ids = [feature.clone().detach().flip(0) for feature in input_ids]
            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True,
                                                        padding_value=tokenizer.pad_token_id).flip(-1)

            attention_mask = get_attention_masks(tokenizer, input_ids)
            position_ids = get_position_ids(model, tokenizer, input_ids)
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            position_ids = position_ids.cuda()
            model_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids}
            model_inputs = BatchEncoding(model_inputs)

            _, _, values = model2(**model_inputs)
            rewards = [reward for reward in values[-1].to(torch.float32)]
            reward_meter = AverageMeter()
            reward_meter.update(torch.stack(rewards).mean().item(), n=len(rewards))
            score = reward_meter.avg
            print(score)
            print('instruction: ', instruction)
            print('answer: ', answer)
            print('score: ', score)
            tmp = {'instruction': instruction, 'answer': answer, 'score': score}
            target_w.write(json.dumps(tmp, ensure_ascii=False) + '\n')
            # print(rewards)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
