# coding=utf-8
# Implements several parameter-efficient supervised fine-tuning method for ChatGLM.
# This code is inspired by https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py
import wandb
from utils import (
    load_pretrained,
    prepare_args,
    prepare_data,
    preprocess_data,
    DataCollatorForChatGLM,
    Seq2SeqTrainerForChatGLM,
    ComputeMetrics,
    get_logits_processor,
    plot_loss
)
from transformers.trainer_callback import TrainerCallback
import deepspeed
import os
import torch
from shutil import copy
from torch.utils.data import RandomSampler, DataLoader


# 创建自定义回调函数，用于集成wandb
class WandbCallback(TrainerCallback):
    def __init__(self):
        super().__init__()

    def on_train_begin(self, args, state, control, **kwargs):
        # 在训练开始时进行初始化
        wandb.log({'epoch': 0, 'loss': 0, 'accuracy': 0})

    def on_epoch_end(self, args, state, control, **kwargs):
        # 在每个epoch结束时记录指标
        wandb.log(
            {'epoch': state.epoch + 1, 'loss': state.log_history[-1]['loss'], 'learning_rate': args.learning_rate})


# 创建自定义回调函数，用于集成wandb

def main():
    # Prepare pretrained model and dataset
    model_args, data_args, training_args, finetuning_args = prepare_args()
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="ChatGLM-Efficient-Tuning",
        # track hyperparameters and run metadata
        config={
            "learning_rate": training_args.learning_rate,
            "batch_size": training_args.per_device_train_batch_size,
            "epochs": training_args.num_train_epochs,
        }
    )
    dataset = prepare_data(model_args, data_args)
    model, tokenizer = load_pretrained(model_args, training_args, finetuning_args, training_args.do_train, stage="sft")
    dataset = preprocess_data(dataset, tokenizer, data_args, training_args, stage="sft")
    data_collator = DataCollatorForChatGLM(tokenizer, model, data_args.ignore_pad_token_for_loss)
    train_dataloader = DataLoader(dataset,
                                  batch_size=training_args.per_device_train_batch_size,
                                  sampler=RandomSampler(dataset),
                                  collate_fn=data_collator,
                                  drop_last=True,
                                  num_workers=0)
    model_engine, optimizer, _, _ = deepspeed.initialize(config='src/conf/mydeepspeed.json',
                                                         model=model,
                                                         model_parameters=model.parameters())
    model_engine.train()
    global_step = 0
    for i_epoch in range(int(training_args.num_train_epochs)):
        train_iter = iter(train_dataloader)
        for step, batch in enumerate(train_iter):
            input_ids = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            position_ids = batch["position_ids"].cuda()
            outputs = model_engine.forward(input_ids=input_ids, attention_mask=attention_mask,
                                           position_ids=position_ids, labels=labels)
            loss = outputs[0]
            if training_args.gradient_accumulation_steps > 1:
                loss = loss / training_args.gradient_accumulation_steps
            model_engine.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if (step + 1) % training_args.gradient_accumulation_steps == 0:
                model_engine.step()
                global_step += 1
            if global_step % training_args.logging_steps == 0:
                print("loss:{}, global_step:{}".format(float(loss.item()), global_step))
        save_dir = os.path.join(training_args.output_dir, f"global_step-{global_step}")
        model_engine.save_pretrained(save_dir)
        copy(os.path.join(model_args.model_name_or_path, "tokenizer_config.json"),
             os.path.join(save_dir, "tokenizer_config.json"))
        copy(os.path.join(model_args.model_name_or_path, "ice_text.model"), os.path.join(save_dir, "ice_text.model"))


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
