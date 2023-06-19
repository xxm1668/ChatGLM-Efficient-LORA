# coding=utf-8
# Implements parameter-efficient training of a reward model based on ChatGLM.
# This code is inspired by:
# https://github.com/lvwerra/trl/blob/main/examples/summarization/scripts/reward_summarization.py
# https://github.com/CarperAI/trlx/blob/main/examples/summarize_rlhf/reward_model/train_reward_model_gptj.py

import wandb
from utils import (
    PairwiseDataCollatorForChatGLM,
    PairwiseTrainerForChatGLM,
    LogCallback,
    load_pretrained,
    prepare_args,
    prepare_data,
    preprocess_data,
    compute_accuracy,
    plot_loss,
)
from transformers.trainer_callback import TrainerCallback


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


def main():
    # prepare pretrained model and dataset
    model_args, data_args, training_args, finetuning_args = prepare_args(stage="rm")

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
    model, tokenizer = load_pretrained(model_args, finetuning_args, training_args.do_train, stage="rm")
    dataset = preprocess_data(dataset, tokenizer, data_args, training_args, stage="rm")
    data_collator = PairwiseDataCollatorForChatGLM(tokenizer, model.pretrained_model)

    training_args.remove_unused_columns = False  # Important for pairwise dataset

    # Split the dataset
    if training_args.do_train:
        if data_args.dev_ratio > 1e-6:
            dataset = dataset.train_test_split(test_size=data_args.dev_ratio)
            trainer_kwargs = {"train_dataset": dataset["train"], "eval_dataset": dataset["test"]}
        else:
            trainer_kwargs = {"train_dataset": dataset}
    else:  # do_eval or do_predict
        trainer_kwargs = {"eval_dataset": dataset}

    # Initialize our Trainer
    trainer = PairwiseTrainerForChatGLM(
        finetuning_args=finetuning_args,
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[WandbCallback()] if training_args.do_train else None,
        compute_metrics=compute_accuracy,
        **trainer_kwargs
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        trainer.save_model()
        if trainer.is_world_process_zero() and model_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
