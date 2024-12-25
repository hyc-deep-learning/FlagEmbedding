import logging
import os
import sys

import transformers
from basic_support.comm_funcs.utils_func import time_this_function
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    BertForMaskedLM,
    AutoConfig,
    HfArgumentParser, set_seed, )
from transformers import (
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl
)
from transformers.trainer_utils import is_main_process

from semantic_dataset import SemanticLogicPretrainDataset, SemanticLogicPretrainDataCollator

os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_DISABLED"] = "false"
os.environ["WANDB_API_KEY"] = "865996538b45bec8e82e89c1017a56f3550bfa85"

# export WANDB_API_KEY=865996538b45bec8e82e89c1017a56f3550bfa85

"""

ndb version 0.17.9
wandb: Run data is saved locally in /root/autodl-tmp/hyc/Qwen/FlagEmbedding/research/old-examples/pretrain/retromae_pretrain/wandb/run-20241108_112237-v1wpozfu
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run cosmic-pine-1
wandb: ⭐️ View project at https://wandb.ai/1832044043-none/huggingface
wandb: 🚀 View run at https://wandb.ai/1832044043-none/huggingface/runs/v1wpozfu

"""

print(sys.path)
from arguments import DataTrainingArguments, ModelArguments
from data import DatasetForPretraining, RetroMAECollator
from modeling import RetroMAEForPretraining
from trainer import PreTrainer

logger = logging.getLogger(__name__)


class TrainerCallbackForSaving(TrainerCallback):
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of an epoch.
        """
        control.should_save = True


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (os.path.exists(training_args.output_dir) and os.listdir(training_args.output_dir)
            and training_args.do_train and not training_args.overwrite_output_dir):
        raise ValueError(f"Output directory ({training_args.output_dir}) already exists and is not empty."
                         "Use --overwrite_output_dir to overcome.")

    model_args: ModelArguments
    data_args: DataTrainingArguments
    training_args: TrainingArguments

    training_args.remove_unused_columns = False

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    if training_args.local_rank in (0, -1):
        logger.info("Training/evaluation parameters %s", training_args)
        logger.info("Model parameters %s", model_args)
        logger.info("Data parameters %s", data_args)

    set_seed(training_args.seed)

    model_class = RetroMAEForPretraining
    collator_class = RetroMAECollator

    if model_args.model_name_or_path:
        model = model_class.from_pretrained(model_args, model_args.model_name_or_path)
        logger.info(f"------Load model from {model_args.model_name_or_path}------")
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    elif model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name)
        bert = BertForMaskedLM(config)
        model = model_class(bert, model_args)
        logger.info("------Init the model------")
        tokenizer = AutoTokenizer.from_pretrained(data_args.tokenizer_name)
    else:
        raise ValueError("You must provide the model_name_or_path or config_name")

    use_self = False
    if use_self:
        dataset = DatasetForPretraining(data_args.train_data)
        data_collator = collator_class(tokenizer,
                                       encoder_mlm_probability=data_args.encoder_mlm_probability,
                                       decoder_mlm_probability=data_args.decoder_mlm_probability,
                                       max_seq_length=data_args.max_seq_length)
        logger.info(f"使用 自带dataset")
    else:
        is_debug = False
        max_size = -1
        is_clean_cache = False
        max_seq_length = 5120
        num_worker = None

        dataset = SemanticLogicPretrainDataset(dataset_files=[data_args.train_data], tokenizer=tokenizer,
                                               is_debug=is_debug, file_suffix='jsonl',
                                               max_size=max_size, is_clean_cache=is_clean_cache,
                                               num_worker=num_worker,
                                               max_seq_length=max_seq_length)
        data_collator = SemanticLogicPretrainDataCollator(is_debug=is_debug, tokenizer=tokenizer,
                                                          max_seq_length=max_seq_length)
        logger.info(f"使用 语义dataset")

    # Initialize our Trainer
    trainer = PreTrainer(model=model, args=training_args, train_dataset=dataset, data_collator=data_collator,
                         tokenizer=tokenizer)
    trainer.add_callback(TrainerCallbackForSaving())

    # # Training
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload


@time_this_function
def main2():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (os.path.exists(training_args.output_dir) and os.listdir(training_args.output_dir)
            and training_args.do_train and not training_args.overwrite_output_dir):
        raise ValueError(f"Output directory ({training_args.output_dir}) already exists and is not empty."
                         "Use --overwrite_output_dir to overcome.")

    model_args: ModelArguments
    data_args: DataTrainingArguments
    training_args: TrainingArguments

    training_args.remove_unused_columns = False

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    if training_args.local_rank in (0, -1):
        logger.info("Training/evaluation parameters %s", training_args)
        logger.info("Model parameters %s", model_args)
        logger.info("Data parameters %s", data_args)

    set_seed(training_args.seed)

    # model_class = RetroMAEForPretraining
    collator_class = RetroMAECollator

    if model_args.model_name_or_path:
        #     model = model_class.from_pretrained(model_args, model_args.model_name_or_path)
        #     logger.info(f"------Load model from {model_args.model_name_or_path}------")
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    elif model_args.config_name:
        #     config = AutoConfig.from_pretrained(model_args.config_name)
        #     bert = BertForMaskedLM(config)
        #     model = model_class(bert, model_args)
        #     logger.info("------Init the model------")
        tokenizer = AutoTokenizer.from_pretrained(data_args.tokenizer_name)
    else:
        raise ValueError("You must provide the model_name_or_path or config_name")

    dataset = DatasetForPretraining(data_args.train_data, num_copies=10)

    data_collator = collator_class(tokenizer,
                                   encoder_mlm_probability=data_args.encoder_mlm_probability,
                                   decoder_mlm_probability=data_args.decoder_mlm_probability,
                                   max_seq_length=data_args.max_seq_length)

    test_loader = DataLoader(dataset=dataset, batch_size=24, shuffle=True, num_workers=12, prefetch_factor=1,
                             drop_last=False, collate_fn=data_collator)
    print(f"样本量为：{len(dataset)}")
    for idx, data in tqdm(enumerate(test_loader)):
        pass

    print("程序结束")

    # Initialize our Trainer
    # trainer = PreTrainer(model=model, args=training_args, train_dataset=dataset, data_collator=data_collator,
    #                      tokenizer=tokenizer)
    # trainer.add_callback(TrainerCallbackForSaving())
    #
    # # # Training
    # trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    # trainer.save_model()  # Saves the tokenizer too for easy upload


if __name__ == "__main__":
    # main2()
    main()

"""

和sem 对比的脚本参数（1w的数据量）

--output_dir outputs --config_name /root/autodl-tmp/pretrained_model/tokenizer/bert-base-chinese --train_data /root/autodl-tmp/hyc/llm_env/ctx_debug/samples/pretrain/100w/samples_1w_clean.jsonl --learning_rate 2e-5 --num_train_epochs 20 --per_device_train_batch_size 32 --dataloader_drop_last True --fp16 True --max_seq_length 512 --logging_steps 10 --dataloader_num_workers 12 --tokenizer_name /root/autodl-tmp/pretrained_model/tokenizer/bert-base-chinese --save_steps 50000




评估结果
数据量：7133  bs=24 num_workers=0
298it [05:43,  1.15s/it]
2024-11-29 10:22:34 utils_func:44 INFO: [main2]执行结束, 耗时344.4174s

数据量：7133  bs=24 num_workers=12
298it [01:04,  4.61it/s]
2024-11-29 10:27:47 utils_func:44 INFO: [main2]执行结束, 耗时66.1641s
"""

"""

--output_dir outputs --config_name /root/autodl-tmp/pretrained_model/tokenizer/bert-base-chinese --train_data /root/autodl-tmp/hyc/llm_env/ctx_debug/samples/pretrain/100w/samples_1w_clean.jsonl --learning_rate 2e-5 --num_train_epochs 20 --per_device_train_batch_size 2 --dataloader_drop_last True --max_seq_length
512 --logging_steps 10 --dataloader_num_workers 12 --tokenizer_name /root/autodl-tmp/pretrained_model/tokenizer/bert-base-chinese --save_steps 50000

"""

"""

torchrun --nproc_per_node {number of gpus} \
-m FlagEmbedding.baai_general_embedding.retromae_pretrain.run \


python FlagEmbedding/baai_general_embedding/retromae_pretrain/run.py --output_dir outputs --config_name /root/autodl-tmp/pretrained_model/tokenizer/bert-base-uncased --train_data toy_pretrain_data.jsonl --learning_rate 2e-5 --num_train_epochs 2 --per_device_train_batch_size {batch size; set 1 for toy data} --dataloader_drop_last True --max_seq_length 512 --logging_steps 10 --dataloader_num_workers 12

python run.py --output_dir outputs --config_name /root/autodl-tmp/pretrained_model/tokenizer/bert-base-uncased --tokenizer_name /root/autodl-tmp/pretrained_model/tokenizer/bert-base-uncased --train_data /root/autodl-tmp/hyc/Qwen/FlagEmbedding/research/old-examples/pretrain/toy_pretrain_data.jsonl --learning_rate 2e-5 --num_train_epochs 20000 --per_device_train_batch_size 32 --dataloader_drop_last True --max_seq_length 512 --logging_steps 10 --dataloader_num_workers 12


python run.py --output_dir outputs --config_name /root/autodl-tmp/pretrained_model/tokenizer/bert-base-uncased --tokenizer_name /root/autodl-tmp/pretrained_model/tokenizer/bert-base-uncased --train_data /root/autodl-tmp/pretrained_model/datasets/miracl-corpus-clear/miracl-corpus-clear.jsonl --learning_rate 2e-5 --num_train_epochs 8 --per_device_train_batch_size 64 --dataloader_drop_last True --max_seq_length 512 --logging_steps 10 --dataloader_num_workers 12 --save_steps 5000


本机器测试脚本参数：
--output_dir outputs --config_name /root/autodl-tmp/pretrained_model/tokenizer/bert-base-uncased --train_data /root/autodl-tmp/hyc/Qwen/FlagEmbedding/research/old-examples/pretrain/toy_pretrain_data.jsonl --learning_rate 2e-5 --num_train_epochs 2 --per_device_train_batch_size 2 --dataloader_drop_last True --max_seq_length 512 --logging_steps 10 --dataloader_num_workers 12 --tokenizer_name /root/autodl-tmp/pretrained_model/tokenizer/bert-base-uncased
--output_dir outputs --config_name /root/autodl-tmp/pretrained_model/tokenizer/bert-base-chinese --train_data /root/autodl-tmp/hyc/llm_env/py_elf_com/src/models/rag/datasets/noup_logic_mock_pretrain/train/samples_clean_txt.jsonl --learning_rate 2e-5 --num_train_epochs 2 --per_device_train_batch_size 2 --dataloader_drop_last True --max_seq_length 512 --logging_steps 10 --dataloader_num_workers 12 --tokenizer_name /root/autodl-tmp/pretrained_model/tokenizer/bert-base-chinese


/root/autodl-tmp/pretrained_model/tokenizer/bert-base-uncased


python run.py --output_dir outputs --config_name /root/autodl-tmp/pretrained_model/tokenizer/roberta-base --tokenizer_name /root/autodl-tmp/pretrained_model/tokenizer/roberta-base --train_data /root/autodl-tmp/pretrained_model/datasets/miracl-corpus-clear/miracl-corpus-clear.jsonl --learning_rate 2e-5 --num_train_epochs 8 --per_device_train_batch_size 16 --dataloader_drop_last True --max_seq_length 512 --logging_steps 10 --dataloader_num_workers 12 --save_steps 5000


# 实际训练机器：
A800专区 / 022机
54a547b170-be4b7458
(真)mae-emd训练机器

nohup deepspeed --num_gpus=4 run.py --output_dir outputs --model_name_or_path /root/autodl-tmp/pretrained_model/tokenizer/bert-base-chinese --train_data /root/autodl-tmp/pretrained_model/datasets/miracl-corpus-clear/samples_clean.jsonl --learning_rate 2e-5 --num_train_epochs 40 --per_device_train_batch_size 64 --dataloader_drop_last True --max_seq_length 512 --logging_steps 10 --dataloader_num_workers 12 --save_steps 50000 --deepspeed ./train_args/ds_z1_config_qwen.json > log_run.log &



--output_dir outputs --model_name_or_path /root/autodl-tmp/hyc/Qwen/FlagEmbedding/research/old-examples/pretrain/retromae_pretrain/outputs/checkpoint-50000 --train_data /root/autodl-tmp/hyc/llm_env/ctx_debug/samples/pretrain/100w/samples_clean_2w.jsonl --learning_rate 2e-5 --num_train_epochs 40 --per_device_train_batch_size 64 --dataloader_drop_last True --max_seq_length 512 --logging_steps 10 --dataloader_num_workers 12 --save_steps 50000

--run 本地
--output_dir outputs --config_name /root/autodl-tmp/pretrained_model/tokenizer/bert-base-chinese --train_data /root/autodl-tmp/hyc/llm_env/ctx_debug/samples/pretrain/100w/samples_1w_clean.jsonl --learning_rate 2e-5 --num_train_epochs 20 --per_device_train_batch_size 32 --dataloader_drop_last True --fp16 True --max_seq_length 512 --logging_steps 10 --dataloader_num_workers 12 --tokenizer_name /root/autodl-tmp/pretrained_model/tokenizer/bert-base-chinese --save_steps 50000

--output_dir outputs --model_name_or_path /root/autodl-tmp/pretrained_model/tokenizer/bert-base-chinese --train_data /root/autodl-tmp/pretrained_model/datasets/miracl-corpus-clear/samples_clean.jsonl --learning_rate 2e-5 --num_train_epochs 40 --per_device_train_batch_size 64 --dataloader_drop_last True --max_seq_length 512 --logging_steps 10 --dataloader_num_workers 12 --save_steps 50000

"""
