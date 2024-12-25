#!/usr/bin/env Python
# -- coding: utf-8 --

"""
@version: v1.0
@author: huangyc
@file: script_data_to_embedding.py
@Description: 
@time: 2024/12/19 10:09
"""

import logging
import os
import sys

import torch
import transformers
from basic_support.comm_classes.comm_batch_helper.impl.parquet_batch_helper import ParquetBatchHelper
from basic_support.comm_funcs.utils_func import time_this_function
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
	AutoTokenizer,
	BertForMaskedLM,
	AutoConfig,
	HfArgumentParser, set_seed, )
from transformers import (
	TrainingArguments
)
from transformers.trainer_utils import is_main_process

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

logger = logging.getLogger(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
is_eval = True


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
		# model = model_class.from_pretrained(model_args, model_args.model_name_or_path)
		tokenizer_path = r"/root/autodl-tmp/pretrained_model/tokenizer/bert-base-chinese"
		model = model_class.from_pretrained(model_args, tokenizer_path)
		state_dict = torch.load(f"{model_args.model_name_or_path}/pytorch_model.bin")
		model.load_state_dict(state_dict)

		logger.info(f"------Load model from {model_args.model_name_or_path}------")
		tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
	elif model_args.config_name:
		config = AutoConfig.from_pretrained(model_args.config_name)
		bert = BertForMaskedLM(config)
		model = model_class(bert, model_args)
		logger.info("------Init the model------")
		tokenizer = AutoTokenizer.from_pretrained(data_args.tokenizer_name)
	else:
		raise ValueError("You must provide the model_name_or_path or config_name")

	dataset = DatasetForPretraining(data_args.train_data, is_eval=is_eval)

	data_collator = collator_class(tokenizer,
								   encoder_mlm_probability=data_args.encoder_mlm_probability,
								   decoder_mlm_probability=data_args.decoder_mlm_probability,
								   max_seq_length=data_args.max_seq_length,
								   is_eval=is_eval)

	return model, dataset, data_collator


def load_data_loader(dataset, data_collator):
	# 准备的测试数据集
	data_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=False, num_workers=12,
							 pin_memory=True, drop_last=False, collate_fn=data_collator, pin_memory_device='cuda:0')
	return data_loader


@time_this_function
def run():
	model, dataset, data_collator = main()

	output_base_path = "noup_embeddings"
	model = model.to(device)
	model.eval()
	data_loader = load_data_loader(dataset, data_collator)
	with torch.no_grad():
		with ParquetBatchHelper(output_base_path=output_base_path, batch_size=1000) as parquet_helper:
			for idx, data in tqdm(enumerate(data_loader), total=len(data_loader)):
				for k, v in data.items():
					if type(v) == torch.Tensor:
						data[k] = v.to(device)

				doc_ids = data.pop('doc_ids')
				model_output = model(**data, eval=is_eval)
				embeddings = model_output[-1][:, 0, :].tolist()

				print(embeddings[0])
				print(doc_ids)
				for doc_id, embedding in zip(doc_ids, embeddings):
					parquet_helper.append_sample({"doc_id": doc_id, "embedding": embedding})

			logger.info(f"{idx}")

	logger.info(f"程序结束")


if __name__ == "__main__":
	run()

"""
--output_dir outputs --model_name_or_path /root/autodl-tmp/hyc/Qwen/FlagEmbedding/research/old-examples/pretrain/retromae_pretrain/outputs/checkpoint-50000 --train_data /root/autodl-tmp/hyc/llm_env/ctx_debug/samples/pretrain/100w/samples_clean_2w.jsonl --learning_rate 2e-5 --num_train_epochs 40 --per_device_train_batch_size 64 --dataloader_drop_last True --max_seq_length 512 --logging_steps 10 --dataloader_num_workers 12 --save_steps 50000

"""
