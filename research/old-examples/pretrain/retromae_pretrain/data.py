import copy
import os
import random
from copy import deepcopy
from dataclasses import dataclass

import torch.utils.data.dataset
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import DataCollatorForWholeWordMask

from utils import tensorize_batch

from typing import Optional


class DatasetForPretraining(torch.utils.data.Dataset):
	def __init__(self, data_dir, num_copies: Optional[int] = None, is_eval: bool = False):
		self.is_eval = is_eval
		if os.path.isdir(data_dir):
			datasets = []
			for file in os.listdir(data_dir):
				print(f"Loading {file}")
				file = os.path.join(data_dir, file)
				datasets.append(self.load_dataset(file))
			self.dataset = concatenate_datasets(datasets)
		else:
			print(f"Loading {data_dir}")
			self.dataset = self.load_dataset(data_dir)

		if num_copies:
			print(f"触发模拟更多的数据：扩展{num_copies} 倍")
			# 计算需要复制的次数
			num_copies = 100  # 目标是扩展到 10000 个样本
			extended_datasets = [self.dataset] * num_copies  # 创建一个包含 10 个副本的列表
			# 使用 concatenate_datasets 合并副本
			self.dataset = concatenate_datasets(extended_datasets)

	def load_dataset(self, file):
		if file.endswith('.jsonl') or file.endswith('.json'):
			return load_dataset('json', data_files=file)['train']
		elif os.path.isdir(file):
			return Dataset.load_from_disk(file)
		else:
			raise NotImplementedError(f"Not support this file format:{file}")

	def __getitem__(self, item):
		if self.is_eval:
			return (self.dataset[item]['text'], self.dataset[item]['doc_id'])
		else:
			return self.dataset[item]['text']

	def __len__(self):
		return len(self.dataset)


@dataclass
class RetroMAECollator(DataCollatorForWholeWordMask):
	max_seq_length: int = 512
	encoder_mlm_probability: float = 0.15
	decoder_mlm_probability: float = 0.15
	is_eval: bool = False

	def __call__(self, examples):
		doc_ids = []
		if self.is_eval:
			examples, doc_ids = zip(*examples)
		input_ids_batch = []
		attention_mask_batch = []
		encoder_mlm_mask_batch = []
		decoder_labels_batch = []
		decoder_matrix_attention_mask_batch = []

		for e in examples:
			e_trunc = self.tokenizer.encode(e, max_length=self.max_seq_length, truncation=True)
			tokens = [self.tokenizer._convert_id_to_token(tid) for tid in e_trunc]

			self.mlm_probability = self.encoder_mlm_probability
			text_encoder_mlm_mask = self._whole_word_mask(tokens)
			if self.is_eval:
				text_encoder_mlm_mask = copy.deepcopy(e_trunc)
				text_encoder_mlm_mask[0] = -100
				text_encoder_mlm_mask[-1] = -100

			self.mlm_probability = self.decoder_mlm_probability
			mask_set = []
			for _ in range(min(len(tokens), 128)):
				mask_set.append(self._whole_word_mask(tokens))

			text_matrix_attention_mask = []
			for i in range(len(tokens)):
				idx = random.randint(0, min(len(tokens), 128) - 1)
				text_decoder_mlm_mask = deepcopy(mask_set[idx])
				text_decoder_mlm_mask[i] = 1
				text_matrix_attention_mask.append(text_decoder_mlm_mask)

			input_ids_batch.append(torch.tensor(e_trunc))
			attention_mask_batch.append(torch.tensor([1] * len(e_trunc)))
			e_trunc[0] = -100
			e_trunc[-1] = -100
			decoder_labels_batch.append(torch.tensor(e_trunc))

			encoder_mlm_mask_batch.append(torch.tensor(text_encoder_mlm_mask))
			decoder_matrix_attention_mask_batch.append(1 - torch.tensor(text_matrix_attention_mask))

		input_ids_batch = tensorize_batch(input_ids_batch, self.tokenizer.pad_token_id)
		attention_mask_batch = tensorize_batch(attention_mask_batch, 0)
		origin_input_ids_batch = input_ids_batch.clone()
		encoder_mlm_mask_batch = tensorize_batch(encoder_mlm_mask_batch, 0)
		if self.is_eval:
			encoder_input_ids_batch = copy.deepcopy(input_ids_batch)
			encoder_labels_batch = copy.deepcopy(encoder_mlm_mask_batch)
		else:
			encoder_input_ids_batch, encoder_labels_batch = self.torch_mask_tokens(input_ids_batch,
																				   encoder_mlm_mask_batch)
		decoder_labels_batch = tensorize_batch(decoder_labels_batch, -100)
		matrix_attention_mask_batch = tensorize_batch(decoder_matrix_attention_mask_batch, 0)

		if self.is_eval:
			batch = {
				"encoder_input_ids": encoder_input_ids_batch,
				"encoder_attention_mask": attention_mask_batch,
				"encoder_labels": encoder_labels_batch,
				"decoder_input_ids": origin_input_ids_batch,
				"decoder_attention_mask": matrix_attention_mask_batch,  # [B,L,L]
				"decoder_labels": decoder_labels_batch,
				"doc_ids": doc_ids
			}
		else:
			batch = {
				"encoder_input_ids": encoder_input_ids_batch,
				"encoder_attention_mask": attention_mask_batch,
				"encoder_labels": encoder_labels_batch,
				"decoder_input_ids": origin_input_ids_batch,
				"decoder_attention_mask": matrix_attention_mask_batch,  # [B,L,L]
				"decoder_labels": decoder_labels_batch,
			}

		return batch
