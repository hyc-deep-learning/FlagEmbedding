#!/usr/bin/env Python
# -- coding: utf-8 --

"""
@version: v1.0
@author: huangyc
@file: semantic_dataset.py
@Description: 
@time: 2024/12/5 10:40
"""
# !/usr/bin/env Python
# -- coding: utf-8 --

"""
@version: v1.0
@author: huangyc
@file: semantic_logic_pretrain_dataset.py
@Description: 语义逻辑图预训练数据集
@time: 2024/11/1 14:01
"""
import abc
import copy
import json
import multiprocessing
import os
import random
import shutil
import sys
import traceback
from concurrent import futures
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Dict

import datasets
import math
import numpy as np
import torch
from basic_support.comm_funcs.comm_utils import platform, find_root_path_by_resources
from basic_support.comm_funcs.io_utils import write_to_file, read_from_file
from basic_support.comm_funcs.utils_func import time_this_function
from basic_support.logger.logger_config import logger
from datasets import Dataset
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizer
from transformers import TrainingArguments

from util.file_utils import FileUtils

sys.path.insert(0, find_root_path_by_resources().joinpath('src').as_posix())
from data_utils import tensorize_batch
from sample import PretrainRecord, EdgeNamesEnum
from mask.masking import LeafMask

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def distribute_items(total, groups):
    if total < 0:
        total = 1e9

    # 每组的数量
    count_per_group = total // groups
    # 余数
    remainder = int(total % groups)

    # 初始化每组的大小
    group_sizes = [count_per_group] * groups

    # 将余数分配到前 remainder 组中
    for i in range(remainder):
        group_sizes[i] = int(group_sizes[i] + 1)

    return group_sizes


class BaseDataCollator:
    def __init__(self, is_debug, tokenizer, max_seq_length):
        logger.info(f'收集器[{self.__class__.__name__}]开始加载数据,状态[{"生产" if is_debug else "调试"}]')
        self.is_debug = is_debug
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id


class ELFDataset(torch.utils.data.Dataset):
    def __init__(self, is_debug, file_suffix, dataset_files: List[str], device="cuda", max_size=-1):
        logger.info(f'[{self.__class__.__name__}]开始加载数据,状态[{"调试" if is_debug else "生产"}]')
        self.is_debug = is_debug
        self.dataset_files = dataset_files
        self.data_list = []
        self.device = device
        self.file_suffix = file_suffix
        self.max_size = max_size
        self.post_init()

    def post_init(self):
        idx = 0
        for one_data in FileUtils.read_data(data_path_or_file=self.dataset_files,
                                            file_suffix=self.file_suffix):
            if self.max_size == -1 or idx < self.max_size:
                self.data_list.append(one_data)
            idx = idx + 1

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        return self.data_list[item]

    def get_dataloader(self,
                       data_collator: BaseDataCollator,
                       sample_args: TrainingArguments,
                       shuffle=True) -> DataLoader:
        if not isinstance(self, torch.utils.data.IterableDataset):
            shuffle = False
        dataloader_params = {
            "collate_fn": data_collator,
            "batch_size": sample_args.per_device_train_batch_size,
            "num_workers": sample_args.dataloader_num_workers,
            "persistent_workers": sample_args.dataloader_persistent_workers,
            "shuffle": shuffle
        }

        if not isinstance(self, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = SequentialSampler(self)
            dataloader_params["drop_last"] = sample_args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = sample_args.dataloader_prefetch_factor

        dataloader_params["pin_memory"] = True

        # We use the same batch_size as for eval.
        return DataLoader(self, **dataloader_params)


class CacheDatasetMixin:
    def __init__(self, dataset_files: List[str], file_suffix, max_size: int, is_clean_cache: bool = False,
                 num_worker: int = None, ):
        """
        样本转dataset缓存
        @param dataset_files: 样本文件列表
        @param file_suffix: 文件后缀格式
        @param max_size: 获取最大的样本数
        @param is_clean_cache: 是否清空dataset缓存
        @param num_worker: 数据预处理的进程数
        """
        # 样本相关配置
        self.dataset_files = dataset_files
        self.file_suffix = file_suffix
        self.max_size = max_size

        # 数据预处理的进程数
        self.num_worker = num_worker or int((multiprocessing.cpu_count() + 1) / 2)

        # 是否清除样本缓存
        self.is_clean_cache = is_clean_cache

        # 缓存相关参数
        base_cache_path = self._get_cache_base_path(dataset_files=self.dataset_files)
        self.cache_dir = os.path.join(base_cache_path, "tmp_processing")
        self.dataset_cache_dir = os.path.join(base_cache_path, 'dataset_cache')
        self.tmp_cache_path = os.path.join(base_cache_path, 'tmp_dataset_cache')

    def _get_cache_base_path(self, dataset_files: List[str]) -> str:
        """
        获取样本缓存的根目录
        @param dataset_files: 样本文件列表
        @return: 样本的跟目录
        """
        dataset_file = dataset_files[0]

        base_cache_path = dataset_file
        if os.path.isfile(dataset_file):
            base_cache_path = os.path.split(dataset_files[0])[0]
        logger.info(f"样本缓存的根目录为：{base_cache_path}")
        return base_cache_path

    def _gen_data(self, choice_idx: int, max_size: int):
        """
        数据分区处理
        @param choice_idx: 分区号
        @param max_size: 当前分区负责的样本数
        """
        # 定义当前分区的文件名
        out_file_path = os.path.join(self.cache_dir, f'cache_file_{choice_idx}.jsonl')

        # 样本计数器
        sample_counter = 0

        with write_to_file(out_file_path) as out_f:
            idx = 0
            data_iter = FileUtils.read_data(data_path_or_file=self.dataset_files, file_suffix=self.file_suffix)
            for idx, line in tqdm(enumerate(data_iter), f"[{choice_idx}]数据预处理中..."):
                # 跳过非当前worker关注的索引
                if idx % self.num_worker != choice_idx:
                    continue

                out_record, is_use = self._prepare_record(line)
                if not is_use:
                    continue

                if sample_counter >= max_size:
                    break
                out_f.write(f"{out_record}\n")
                sample_counter = sample_counter + 1
                if sample_counter % 2000 == 0:
                    out_f.flush()
            logger.info(f"[{choice_idx}]: 完成数据预处理, 数据量为: {idx + 1}")

    @time_this_function
    def _build_cache(self):
        """ 数据预处理 """
        try:
            executor = ProcessPoolExecutor(max_workers=self.num_worker)
            logger.info("数据预处理开始")
            tasks = []
            per_nums = distribute_items(total=self.max_size, groups=self.num_worker)
            logger.info(f"每个worker生产的目标样本数为: {per_nums}")

            for choice_idx in range(self.num_worker):
                task = executor.submit(self._gen_data, choice_idx=choice_idx, max_size=per_nums[choice_idx])
                tasks.append(task)
            [future.result() for future in futures.as_completed(tasks)]

            logger.info("数据处理成功")
            return True
        except Exception as e:
            logger.info(f"数据处理失败 {traceback.format_exc()}")
            raise e
        finally:
            logger.info("数据预处理结束")

    @abc.abstractmethod
    def _prepare_record(self, line: Dict) -> Tuple[str, bool]:
        """
        单行样本处理
        @param line: 一行样本
        @return: (输出到文件的样本, 是否可用)
        """
        raise NotImplemented("未实现的样本处理函数")

    @time_this_function
    def prepare_samples(self):
        """
        读取数据
        """
        if self.is_clean_cache:
            logger.info(f"删除缓存文件: {self.dataset_cache_dir}")
            shutil.rmtree(self.dataset_cache_dir, ignore_errors=True)

        def gen(inner_file_names: List[str], max_size: int):
            """
            读取样本文件
            @param inner_file_names: 文件列表
            @param max_size: 最大样本数
            @return:
            """
            if max_size < 0:
                max_size = 1e10
            total_count = 0
            for file_name in inner_file_names:
                with read_from_file(file_name) as f:
                    for line in f:
                        yield json.loads(line.strip())
                        total_count += 1
                        if total_count >= max_size:
                            break
                if total_count >= max_size:
                    break

        try:
            logger.info(f"直接从缓存加载: {self.dataset_cache_dir}")
            self.data_list = datasets.load_from_disk(self.dataset_cache_dir, keep_in_memory=False)
            logger.info(f"从缓存加载成功: {self.dataset_cache_dir}")
        except Exception as e:
            logger.info(f"从缓存加载失败: {self.dataset_cache_dir}，故重新构建所有样本")
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir, exist_ok=True)
            self._build_cache()
            file_names = [os.path.join(self.cache_dir, i) for i in os.listdir(self.cache_dir)]
            self.data_list = Dataset.from_generator(gen, gen_kwargs={"inner_file_names": file_names,
                                                                     "max_size": self.max_size},
                                                    cache_dir=self.tmp_cache_path)
            logger.info(f"dataset存储至: {self.dataset_cache_dir}")
            self.data_list.save_to_disk(self.dataset_cache_dir)
            del self.data_list
            logger.info(f"从缓存加载成功: {self.dataset_cache_dir}")
            self.data_list = datasets.load_from_disk(self.dataset_cache_dir, keep_in_memory=False)

            logger.info(f"删除dataset临时文件: {self.tmp_cache_path}")
            shutil.rmtree(self.tmp_cache_path, ignore_errors=True)
            logger.info(f"删除中间文件: {self.cache_dir}")
            shutil.rmtree(self.cache_dir, ignore_errors=True)

        if self.max_size > 0 and len(self.data_list) >= self.max_size:
            self.data_list = self.data_list.take(self.max_size)

        # num_copies = 100  # 目标是扩展到 10000 个样本
        # extended_datasets = [self.data_list] * num_copies  # 创建一个包含 10 个副本的列表
        # # 使用 concatenate_datasets 合并副本
        # self.data_list = concatenate_datasets(extended_datasets)

        logger.info(f"本次样本量为: {len(self.data_list)}")


def pad_or_truncate(nested_list, edge_token_length, pad_token_id) -> List:
    """
    对嵌套列表进行填充和截断处理。

    :param nested_list: List[List]，待处理的嵌套列表
    :param edge_token_length: int，目标长度
    :param pad_token_id: int，填充值
    :return: List[List]，处理后的嵌套列表
    """
    processed_list = []

    for inner_list in nested_list:
        if len(inner_list) > edge_token_length:
            # 截断
            processed_list.append(inner_list[:edge_token_length])
            logger.warning(f"边信息id列表，发生截断")
        else:
            # 填充
            processed_list.append(inner_list + [pad_token_id] * (edge_token_length - len(inner_list)))

    return processed_list


def gen_edge_token_dict(tokenizer: PreTrainedTokenizer, edge_token_length: int) -> Tuple[Dict[str, int], List]:
    """

    @param tokenizer:
    @param edge_token_length:
    @return:
    """
    edge_token_to_idx = {}
    edge_token_ids = []
    for idx, (name, member) in enumerate(EdgeNamesEnum.__members__.items()):
        value = member.value
        value_id = tokenizer.encode(value)[1:-1]
        edge_token_to_idx[value] = idx
        edge_token_ids.append(value_id)

    edge_token_ids = pad_or_truncate(edge_token_ids, edge_token_length=edge_token_length,
                                     pad_token_id=tokenizer.pad_token_id)
    # edge_token_ids = torch.tensor(edge_token_ids)

    return edge_token_to_idx, edge_token_ids


def generate_list_with_zeros(length, n):
    # 生成一个全 0 的列表
    result = [0] * length

    # 随机选择 n 个位置
    indices_to_zero = random.sample(range(length), n)

    # 将选中的位置置为 1
    for index in indices_to_zero:
        result[index] = 1

    return result


class SemanticLogicPretrainDataset(ELFDataset, CacheDatasetMixin):
    def __init__(self, tokenizer: PreTrainedTokenizer, is_debug, file_suffix, dataset_files: List[str], device="cuda",
                 max_seq_length: int = 512, max_size=512, edge_token_length: int = 12, num_worker: int = None,
                 is_clean_cache: bool = False, leaf_node_mask_prob: float = 0.15,
                 none_leaf_node_mask_prob: float = 0.15, ):
        """

        @param tokenizer:
        @param is_debug:
        @param file_suffix:
        @param dataset_files:
        @param device:
        @param max_seq_length: 文本最大长度
        @param max_size: 样本数据量
        @param edge_token_length: 边的token长度
        @param leaf_node_mask_prob:
        @param none_leaf_node_mask_prob:
        """
        CacheDatasetMixin.__init__(self=self, dataset_files=dataset_files, file_suffix=file_suffix, max_size=max_size,
                                   is_clean_cache=is_clean_cache, num_worker=num_worker)
        self.max_seq_length = max_seq_length
        logger.info(f"num_worker： {self.num_worker}")

        # 其他参数
        self.edge_token_length = edge_token_length

        self.leaf_node_mask_prob = leaf_node_mask_prob
        self.none_leaf_node_mask_prob = none_leaf_node_mask_prob

        self.tokenizer = tokenizer
        assert self.tokenizer.pad_token_id is not None, f"tokenizer 需要设置pad_token_id"
        assert self.tokenizer.mask_token_id is not None, f"tokenizer 需要设置mask_token_id"
        assert self.tokenizer.cls_token_id is not None, f"tokenizer 需要设置cls_token_id"
        assert self.tokenizer.sep_token_id is not None, f"tokenizer 需要设置sep_token_id"

        self.masking = LeafMask(tokenizer=self.tokenizer, mask_probability=self.leaf_node_mask_prob)
        super(SemanticLogicPretrainDataset, self).__init__(is_debug=is_debug, file_suffix=file_suffix,
                                                           dataset_files=dataset_files, device=device,
                                                           max_size=max_size)

        self.edge_token_to_idx, self.edge_token_ids = gen_edge_token_dict(tokenizer=tokenizer,
                                                                          edge_token_length=edge_token_length)

    def _prepare_record(self, line: Dict) -> Tuple[str, bool]:
        """

        @param line:
        @return:
        """
        record = json.loads(line['text'].strip())
        is_use = True
        input_ids = record['input_ids']
        pretrain_record = PretrainRecord.build_from_dict(record=record)
        if len(pretrain_record.all_leafs) == 0:
            is_use = False

        if len(pretrain_record.content) > self.max_seq_length:
            is_use = False

        mask_set = []
        decoder_input_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]
        for _ in range(int(min(len(decoder_input_ids), 128) * 1.5)):
            mask_lst = self.masking.masking(input_ids=decoder_input_ids, all_leafs=pretrain_record.all_leafs)[1]
            mask_idx = [idx for idx, v in enumerate(mask_lst) if v == 1]
            mask_set.append(mask_idx)
        pretrain_record.mask_set = mask_set

        out_record = json.dumps(pretrain_record.to_dict, ensure_ascii=False)
        return out_record, is_use

    @time_this_function
    def post_init(self):
        """
        读取数据
        """
        self.prepare_samples()
        logger.info(f"本次样本量为: {len(self.data_list)}")

    def _mask_none_leaf(self, edge_idxs: List[List[int]]) -> List[List[int]]:
        """
        获取随机mask的非叶子节点索引
        @param edge_idxs:
        @return:
        """
        none_leaf_idxs = list(set([tuple(edge_idx[start:start + 2]) for edge_idx in edge_idxs for start in
                                   range(0, len(edge_idx), 2)]))

        none_leaf_num = len(none_leaf_idxs)

        # 确定要掩蔽的 token 数量
        num_tokens_to_mask = min(math.ceil(none_leaf_num * self.none_leaf_node_mask_prob), none_leaf_num)

        # 获取 token 的索引，排除 [CLS] 和 [SEP]
        token_indexes = list(range(none_leaf_num))  # 忽略 CLS 和 SEP

        # 随机选择要掩蔽的 token 索引
        selected_indexes = random.sample(token_indexes, num_tokens_to_mask)

        semantic_attn_mask = [list(none_leaf_idxs[selected_index]) for selected_index in selected_indexes]

        return semantic_attn_mask

    def _build_edge_info(self, data: PretrainRecord, relations):
        def _build_semantic_edge(edge_idx_dict, edge_idx: List, relation_emb) -> Tuple:
            """
            @param edge_idx_dict:
            @param edge_idx:
            @param relation_emb:
            @return:
            """
            id_start = edge_idx_dict[(edge_idx[0], edge_idx[1])]
            id_end = edge_idx_dict[(edge_idx[2], edge_idx[3])]
            return id_start, id_end, relation_emb

        zip_iter = zip(data.edge_idxs, relations)
        semantic_edges = [_build_semantic_edge(edge_idx_dict=data.edge_idx_dict, edge_idx=edge_idx,
                                               relation_emb=self.edge_token_to_idx[relation.relation])
                          for edge_idx, relation in zip_iter]

        return semantic_edges

    def _get_decode_inputs(self, input_ids: List, mask_set: List):
        """
        参考mae的代码
        @param input_ids:
        @param all_leafs:
        @return:
        """
        # decode相关参数
        decoder_input_ids = copy.deepcopy(input_ids)
        decoder_labels = copy.deepcopy(input_ids)
        decoder_labels[0] = -100
        decoder_labels[-1] = -100

        decoder_attention_mask = []
        for i in range(len(input_ids)):
            idx = random.randint(0, min(len(input_ids), 128) - 1)

            my_array = np.zeros(len(input_ids), dtype=int)
            my_array[mask_set[idx]] = 1
            text_decoder_mlm_mask = my_array.tolist()

            decoder_attention_mask.append(text_decoder_mlm_mask)
        # decoder_matrix_attention_mask = 1 - torch.tensor(decoder_attention_mask)

        return {'decoder_input_ids': decoder_input_ids, 'decoder_attention_mask': decoder_attention_mask,
                'decoder_labels': decoder_labels, }

    # @memory_profiler.profile
    def __getitem__(self, index):
        """

        注意：token_mask为全1、semantic_attn_mask为

        token级别：
        节点token_id
        token_ids: List[int] -> [101, 1762, 6898, 6823, 4638, 5687, 2209]
        节点token_mask
        token_attn_mask: List[int] -> [1, 1, 1, 1, 1, 1, 1]
        节点token_id label
        token_labels: List[int] -> [101, 1762, 6898, 6823, 4638, 5687, 2209]

        片段及以上级别：
        边索引元组列表
        semantic_edge_ids: List[List[int]] -> [[0,2,0,1], [0,2,1,2], [2,4,2,4]]
        边的token_ids(不够要padding)
        semantic_edge_component: List[List[int]] -> [[a,b,c,d], [e,d,f,g], [x,y,z,s]]
        片段及以上节点对应的mask
        semantic_attn_mask: [[2,6], [7, 8]]

        @param index:
        @return:
        """
        # 获取数据
        tmp_data = self.data_list[index]
        data = PretrainRecord.build_from_dict(tmp_data)
        del tmp_data

        # 对句子做编码
        # encoding = self.tokenizer.encode_plus(data.content, return_attention_mask=False)
        input_ids = [self.tokenizer.cls_token_id] + data.input_ids + [self.tokenizer.sep_token_id]
        decode_inputs = [self.tokenizer.cls_token_id] + data.input_ids + [self.tokenizer.sep_token_id]
        decode_inputs = self._get_decode_inputs(input_ids=decode_inputs, mask_set=data.mask_set)
        masked_input_ids, attention_mask, token_labels = self.masking.masking(input_ids=input_ids,
                                                                              all_leafs=data.all_leafs, )

        # 构建边信息
        semantic_edges = self._build_edge_info(data=data, relations=data.relations)
        semantic_node_mapping = [[] for _ in range(max(data.edge_idx_dict.values()) + 1)]
        for k, v in data.edge_idx_dict.items():
            semantic_node_mapping[v].append([k[0] + 1, k[1] + 1])

        # 获取随机mask的非叶子节点索引
        # semantic_attn_mask = self._mask_none_leaf(edge_idxs=data.edge_idxs)
        none_leaf_num = len(semantic_node_mapping)
        num_tokens_to_mask = min(math.ceil(none_leaf_num * self.none_leaf_node_mask_prob), none_leaf_num)
        semantic_attn_mask = generate_list_with_zeros(length=none_leaf_num, n=num_tokens_to_mask)

        token_attn_mask = [1] * len(input_ids)
        edge_token_ids = copy.deepcopy(self.edge_token_ids)

        del data
        return {'input_ids': masked_input_ids,
                'token_attn_mask': token_attn_mask,
                'token_labels': token_labels,

                'edge_token_ids': edge_token_ids,
                'semantic_node_mapping': semantic_node_mapping,
                'semantic_edges': semantic_edges,
                'semantic_attn_mask': semantic_attn_mask,
                **decode_inputs}


# 创建数据整理器
class SemanticLogicPretrainDataCollator(BaseDataCollator):
    def __init__(self, is_debug, tokenizer, max_seq_length):
        super(SemanticLogicPretrainDataCollator, self).__init__(is_debug=is_debug, tokenizer=tokenizer,
                                                                max_seq_length=max_seq_length)
        self.tokenizer = tokenizer
        self.attention_mask_padding_value = 0
        self.label_padding_value = -100

    def __call__(self, batch) -> Dict:
        """
        token级别：
        节点token_id
        token_ids: Tensor[batch_size, length] -> tensor([[101, 1762, 6898, 6823, 4638, 5687, 2209], ])
        节点token_mask
        token_attn_mask: Tensor[batch_size, length] -> tensor([[1, 1, 1, 1, 1, 1, 1] , ])
        节点token_id label
        token_labels: Tensor[batch_size, length] -> tensor([[101, 1762, 6898, 6823, 4638, 5687, 2209], ])

        片段及以上级别：
        所有边token ids
        edge_token_ids: Tensor[N, M] -> [[x,x,x], [x,x,x]]
        语义边列表
        semantic_node_mapping: List[ List[List[Tuple[int,int]], ], ] -> [ [[(0,2)], [(0,1)], [(1,2)], ]]
        边索引和边tensor 信息
        semantic_edges: List[ List[Tuple[int, int, int]] ] -> [ [(0,1,0), (0,2,1)], ]
        片段及以上节点对应的mask
        semantic_attn_mask: Tensor[batch_size, max_node_size] -> tensor([ [0, 5,] , ])
        """
        # snapshot1 = tracemalloc.take_snapshot()
        edge_token_ids = batch[0]['edge_token_ids']
        edge_token_ids = torch.tensor(np.array(edge_token_ids))

        # batch token
        input_ids_batch = [torch.tensor(np.array(item['input_ids'])) for item in batch]
        input_ids = tensorize_batch(sequences=input_ids_batch, padding_value=self.tokenizer.pad_token_id)

        attention_mask_batch = [torch.tensor(np.array(item['token_attn_mask'])) for item in batch]
        attention_mask = tensorize_batch(sequences=attention_mask_batch,
                                         padding_value=self.attention_mask_padding_value)

        token_labels_batch = [torch.tensor(np.array(item['token_labels'])) for item in batch]
        token_labels = tensorize_batch(sequences=token_labels_batch, padding_value=self.label_padding_value)

        # 语义信息
        semantic_node_mapping = [item['semantic_node_mapping'] for item in batch]
        semantic_edges = [[semantic_edges for semantic_edges in item['semantic_edges']] for item in batch]

        semantic_attn_mask = tensorize_batch([torch.tensor(np.array(item['semantic_attn_mask'])) for item in batch],
                                             self.attention_mask_padding_value)

        # decode输入
        decoder_input_ids_batch = [torch.tensor(np.array(item['decoder_input_ids'])) for item in batch]
        decoder_input_ids = tensorize_batch(sequences=decoder_input_ids_batch,
                                            padding_value=self.tokenizer.pad_token_id)

        # 注意这里有个1-tensor的操作，原来在dataset那边，但是文章说 dataset应要用基础类型
        decoder_attention_mask = tensorize_batch(
            [1 - torch.tensor(np.array(item['decoder_attention_mask'])) for item in batch],
            padding_value=self.attention_mask_padding_value)

        decoder_labels_batch = [torch.tensor(np.array(item['decoder_labels'])) for item in batch]
        decoder_labels = tensorize_batch(sequences=decoder_labels_batch, padding_value=self.label_padding_value)

        # logger.debug(f"生成样本的时间为: {DateTimeGen.get_now()}")
        # return {'node_token_ids': input_ids, 'token_attn_mask': attention_mask, 'token_labels': token_labels,
        #         'edge_token_ids': edge_token_ids, 'semantic_node_ids': semantic_node_mapping,
        #         'semantic_edge_ids': semantic_edges, 'semantic_attn_mask': semantic_attn_mask,
        # 
        #         'decoder_input_ids': decoder_input_ids,
        #         'decoder_attention_mask': decoder_attention_mask,
        #         'decoder_labels': decoder_labels}
        return {'encoder_input_ids': input_ids, 'encoder_attention_mask': attention_mask, 'encoder_labels': token_labels,
                'decoder_input_ids': decoder_input_ids,
                'decoder_attention_mask': decoder_attention_mask,
                'decoder_labels': decoder_labels}


@time_this_function
def run():
    data_path = 'noup_logic_mock_pretrain/train/samples.jsonl'
    data_path = 'noup_logic_mock_pretrain/train/samples_1w.jsonl'
    data_path = '/root/autodl-tmp/hyc/llm_env/ctx_debug/samples/pretrain/100w/samples_1w.jsonl'

    is_debug = False
    file_suffix = None
    max_seq_length = 5120
    max_size = -1
    is_clean_cache = False
    num_worker = None

    # 获取tokenizer
    if platform() == 'linux':
        tokenizer_path = r'/root/autodl-tmp/pretrained_model/tokenizer/bert-base-chinese'
    else:
        tokenizer_path = r"D:\work\coding\pycharm\pretrained_models\bert-base-chinese"
        tokenizer_path = r"Q:\pyCharmWS\fadu_code\bian_llm\models\bert-base-chinese"

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, )
    # 定义dataset和collate_fn
    semantic_logic_pretrain_dataset = SemanticLogicPretrainDataset(dataset_files=[data_path], tokenizer=tokenizer,
                                                                   is_debug=is_debug, file_suffix='jsonl',
                                                                   max_size=max_size, is_clean_cache=is_clean_cache,
                                                                   num_worker=num_worker, max_seq_length=max_seq_length)
    collate_fn = SemanticLogicPretrainDataCollator(is_debug=is_debug, tokenizer=tokenizer,
                                                   max_seq_length=max_seq_length)

    # out_file = r"/root/autodl-tmp/hyc/llm_env/ctx_debug/samples/pretrain/100w/samples_1w_clean.jsonl"
    # with write_to_file(out_file) as out_f:
    #     for idx in range(len(semantic_logic_pretrain_dataset)):
    #         data = {'text': semantic_logic_pretrain_dataset.data_list[idx]['content']}
    #         out_f.write(f"{json.dumps(data, ensure_ascii=False)}\n")

    # for i in range(len(semantic_logic_pretrain_dataset)):
    #     d = semantic_logic_pretrain_dataset[i]
    #     logger.info(
    #         f"semantic_logic_pretrain_dataset:  {asizeof.asizeof(semantic_logic_pretrain_dataset) / 1024 / 1024}")

    # 准备的测试数据集
    test_loader = DataLoader(dataset=semantic_logic_pretrain_dataset, batch_size=32, shuffle=False, num_workers=12,
                             drop_last=False, collate_fn=collate_fn)
    logger.info(f"开始打印样本")
    for idx, data in tqdm(enumerate(test_loader)):
        pass

    logger.info("程序结束")


if __name__ == '__main__':
    run()
