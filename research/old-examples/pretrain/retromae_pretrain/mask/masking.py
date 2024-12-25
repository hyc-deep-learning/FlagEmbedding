#!/usr/bin/env Python
# -- coding: utf-8 --

"""
@version: v1.0
@author: huangyc
@file: masking.py
@Description: 
@time: 2024/11/21 20:23
"""
import copy
import random
from abc import abstractmethod

from transformers import PreTrainedTokenizer


class AbsMask:

    def __init__(self, tokenizer: PreTrainedTokenizer, mask_probability: float = 0.3):
        self.tokenizer = tokenizer
        self.mask_probability = mask_probability

    @abstractmethod
    def masking(self, input_ids, all_leafs):
        pass


class BertMask(AbsMask):
    def __init__(self, tokenizer: PreTrainedTokenizer, mask_probability: float = 0.3):
        super(BertMask, self).__init__(tokenizer=tokenizer, mask_probability=mask_probability)

    def masking(self, input_ids, all_leafs):
        """
        以 BERT 类似的训练方式掩蔽部分输入 token

        BERT的mask方式：在选择mask的15%的词当中，80%情况下使用mask掉这个词，10%情况下采用一个任意词替换，剩余10%情况下保持原词汇不变

        参数:
        - input_ids (list of int): 标记化后的输入 id 列表。
        - tokenizer (PreTrainedTokenizer): 训练好的 tokenizer 对象。
        - mask_probability (float): 掩蔽 token 的概率 (0.0 到 1.0)。

        返回:
        - masked_input_ids (list of int): 被掩蔽的输入 id 列表。
        - attention_mask (list of int): 对应于输入 id 的注意力掩码。
        """
        # 用相同的 id 初始化掩蔽后的输入
        masked_input_ids = input_ids.copy()
        # 初始化注意力掩码
        attention_mask = [0] * len(input_ids)  # 所有 token 最初都是有效的

        # 确定要掩蔽的 token 数量
        num_tokens_to_mask = int(len(input_ids) * self.mask_probability)

        # 获取 token 的索引，排除 [CLS] 和 [SEP]
        token_indexes = list(range(1, len(input_ids) - 1))  # 忽略 CLS 和 SEP

        # 随机选择要掩蔽的 token 索引
        selected_indexes = random.sample(token_indexes, num_tokens_to_mask)

        for index in selected_indexes:
            # 有 80% 的概率，替换为 [MASK]
            if random.random() < 0.8:
                masked_input_ids[index] = self.tokenizer.mask_token_id  # [MASK] 标记
                attention_mask[index] = 1  # 对于掩蔽的 token，注意力掩码设为 0
            # 有 10% 的概率，替换为一个随机 token
            elif random.random() < 0.9:
                masked_input_ids[index] = random.randint(1, self.tokenizer.vocab_size - 1)
                attention_mask[index] = 1  # 随机 token 的注意力掩码也设为 0
            # 10% 的概率保持原始 token（不变）

        return masked_input_ids, attention_mask


class LeafMask(AbsMask):
    def __init__(self, tokenizer: PreTrainedTokenizer, mask_probability: float = 0.3):
        super(LeafMask, self).__init__(tokenizer=tokenizer, mask_probability=mask_probability)

    def masking(self, input_ids, all_leafs):
        """
        以 BERT 类似的训练方式掩蔽部分输入 token

        BERT的mask方式：在选择mask的15%的词当中，80%情况下使用mask掉这个词，10%情况下采用一个任意词替换，剩余10%情况下保持原词汇不变

        参数:
        - input_ids (list of int): 标记化后的输入 id 列表。
        - tokenizer (PreTrainedTokenizer): 训练好的 tokenizer 对象。
        - mask_probability (float): 掩蔽 token 的概率 (0.0 到 1.0)。

        返回:
        - masked_input_ids (list of int): 被掩蔽的输入 id 列表。
        - attention_mask (list of int): 对应于输入 id 的注意力掩码。
        """
        # 用相同的 id 初始化掩蔽后的输入
        masked_input_ids = input_ids.copy()
        # 初始化注意力掩码 0表示没有被mask
        attention_mask = [0] * len(input_ids)  # 所有 token 最初都是有效的

        # 确定要掩蔽的 token 数量
        num_tokens_to_mask = int(len(all_leafs) * self.mask_probability)
        # 至少要mask一个
        num_tokens_to_mask = max(1, num_tokens_to_mask)

        # 获取 token 的索引，排除 [CLS] 和 [SEP]
        token_indexes = list(range(len(all_leafs)))  # 忽略 CLS 和 SEP

        # 随机选择要掩蔽的 token 索引
        selected_indexes = random.sample(all_leafs, num_tokens_to_mask)

        for start_index, end_index in selected_indexes:
            # 有 90% 的概率，替换为 [MASK]
            if random.random() < 0.9:
                for index in range(start_index, end_index):
                    index += 1
                    masked_input_ids[index] = self.tokenizer.mask_token_id  # [MASK] 标记
                    attention_mask[index] = 1  # 对于掩蔽的 token，注意力掩码设为 1
            # 有 10% 的概率，替换为一个随机 token
            else:
                for index in range(start_index, end_index):
                    index += 1
                    masked_input_ids[index] = random.randint(1, self.tokenizer.vocab_size - 1)
                    attention_mask[index] = 1  # 随机 token 的注意力掩码也设为 1

        token_labels = copy.deepcopy(input_ids)
        for idx, mask in enumerate(attention_mask[1:-1], start=1):
            if not mask:
                token_labels[idx] = -100

        token_labels[0] = -100
        token_labels[-1] = -100

        return masked_input_ids, attention_mask, token_labels
