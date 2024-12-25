#!/usr/bin/env Python
# -- coding: utf-8 --

"""
@version: v1.0
@author: huangyc
@file: sample.py
@Description: 
@time: 2024/11/5 9:24
"""
import json
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

from basic_support.comm_classes.mixin_class import ToDictMixin


class EdgeNamesEnum(Enum):
    SUBJECT = "主语"
    PARAGRAPH = "段落"
    COMPLEMENT = "补语"
    SUB_ADVERBIAL = "子状语"
    HYPOTHETICAL_CONSISTENCY = "假设_一致"
    ASSUMPTION_CONTRADICTORY = "假设_相背"
    CAUSAL = "因果"
    SEQUENCE = "顺承"
    ATTRIBUTE = "定语"
    ADVERBIAL = "状语"
    CONTRAST = "转折"
    PROGRESSION = "递进"
    SUB_PREDICATE = "子谓词"
    SUB_FACT = "子事实"
    SUB_COMPLEMENT = "子补语"
    PREDICATE = "谓词"
    SENTENCE = "句子"
    UNCONDITIONAL_CONDITION = "条件_无条件"
    SUB_OBJECT = "子宾语"
    SUFFICIENT_CONDITION = "条件_充分条件"
    CONDITION_NECESSARY = "条件_必要条件"
    SUB_ATTRIBUTE = "子定语"
    PREDICATE_VERB = "谓语"
    OBJECT = "宾语"
    SUB_PREDICATE_VERB = "子谓语"
    SUB_SUBJECT = "子主语"
    PARALLEL_CONTRAST = "并列_对照"
    PARALLEL_PARALLEL = "并列_平列"
    ALTERNATIVE_DETERMINED = "选择_已定"
    ALTERNATIVE_UNDETERMINED = "选择_未定"


@dataclass
class Relation(ToDictMixin):
    relation: str = field(default=None, metadata={'help': '文本信息'})

    @classmethod
    def build_from_dict(cls, record: Dict or str):
        if type(record) == str:
            return cls(relation=record)
        else:
            return cls(relation=record['relation'])


@dataclass
class PretrainRecord(ToDictMixin):
    content: str = field(metadata={'help': '文本信息'})
    doc_id: str = field(metadata={'help': 'doc_id'})
    input_ids: List[int] = field(metadata={'help': '文本信息的token_ids'})
    edge_idxs: List[List[int]] = field(default_factory=list, metadata={'help': '边索引信息列表'})
    relations: List[Relation] = field(default_factory=list, metadata={'help': '边索引信息列表'})
    all_leafs: List[Tuple[int, int]] = field(default_factory=list, metadata={'help': '叶子索引列表'})
    mask_set: List = field(default_factory=list, metadata={'help': '随机mask列表'})
    edge_idx_dict: Dict = field(default_factory=dict, metadata={'help': 'token索引对应的id索引'})

    @classmethod
    def build_from_dict(cls, record: Dict):
        content = record['content']
        input_ids = record['input_ids']
        edge_idxs = record['edge_idxs']
        doc_id = record['doc_id']
        all_leafs = record.get('all_leafs', [])
        mask_set = record.get('mask_set', [])
        edge_idx_dict = record['edge_idx_dict']

        if type(edge_idx_dict) == str:
            edge_idx_dict = json.loads(edge_idx_dict)
        edge_idx_dict = {eval(k): v for k, v in edge_idx_dict.items()}

        relations = [Relation.build_from_dict(relation) for relation in record['relations']]
        return cls(content=content, doc_id=doc_id, all_leafs=all_leafs, mask_set=mask_set, input_ids=input_ids,
                   edge_idxs=edge_idxs, relations=relations, edge_idx_dict=edge_idx_dict)

    @property
    def to_dict(self):
        record = super(PretrainRecord, self).to_dict
        record['edge_idx_dict'] = {str(k): v for k, v in record['edge_idx_dict'].items()}
        record['edge_idx_dict'] = json.dumps(record['edge_idx_dict'], ensure_ascii=False)
        return record

    @classmethod
    def build_from_dataset(cls, record):
        pass


if __name__ == '__main__':
    value_ids = []
    for name, member in EdgeNamesEnum.__members__.items():
        value = member.value
        value_id = tokenizer.encode(value)
        value_ids.append(value_id)

    # for name, member in enumerate(EdgeNamesEnum):
    #     print(name, '=>', member)
