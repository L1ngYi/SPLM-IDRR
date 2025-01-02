import itertools  # 引入itertools模块，提供迭代器相关功能
import warnings  # 引入warnings模块，用于控制警告信息
import torch  # 引入torch模块，PyTorch库
import torch.nn as nn  # 引入nn模块，用于神经网络层的构建
from torch.utils.data import Dataset  # 引入Dataset类，用于数据集的管理
from typing import Union, List, Tuple, Dict, Optional  # 引入类型提示相关类
from transformers.data.processors.utils import InputExample, InputFeatures  # 引入InputExample和InputFeatures，用于数据处理
from transformers.tokenization_utils import PreTrainedTokenizer  # 引入PreTrainedTokenizer类，用于处理tokenizer
from collections import defaultdict  # 引入defaultdict，默认字典类型
import numpy as np  # 引入numpy库，进行数值计算

'''
相比原文判断截断方式，它直接采用了尾截断，没有判断逻辑
加了一句
self.segment_emb = segment_emb  # 是否使用segment embedding   这一句是原文没有的
禁用了下面两个模板标记
        # self.template_eos_token = '<eos>'
        # self.template_bos_token = '<bos>'
'''



# 定义TokenizerWrapper类，封装tokenizer功能
class TokenizerWrapper:
    # 初始化函数，设置tokenizer的相关参数
    def __init__(self,
                 max_seq_length: int,  # 最大序列长度
                 tokenizer: PreTrainedTokenizer,  # 预训练tokenizer
                 create_token_type_ids: Optional[str] = False,  # 是否创建token类型ID
#                truncate_method: Optional[str] = 'tail',   ##原文有的
                 segment_emb: Optional[str] = False,  # 是否使用segment embedding
                 **kwargs):
        self.max_seq_length = max_seq_length  # 设置最大序列长度
        self.tokenizer = tokenizer  # 设置tokenizer


        self.truncate_fct = self.truncate_from_tail  # 设置截断方式，默认从尾部截断。原文在这里做了参数判断，看截取方法。它只取用了尾部截断

        self.create_token_type_ids = create_token_type_ids  # 是否创建token类型ID
        self.segment_emb = segment_emb  # 是否使用segment embedding   这一句是原文没有的


        # 模板标记，用于替换不同的token
        self.template_mask_token = '<mask>'
        # self.template_eos_token = '<eos>'
        # self.template_bos_token = '<bos>'
        self.template_sep_token = '<sep>'
        self.template_cls_token = '<cls>'
        self.template_pad_token = '<pad>'

        from transformers import logging  # 引入日志模块
        verbosity_before = logging.get_verbosity()  # 保存原来的日志级别
        logging.set_verbosity(logging.CRITICAL)  # 设置日志级别为CRITICAL，避免冗余日志输出
        # 映射模板标记到实际tokenizer中的token
        self.mask_token_map = {self.template_mask_token: self.tokenizer.mask_token if hasattr(self.tokenizer, 'mask_token') else ''}  
        # self.eos_token_map = {self.template_eos_token: self.tokenizer.eos_token if hasattr(self.tokenizer, 'eos_token') else ''}
        # self.bos_token_map = {self.template_bos_token: self.tokenizer.bos_token if hasattr(self.tokenizer, 'bos_token') else ''}
        self.sep_token_map = {self.template_sep_token: self.tokenizer.sep_token if hasattr(self.tokenizer, 'sep_token') else ''}
        self.cls_token_map = {self.template_cls_token: self.tokenizer.cls_token if hasattr(self.tokenizer, 'cls_token') else ''}
        self.pad_token_map = {self.template_pad_token: self.tokenizer.pad_token if hasattr(self.tokenizer, 'pad_token') else ''}
        logging.set_verbosity(verbosity_before)  # 恢复原来的日志级别

        # 统计截断情况
        self.num_truncated_sentences = 0
        self.total_passed_sentences = 0

    @property
    def truncate_rate(self,):
        """
        计算并返回截断的比例，即被截断的句子数占总句子数的比例
        """
        if self.total_passed_sentences == 0:
            return None  # 如果没有句子处理，返回None
        else:
            return self.num_truncated_sentences / self.total_passed_sentences  # 截断率

    @property
    def special_tokens_maps(self,) -> Dict:
        """
        返回一个包含特殊token的字典，适配特定语言模型
        """
        if not hasattr(self, "_special_tokens_map"):  # 如果没有特殊token映射，创建一个
            _special_tokens_map = {}
            for attrname in self.__dict__.keys():
                if attrname.endswith('_token_map'):  # 如果属性名以'_token_map'结尾，则添加到映射中
                    _special_tokens_map.update(getattr(self, attrname))
        return _special_tokens_map

    # 以下两个方法需要子类实现，根据是否使用mask来处理tokenization
    def tokenize_with_mask(self,
                            wrapped_example: List[Dict],
                            ) -> InputFeatures:
        raise NotImplementedError  # 抛出未实现异常，子类需要重写

    def tokenize_without_mask(self,
                            wrapped_example: List[Dict],
                            ) -> InputFeatures:
        raise NotImplementedError  # 抛出未实现异常，子类需要重写

    @staticmethod
    def truncate_from_tail(input_dict: Dict,
                 num_tokens_to_truncate: int=0) -> Dict:
        """
        从尾部截断输入数据，直到达到指定的token数
        """
        truncated_example = defaultdict(list)  # 使用defaultdict，确保没有key时会返回空列表
        shortenable_ids = input_dict['shortenable_ids']  # 获取可截断的部分ID
        for key in input_dict:  # 遍历输入字典的所有键
            parts = input_dict[key]  # 获取该键的对应值
            to_trunc = num_tokens_to_truncate  # 要截断的token数量
            for i, part in enumerate(parts[::-1]):  # 从后往前处理每一部分
                if len(part) == 0:  # 如果该部分为空，则跳过
                    continue
                if shortenable_ids[-1-i][0] == 0:  # 如果该部分不可截断，跳过
                    continue
                # 截断该部分，直到达到要求的token数
                parts[-1-i] = part[:-to_trunc] if to_trunc < len(part) else []
                to_trunc -= len(part)  # 更新剩余要截断的token数
                if to_trunc <= 0:  # 如果已经达到截断目标，退出
                    break
            truncated_example[key] = parts  # 更新截断后的部分
        return truncated_example  # 返回截断后的字典

    @staticmethod
    def concate_parts(input_dict: Dict) -> Dict:
        """
        将输入字典中各部分的token合并为一个长的token列表
        """
        for key in input_dict:  # 遍历字典中的所有键
            input_dict[key] = list(itertools.chain(*input_dict[key]))  # 使用itertools.chain将每个部分拼接在一起
        return input_dict  # 返回合并后的字典

    @staticmethod
    def padding(input_dict: Dict,
                max_len: int, pad_id_for_inputs: int=0, pad_id_for_others: int=0) -> None:
        """
        对输入字典中的token进行padding，保证每个输入的长度一致
        """
        for key, value in input_dict.items():  # 遍历字典中的每个键值对
            if len(input_dict[key]) > max_len:  # 如果某个输入的长度超过了最大长度
                raise ValueError(f'''Truncated seq length of '{key}' still greater than max length {max_len}."\
                    "One possible reason is that no enough shortenable parts in template. Try adding {{"shortenable": "True"}} property.
                ''')
            if 'input' in key:  # 对于包含'input'的key，使用pad_id_for_inputs进行填充
                input_dict[key].extend([pad_id_for_inputs] * (max_len - len(value)))
            else:  # 对于其他key，使用pad_id_for_others进行填充
                input_dict[key].extend([pad_id_for_others] * (max_len - len(value)))
        return input_dict  # 返回padding后的字典

    def add_special_tokens(self, encoder_inputs):
        """
        为encoder输入添加特殊tokens
        """
        for key in encoder_inputs:  # 遍历encoder输入中的每个键
            if key == "input_ids":
                # 对input_ids进行特殊tokens添加
                with warnings.catch_warnings():  # 捕获并忽略警告
                    warnings.simplefilter("ignore")
                    encoder_inputs[key] = self.tokenizer.build_inputs_with_special_tokens(
                                                            encoder_inputs[key])
            else:
                # 计算特殊token的mask
                special_tokens_mask = np.array(self.tokenizer.get_special_tokens_mask(encoder_inputs[key]))
                # 使用特殊tokens进行填充
                with_special_tokens = np.array(self.tokenizer.build_inputs_with_special_tokens(encoder_inputs[key]))
                if key in ["soft_token_ids"]:  # 对于soft_token_ids，处理方式不同
                    encoder_inputs[key] =  ((1-special_tokens_mask) * with_special_tokens).tolist()  # 使用0替换特殊token
                else:
                    encoder_inputs[key] =  ((1-special_tokens_mask) * with_special_tokens - special_tokens_mask * 100).tolist()  # 使用-100替换特殊token
        return encoder_inputs  # 返回添加了特殊token的输入

    def truncate(self, encoder_inputs):
        """
        对encoder输入进行截断，确保总长度不超过最大长度
        """
        total_tokens = sum([len(part) for part in encoder_inputs['input_ids']])  # 计算输入的总token数
        num_specials = self.num_special_tokens_to_add  # 获取特殊tokens的数量
        num_tokens_to_truncate = total_tokens - self.max_seq_length + num_specials  # 计算需要截断的token数
        self.total_passed_sentences += 1  # 更新总句子数
        if num_tokens_to_truncate > 0:  # 如果需要截断
            self.num_truncated_sentences += 1  # 更新截断句子数
            encoder_inputs = self.truncate_fct(input_dict=encoder_inputs,
                          num_tokens_to_truncate=num_tokens_to_truncate)  # 调用截断函数进行处理
        return encoder_inputs  # 返回截断后的输入
