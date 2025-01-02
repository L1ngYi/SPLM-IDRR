import json
# 引入transformers库中定义的PreTrainedTokenizer，用于词表和分词
from transformers.tokenization_utils import PreTrainedTokenizer

# 导入其他自定义模块和必要的工具
from .data_utils import InputFeatures
import re
from .prompt_base import Verbalizer
from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
重写了 process_logits
aggregate被改写了

'''

# 定义ManualVerbalizer类，继承自Verbalizer，表示一个手动定义的Verbalizer类，用于处理标签词的映射
class ManualVerbalizer(Verbalizer):
    r"""
    基础的手动定义的Verbalizer类，继承自Verbalizer类，用于将标签映射到标签词

    Args:
        tokenizer (PreTrainedTokenizer): 当前预训练模型的tokenizer，用于获取词汇表。
        classes (List[Any]): 当前任务的标签类别列表。
        label_words (Union[List[str], List[List[str]], Dict[List[str]]], optional): 通过标签映射的标签词。
        prefix (str, optional): Verbalizer的前缀字符串（用于像RoBERTa这样的对前缀空格敏感的PLMs）
        multi_token_handler (str, optional): 当生成的标签词由多个token组成时的处理策略。
        post_log_softmax (bool, optional): 是否在标签词的logits上应用log softmax后处理。默认是True。
    """
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 classes: Optional[List] = None,
                 num_classes: Optional[Sequence[str]] = None,
                 label_words: Optional[Union[Sequence[str], Mapping[str, str]]] = None,
                 prefix: Optional[str] = " ",
                 multi_token_handler: Optional[str] = "first",
                 post_log_softmax: Optional[bool] = False,
                ):
        # 初始化父类Verbalizer
        super().__init__(tokenizer=tokenizer, num_classes=num_classes, classes=classes)
        # 定义一些实例变量，用于存储前缀、处理策略、标签词和后处理选项
        self.prefix = prefix
        self.multi_token_handler = multi_token_handler
        self.label_words = label_words
        self.post_log_softmax = post_log_softmax
    
    # 当设置了标签词时执行的函数
    def on_label_words_set(self):
        # 调用父类的on_label_words_set方法
        super().on_label_words_set()
        # 为标签词添加前缀，并更新为新的标签词列表
        self.label_words = self.add_prefix(self.label_words, self.prefix)
        # 生成映射标签词到模型参数的张量
        self.generate_parameters()
        
    # 静态方法：为标签词添加前缀
    @staticmethod
    def add_prefix(label_words, prefix):
        r"""为标签词添加前缀。例如，当标签词位于模板的中间时，前缀应为' '。

        Args:
            label_words (Union[Sequence[str], Mapping[str, str]], optional): 通过标签映射的标签词。
            prefix (str, optional): Verbalizer的前缀字符串。

        Returns:
            Sequence[str]: 带前缀的新标签词。
        """
        new_label_words = []
        # 如果标签词是字符串列表，转换为包含标签词的列表列表
        if isinstance(label_words[0], str):
            label_words = [[w] for w in label_words]

        # 为每个标签的标签词添加前缀
        for label_words_per_label in label_words:
            new_label_words_per_label = []
            for word in label_words_per_label:
                if word.startswith("<!>"):
                    new_label_words_per_label.append(word.split("<!>")[1])
                else:
                    new_label_words_per_label.append(prefix + word)
            new_label_words.append(new_label_words_per_label)
        return new_label_words
    
    # 生成标签词的参数
    def generate_parameters(self) -> List:
        r"""在基本手动模板中，参数直接从标签词生成。
        实现中要求标签词不能被分词器分割为多个token。
        """

        all_ids = []
        for words_per_label in self.label_words:
            ids_per_label = []
            for word in words_per_label:
                # 将标签词通过tokenizer编码为token ID
                ids = self.tokenizer.encode(word, add_special_tokens=False)
                ids_per_label.append(ids)
            all_ids.append(ids_per_label)
        
        # 计算标签词中最长的长度和最大数量
        max_len  = max([max([len(ids) for ids in ids_per_label]) for ids_per_label in all_ids])
        max_num_label_words = max([len(ids_per_label) for ids_per_label in all_ids])
        
        # 创建用于存储标签词ID和mask的张量
        words_ids_mask = [[[1]*len(ids) + [0]*(max_len-len(ids)) for ids in ids_per_label]
                             + [[0]*max_len]*(max_num_label_words-len(ids_per_label))
                             for ids_per_label in all_ids]
        words_ids = [[ids + [0]*(max_len-len(ids)) for ids in ids_per_label]
                             + [[0]*max_len]*(max_num_label_words-len(ids_per_label))
                             for ids_per_label in all_ids]
        
        # 将标签词ID和mask转换为张量
        words_ids_tensor = torch.tensor(words_ids)
        words_ids_mask = torch.tensor(words_ids_mask)
        
        # 将标签词ID和mask设置为不可训练的参数
        self.label_words_ids = nn.Parameter(words_ids_tensor, requires_grad=False)
        self.words_ids_mask = nn.Parameter(words_ids_mask, requires_grad=False)
        self.label_words_mask = nn.Parameter(torch.clamp(words_ids_mask.sum(dim=-1), max=1), requires_grad=False)

    # 投影函数，将模型输出logits映射到标签词空间
    def project(self,
                logits: torch.Tensor,
                conn_linear_logits = None,
                **kwargs,
                ) -> torch.Tensor:
        r"""
        投影到标签词，返回的是标签词的概率。

        Args:
            logits (torch.Tensor): 原始logits。

        Returns:
            torch.Tensor: 标签词的概率。
        """
        label_words_logits = logits[:, self.label_words_ids]
        label_words_logits = self.handle_multi_token(label_words_logits, self.words_ids_mask)
        # 使用mask剔除无效的标签词logits
        label_words_logits -= 1e9*(1-self.label_words_mask)#label_words_logits -= 10000*(1-self.label_words_mask) 原文
        
        return label_words_logits

    # 处理原始logits的函数
    def process_logits(self, logits: torch.Tensor, conn_linear_logits = None, **kwargs): #这个函数被重写了
        r"""处理原始logits的完整流程：

        (1) 将logits投影到标签词logits

        (2) 聚合标签词logits

        Args:
            logits (torch.Tensor): 原始logits

        Returns:
            torch.Tensor: 最终的标签logits
        原函数
        label_words_logits = self.project(logits, **kwargs)  #Output: (batch_size, num_classes) or  (batch_size, num_classes, num_label_words_per_label)


        if self.post_log_softmax:
            # normalize
            label_words_probs = self.normalize(label_words_logits)

            # calibrate
            if  hasattr(self, "_calibrate_logits") and self._calibrate_logits is not None:
                label_words_probs = self.calibrate(label_words_probs=label_words_probs)

            # convert to logits
            label_words_logits = torch.log(label_words_probs+1e-15)

        # aggregate
        label_logits = self.aggregate(label_words_logits)
        return label_logits
        """
        # 投影到标签词的logits
        label_words_logits = self.project(logits, **kwargs)
        # 聚合标签词logits
        label_logits = self.aggregate(label_words_logits)
        return label_logits    

    # 对logits进行归一化处理
    def normalize(self, logits: torch.Tensor) -> torch.Tensor:
        """
        对logits进行归一化，返回标签词集合的概率。

        Args:
            logits (Tensor): 完整词汇表的logits

        Returns:
            Tensor: 标签词集合的概率。
        """
        batch_size = logits.shape[0]
        return F.softmax(logits.reshape(batch_size, -1), dim=-1).reshape(*logits.shape)

    # 聚合函数，将标签词logits聚合成标签logits
    def aggregate(self, label_words_logits: torch.Tensor) -> torch.Tensor:
        r"""使用权重聚合标签词logits。

        Args:
            label_words_logits(torch.Tensor): 标签词的logits。

        Returns:
            torch.Tensor: 聚合后的标签logits。
        """
        # 对每个标签的标签词logits取最大值
       

        label_words_logits = (label_words_logits).max(axis=-1)[0] #label_words_logits = (label_words_logits * self.label_words_mask).sum(-1)/self.label_words_mask.sum(-1) 原文
        return label_words_logits

    # 校准函数，对标签词概率进行校准
    def calibrate(self, label_words_probs: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""
        校准标签词概率。

        Args:
            label_words_probs (torch.Tensor): 标签词的概率分布，形状为 [batch_size, num_classes, num_label_words_per_class]

        Returns:
            torch.Tensor: 校准后的标签词概率。
        """
        shape = label_words_probs.shape
        assert self._calibrate_logits.dim() == 1, "self._calibrate_logits不是1维张量"
        
        # 计算校准的标签词概率
        calibrate_label_words_probs = self.normalize(self.project(self._calibrate_logits.unsqueeze(0), **kwargs))
        
        # 校准概率计算
        label_words_probs /= (calibrate_label_words_probs + 1e-15)
        
        # 归一化
        norm = label_words_probs.reshape(shape[0], -1).sum(dim=-1, keepdim=True)
        label_words_probs = label_words_probs.reshape(shape[0], -1) / norm
        label_words_probs = label_words_probs.reshape(*shape)
        return label_words_probs
