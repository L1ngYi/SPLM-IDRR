import copy
import json
import pickle
from typing import *

import torch
from torch.utils.data._utils.collate import default_collate

from typing import Union


class InputExample(object):
    """定义一个表示输入样本的类，包含文本段、标签等信息"""

    def __init__(self,
                 guid = None,  # 样本的唯一标识
                 text_a = "",  # 主文本段
                 text_b = "",  # 次要文本段，可能为空
                 label = None,  # 分类任务中的标签
                 meta: Optional[Dict] = None,  # 额外信息的字典，默认为空字典
                 tgt_text: Optional[Union[str,List[str]]] = None  # 生成任务的目标文本
                ):
        # 初始化类中的各个属性
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.meta = meta if meta else {}
        self.tgt_text = tgt_text

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """将当前实例序列化为字典，返回一个深拷贝"""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """将当前实例序列化为JSON字符串格式"""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def keys(self, keep_none=False):
        """返回实例中属性的键名列表，可选是否保留值为None的键"""
        return [key for key in self.__dict__.keys() if getattr(self, key) is not None]

    @staticmethod
    def load_examples(path: str) -> List['InputExample']:
        """从文件中加载一组输入样本"""
        with open(path, 'rb') as fh:
            return pickle.load(fh)

    @staticmethod
    def save_examples(examples: List['InputExample'], path: str) -> None:
        """将一组输入样本保存到文件中"""
        with open(path, 'wb') as fh:
            pickle.dump(examples, fh)


class InputFeatures(dict):
    """
    用于存储模型输入特征的类，定义了一个包含一组预定义键的字典，默认值为None。
    支持字典的常规操作，并可以转换为torch.Tensor格式，方便使用于模型输入。
    """

    # 可转换为tensor的键列表
    tensorable_keys = ['input_ids', 'inputs_embeds', 'attention_mask', 'token_type_ids', 'label',
        'decoder_input_ids', 'decoder_inputs_embeds', 'soft_token_ids',
        'past_key_values', 'loss_ids','conns_index']
    
    # 所有键，包括非tensor键
    all_keys = ['input_ids', 'inputs_embeds', 'attention_mask', 'token_type_ids', 'label',
        'decoder_input_ids', 'decoder_inputs_embeds', 'soft_token_ids',
        'past_key_values', 'loss_ids','guid', 'tgt_text', 'encoded_tgt_text', 'input_ids_len','conns_index']
    non_tensorable_keys = []

    def __init__(self,
                input_ids: Optional[Union[List, torch.Tensor]] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                attention_mask: Optional[Union[List[int], torch.Tensor]] = None,
                token_type_ids: Optional[Union[List[int], torch.Tensor]] = None,
                label: Optional[Union[int, torch.Tensor]] = None,
                decoder_input_ids: Optional[Union[List, torch.Tensor]] = None,
                decoder_inputs_embeds: Optional[torch.Tensor] = None,
                soft_token_ids: Optional[Union[List, torch.Tensor]] = None,
                past_key_values: Optional[torch.Tensor] = None,  # 用于前缀调优
                loss_ids: Optional[Union[List, torch.Tensor]] = None,
                guid: Optional[str] = None,
                tgt_text: Optional[str] = None,
                use_cache: Optional[bool] = None,
                encoded_tgt_text: Optional[str] = None,
                input_ids_len: Optional[int] = None,
                conns_index = None,
                **kwargs):

        # 初始化类中的各个属性，便于后续处理
        self.input_ids = input_ids
        self.inputs_embeds = inputs_embeds
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label
        self.decoder_input_ids = decoder_input_ids
        self.decoder_inputs_embeds = decoder_inputs_embeds
        self.soft_token_ids = soft_token_ids
        self.past_key_values = past_key_values
        self.loss_ids = loss_ids
        self.guid = guid
        self.tgt_text = tgt_text
        self.encoded_tgt_text = encoded_tgt_text
        self.use_cache = use_cache
        self.input_ids_len = input_ids_len
        self.conns_index = conns_index

        # 将不定长的键值对存入对象属性
        for k in kwargs.keys():
            setattr(self, k, kwargs[k])

    @classmethod
    def add_tensorable_keys(cls, *args):
        cls.tensorable_keys.extend(args)

    @classmethod
    def add_not_tensorable_keys(cls, *args):
        cls.not_tensorable_keys.extend(args)

    @classmethod
    def add_keys(cls, *args):
        cls.all_keys.extend(args)

    def __repr__(self):
        return str(self.to_json_string())

    def __len__(self):
        return len(self.keys())

    def to_tensor(self, device: str = 'cuda'):
        """就地操作，将所有可转换为tensor的特征转为torch.tensor格式"""
        for key in self.tensorable_keys:
            value = getattr(self, key)
            if value is not None:
                setattr(self, key, torch.tensor(value))
        return self

    def to(self, device: str = "cuda:0"):
        """将可转换的键移动到指定设备，如GPU"""
        for key in self.tensorable_keys:
            value = getattr(self, key)
            if value is not None:
                setattr(self, key, value.to(device))
        return self

    def cuda(self, device: str = "cuda:0"):
        """模拟tensor的cuda方法，将可转换的键移动到指定设备"""
        return self.to(device)

    def to_json_string(self, keep_none=False):
        """将当前实例序列化为JSON字符串格式"""
        data = {}
        for key in self.all_keys:
            value = getattr(self, key)
            # 将torch.Tensor转为列表，否则直接存储
            if isinstance(value, torch.Tensor):
                data[key] =  value.detach().cpu().tolist()
            elif value is None and keep_none:
                data[key] = None
            else:
                data[key] = value
        return json.dumps(data) + "\n"

    def keys(self, keep_none=False) -> List[str]:
        """返回InputFeatures的所有键，可选是否保留None值的键"""
        if keep_none:
            return self.all_keys
        else:
            return [key for key in self.all_keys if getattr(self, key) is not None]

    def to_dict(self, keep_none=False) -> Dict[str, Any]:
        """返回InputFeatures的键值映射字典"""
        data = {}
        for key in self.all_keys:
            value = getattr(self, key)
            if value is not None:
                data[key] =  value
            elif value is None and keep_none:
                data[key] = None
        return data

    def __getitem__(self, key):
        return getattr(self, key)

    def __iter__(self):
        return iter(self.keys())

    def __setitem__(self, key, item):
        if key not in self.all_keys:
            raise KeyError("Key {} not in predefined set of keys".format(key))
        setattr(self, key, item)

    def values(self, keep_none=False) -> List[Any]:
        """返回InputFeatures中与键对应的所有值"""
        return [getattr(self, key) for key in self.keys(keep_none=keep_none)]

    def __contains__(self, key, keep_none=False):
        return key in self.keys(keep_none)

    def items(self,):
        """返回InputFeatures中的（键, 值）对"""
        return [(key, self.__getitem__(key)) for key in self.keys()]

    @staticmethod
    def collate_fct(batch: List):
        """
        用于将输入特征整理成批处理格式的方法

        Args:
            batch (List[Union[Dict, InputFeatures]]): 当前数据的批次

        Returns:
            InputFeatures: 返回当前批次数据的InputFeatures
        """

        elem = batch[0]
        return_dict = {}
        for key in elem:
            # 针对编码的目标文本，直接添加进返回字典中
            if key == "encoded_tgt_text":
                return_dict[key] = [d[key] for d in batch]
            else:
                # 使用default_collate进行批处理
                try:
                    return_dict[key] = default_collate([d[key] for d in batch])
                except:
                    print(f"key{key}\n d {[batch[i][key] for i in range(len(batch))]} ")

        return InputFeatures(**return_dict)
