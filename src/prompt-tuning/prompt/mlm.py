from transformers.models.auto.tokenization_auto import tokenizer_class_from_name
# 导入transformers库中的tokenizer_class_from_name函数，用于根据模型名称获取相应的分词器类

from .utils import TokenizerWrapper
# 从当前模块的utils文件中导入TokenizerWrapper类，这个类是分词器包装类的父类

from typing import List, Dict
from collections import defaultdict
# 引入类型提示工具，defaultdict用作初始化时自动填充默认值的字典类型

'''
相比原文，多了一句这个
 # 获取掩码标记在input_ids中的索引，存储在conns_index字段中
 encoder_inputs['conns_index'] = [encoder_inputs['input_ids'].index(self.tokenizer.mask_token_id)]

'''




# 定义MLMTokenizerWrapper类，继承自TokenizerWrapper类，用于包装分词器
class MLMTokenizerWrapper(TokenizerWrapper):
    # 定义要添加的输入键列表，包括'input_ids'、'attention_mask'、'token_type_ids'和'conns_index'
    add_input_keys = ['input_ids', 'attention_mask', 'token_type_ids', 'conns_index']

    # 定义mask_token属性，用于获取当前模型的掩码标记
    @property
    def mask_token(self):
        return self.tokenizer.mask_token

    # 定义mask_token_ids属性，用于获取掩码标记的ID
    @property
    def mask_token_ids(self):
        return self.tokenizer.mask_token_id

    # 定义特殊标记数量属性num_special_tokens_to_add
    @property
    def num_special_tokens_to_add(self):
        if not hasattr(self, '_num_specials'):
            # 如果还未计算过，则调用分词器中的num_special_tokens_to_add()方法
            self._num_specials = self.tokenizer.num_special_tokens_to_add()
        return self._num_specials

    # 定义tokenize_one_example方法，用于对单个样本进行分词    #这个可能需要改，但是我还没找到这个函数的调用位置。应该不太可能是由框架自动调用
    def tokenize_one_example(self, wrapped_example, teacher_forcing):
        ''' # TODO doesn't consider the situation that input has two parts
        '''
        
        # 获取输入样本的包裹内容和其他附加信息
        wrapped_example, others = wrapped_example

        # encoded_tgt_text用于存储经过编码的目标文本，在一些任务（如SuperGLUE.COPA数据集）
        # 需要根据输入预测出答案或生成目标文本
        encoded_tgt_text = []
        if 'tgt_text' in others:
            tgt_text = others['tgt_text']
            if isinstance(tgt_text, str):
                tgt_text = [tgt_text]
            # 遍历每个目标文本并进行编码
            for t in tgt_text:
                encoded_tgt_text.append(self.tokenizer.encode(t, add_special_tokens=False))

        # mask_id用于记录当前填充的第几个掩码标记
        mask_id = 0 # the i-th the mask token in the template.

        # 初始化encoder_inputs字典，用于存储编码后的输入信息，包含多个子项
        encoder_inputs = defaultdict(list)
        for piece in wrapped_example:
            # 如果当前片段要求计算损失（loss_ids为1），则对其使用掩码标记
            if piece['loss_ids'] == 1:
                if teacher_forcing: # 使用teacher forcing时会触发异常，因为MLM不支持teacher forcing
                    raise RuntimeError("Masked Language Model can't perform teacher forcing training!")
                else:
                    encode_text = [self.mask_token_ids] # 使用掩码标记ID作为填充值
                mask_id += 1 # 递增掩码标记ID

            # 检查当前文本是否在special_tokens_maps中特殊标记映射中
            if piece['text'] in self.special_tokens_maps.keys():
                to_replace = self.special_tokens_maps[piece['text']]
                if to_replace is not None:
                    piece['text'] = to_replace # 替换为映射中的特殊标记
                else:
                    raise KeyError("This tokenizer doesn't specify {} token.".format(piece['text']))

            # 检查是否包含soft_token_ids键且不为0
            if 'soft_token_ids' in piece and piece['soft_token_ids'] != 0:
                encode_text = [0] # 可用任意标记替代，因为这些标记将用其自定义嵌入表示
            else:
                # 对文本进行编码，并去掉特殊标记
                encode_text = self.tokenizer.encode(piece['text'], add_special_tokens=False)

            # 获取编码后文本的长度
            encoding_length = len(encode_text)
            # 将编码后的文本添加到encoder_inputs中的input_ids
            encoder_inputs['input_ids'].append(encode_text)
            # 其他字段的值根据编码长度进行填充
            for key in piece:
                if key not in ['text']:
                    encoder_inputs[key].append([piece[key]] * encoding_length)

        # 调用truncate方法对输入进行截断操作
        encoder_inputs = self.truncate(encoder_inputs=encoder_inputs)
        # 删除短可缩减ID
        encoder_inputs.pop("shortenable_ids")
        # 将片段合并为整体输入
        encoder_inputs = self.concate_parts(input_dict=encoder_inputs)
        # 添加特殊标记
        encoder_inputs = self.add_special_tokens(encoder_inputs=encoder_inputs)
        # 创建特殊输入ID，并设置attention_mask和token_type_ids
        encoder_inputs['attention_mask'] = [1] * len(encoder_inputs['input_ids'])
        if self.create_token_type_ids:
            encoder_inputs['token_type_ids'] = [0] * len(encoder_inputs['input_ids'])
        # 填充，保证输入达到最大长度
        encoder_inputs = self.padding(input_dict=encoder_inputs, max_len=self.max_seq_length, pad_id_for_inputs=self.tokenizer.pad_token_id)
        
        # 获取掩码标记在input_ids中的索引，存储在conns_index字段中
        encoder_inputs['conns_index'] = [encoder_inputs['input_ids'].index(self.tokenizer.mask_token_id)]
        # 如果目标文本不为空，则将编码后的目标文本添加到encoder_inputs中
        if len(encoded_tgt_text) > 0:
            encoder_inputs = {**encoder_inputs, "encoded_tgt_text": encoded_tgt_text} # 将defaultdict转换为普通字典
        else:
            encoder_inputs = {**encoder_inputs}
        # 返回编码后的输入字典
        return encoder_inputs
