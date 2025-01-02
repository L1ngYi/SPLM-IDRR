# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes."""  # 这是代码的文件头，描述了这是一个实现了分词类的脚本。


#已纠正


from __future__ import absolute_import  # 兼容Python 2与3的导入
from __future__ import division  # 兼容Python 2与3的除法行为
from __future__ import print_function  # 兼容Python 2与3的print语句

import collections  # 导入collections模块，主要用于有序字典等数据结构
import unicodedata  # 导入unicodedata模块，处理Unicode字符
import six  # 导入six模块，Python2与Python3兼容工具

# 将文本转换为Unicode格式
def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""  # 将文本转换为Unicode，如果输入不是Unicode，假设为UTF-8编码
    if six.PY3:  # 如果是Python 3
        if isinstance(text, str):  # 如果是字符串类型
            return text  # 直接返回
        elif isinstance(text, bytes):  # 如果是字节类型
            return text.decode("utf-8", "ignore")  # 解码成Unicode，忽略错误字符
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))  # 如果既不是字符串也不是字节类型，抛出异常
    elif six.PY2:  # 如果是Python 2
        if isinstance(text, str):  # 如果是字符串类型
            return text.decode("utf-8", "ignore")  # 解码成Unicode
        elif isinstance(text, unicode):  # 如果是Unicode类型
            return text  # 直接返回
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))  # 抛出异常
    else:
        raise ValueError("Not running on Python2 or Python 3?")  # 如果既不是Python 2也不是Python 3，抛出异常


def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""  # 返回适合打印或日志的编码文本

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if six.PY3:  # 如果是Python 3
        if isinstance(text, str):  # 如果是字符串类型
            return text  # 直接返回
        elif isinstance(text, bytes):  # 如果是字节类型
            return text.decode("utf-8", "ignore")  # 解码成字符串
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))  # 如果既不是字符串也不是字节类型，抛出异常
    elif six.PY2:  # 如果是Python 2
        if isinstance(text, str):  # 如果是字符串类型
            return text  # 直接返回
        elif isinstance(text, unicode):  # 如果是Unicode类型
            return text.encode("utf-8")  # 编码成UTF-8字节
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))  # 抛出异常
    else:
        raise ValueError("Not running on Python2 or Python 3?")  # 如果既不是Python 2也不是Python 3，抛出异常


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""  # 加载词汇表文件，返回字典形式
    vocab = collections.OrderedDict()  # 创建一个有序字典，用于存储词汇表
    index_vocab = collections.OrderedDict()  # 创建一个有序字典，用于存储词汇表的反向映射
    index = 0  # 初始化索引
    with open(vocab_file, "rb") as reader:  # 以二进制方式打开词汇表文件
        while True:  # 持续读取文件
            tmp = reader.readline()  # 读取一行
            token = convert_to_unicode(tmp)  # 将行内容转换为Unicode格式

            if not token:  # 如果没有内容
                break  # 跳出循环
            
            #file_out.write("%d\t%s\n" %(index,token))
            token = token.strip()  # 去掉前后空格
            vocab[token] = index  # 将词汇和索引添加到词汇表字典
            index_vocab[index] = token  # 将索引和词汇添加到反向映射字典
            index += 1  # 索引递增

    return vocab, index_vocab  # 返回词汇表字典和反向映射字典


def convert_tokens_to_ids(vocab, tokens):
    """Converts a sequence of tokens into ids using the vocab."""  # 使用词汇表将一系列token转换为id
    ids = []  # 创建空列表存储id
    for token in tokens:  # 遍历每个token
        ids.append(vocab[token])  # 查找token对应的id，并添加到id列表
    return ids  # 返回id列表


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""  # 执行基本的空格清理和分割操作
    text = text.strip()  # 去除前后空格
    if not text:  # 如果文本为空
        return []  # 返回空列表
    tokens = text.split()  # 按空格分割文本为tokens
    return tokens  # 返回tokens列表


class FullTokenizer(object):
    """Runs end-to-end tokenization."""  # 完整的分词器，执行完整的分词过程

    def __init__(self, vocab_file, do_lower_case=True):
        """初始化分词器，加载词汇表，并创建基本分词器和WordPiece分词器"""
        self.vocab, self.index_vocab = load_vocab(vocab_file)  # 加载词汇表
        self.mask_token = 103  # 定义mask token的id（用于BERT模型）
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)  # 创建基本分词器
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)  # 创建WordPiece分词器

    def tokenize(self, text):
        """执行分词操作，将文本分成token"""
        split_tokens = []  # 存储分割出的tokens
        for token in self.basic_tokenizer.tokenize(text):  # 使用基本分词器分词
            for sub_token in self.wordpiece_tokenizer.tokenize(token):  # 对每个token进一步进行WordPiece分词
                split_tokens.append(sub_token)  # 将分词后的子token添加到分词列表中

        return split_tokens  # 返回分词结果

    def convert_tokens_to_ids(self, tokens):
        """将token列表转换为id列表"""
        return convert_tokens_to_ids(self.vocab, tokens)  # 使用词汇表将tokens转换为id

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        """生成特殊tokens的mask，返回一个0-1的列表"""

        """
从未添加特殊标记的标记列表中检索序列 ID。使用标记器 ``prepare_for_model`` 或 ``encode_plus`` 方法添加特殊标记时，将调用此方法。

参数：
token_ids_0：ID 列表（不得包含特殊标记）
token_ids_1：可选 ID 列表（不得包含特殊标记），在获取序列对的序列 ID 时必不可少
already_has_special_tokens：（默认为 False）如果标记列表已使用模型的特殊标记格式化，则设置为 True

返回：
范围为 [0, 1] 的整数列表：1 表示特殊标记，0 表示序列标记。
"""

        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.

        Args:
            token_ids_0: list of ids (must not contain special tokens)
            token_ids_1: Optional list of ids (must not contain special tokens), necessary when fetching sequence ids
                for sequence pairs
            already_has_special_tokens: (default False) Set to True if the token list is already formated with
                special tokens for the model

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        return [0] * ((len(token_ids_1) if token_ids_1 else 0) + len(token_ids_0))  # 返回一个全是0的mask列表


    def __len__(self):
        """返回词汇表的大小"""
        return len(self.vocab)  # 返回词汇表中词汇的数量


class BasicTokenizer(object):
    """执行基本的分词操作，如标点符号切分、大小写转换等。"""

    def __init__(self, do_lower_case=True):
        """初始化基本分词器

        Args:
          do_lower_case: 是否将输入文本转为小写
        """
        self.do_lower_case = do_lower_case  # 设置是否转换为小写

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = convert_to_unicode(text)  # 将文本转换为Unicode格式
        text = self._clean_text(text)  # 清理文本中的无效字符和空白字符
                # This was added on November 1st, 2018 for the multilingual and Chinese
            # models. This is also applied to the English models now, but it doesn't
            # matter since the English models were not trained on any Chinese data
            # and generally don't have any Chinese data in them (there are Chinese
            # characters in the vocabulary because Wikipedia does have some Chinese
            # words in the English Wikipedia.).
        # 这部分代码在2018年11月1日为多语言和中文模型添加。对于英语模型虽然也适用，但没有实际影响，
        # 因为英语模型并没有在中文数据上进行训练，通常也不包含中文数据（尽管维基百科有一些中文词汇会出现在英语维基百科中）。
        text = self._tokenize_chinese_chars(text)  # 对中文字符两边加空格
        orig_tokens = whitespace_tokenize(text)  # 按照空格进行初步分词
        split_tokens = []  # 用于存储最终的分词结果
        for token in orig_tokens:
            if self.do_lower_case:  # 如果设置了大小写转换
                token = token.lower()  # 将token转换为小写

                token = self._run_strip_accents(token)  # 去除token中的重音符号
            split_tokens.extend(self._run_split_on_punc(token))  # 根据标点符号拆分token并扩展到结果中

        output_tokens = whitespace_tokenize(" ".join(split_tokens))  # 将分割的token合并为字符串并重新按空格分词
        return output_tokens  # 返回分词结果

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)  # 使用NFD形式分解文本，NFD是标准化的形式，重音符号会与字母分离
        output = []  # 存储处理后的字符
        for char in text:
            cat = unicodedata.category(char)  # 获取字符的Unicode类别
            if cat == "Mn":  # 如果是重音符号（Mn类别），则跳过
                continue
            output.append(char)  # 否则，保留字符
        return "".join(output)  # 返回去除重音符号后的文本

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)  # 将文本转换为字符列表
        i = 0  # 初始化指针i
        start_new_word = True  # 是否开始一个新单词的标志
        output = []  # 存储分割结果
        while i < len(chars):  # 遍历每个字符
            char = chars[i]
            if _is_punctuation(char):  # 如果字符是标点符号
                output.append([char])  # 将标点符号作为单独的token加入
                start_new_word = True  # 下一个字符开始新的单词
            else:
                if start_new_word:  # 如果是新单词
                    output.append([])  # 开始新的token列表
                start_new_word = False  # 继续当前单词
                output[-1].append(char)  # 将字符添加到当前token列表中
            i += 1  # 移动指针到下一个字符

        return ["".join(x) for x in output]  # 将每个token合并为字符串，并返回

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []  # 存储处理后的字符
        for char in text:
            cp = ord(char)  # 获取字符的Unicode码点
            if self._is_chinese_char(cp):  # 如果是中文字符
                output.append(" ")  # 在中文字符前添加空格
                output.append(char)  # 添加中文字符
                output.append(" ")  # 在中文字符后添加空格
            else:
                output.append(char)  # 其他字符直接添加
        return "".join(output)  # 返回添加空格后的文本

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # 判断Unicode码点是否属于CJK（中日韩）字符集范围：
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  # CJK统一表意文字
            (cp >= 0x3400 and cp <= 0x4DBF) or  # CJK扩展A
            (cp >= 0x20000 and cp <= 0x2A6DF) or  # CJK扩展B
            (cp >= 0x2A700 and cp <= 0x2B73F) or  # CJK扩展C
            (cp >= 0x2B740 and cp <= 0x2B81F) or  # CJK扩展D
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  # CJK兼容表意文字
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  # CJK兼容扩展
            return True
        
        return False  # 如果不在上述范围内，返回False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []  # 存储处理后的字符
        for char in text:
            cp = ord(char)  # 获取字符的Unicode码点
            if cp == 0 or cp == 0xfffd or _is_control(char):  # 跳过无效字符和控制字符
                continue
            if _is_whitespace(char):  # 如果是空白字符
                output.append(" ")  # 替换为空格
            else:
                output.append(char)  # 其他字符直接添加
        return "".join(output)  # 返回清理后的文本

class WordpieceTokenizer(object):
    """执行WordPiece分词。"""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=100):
        """
        初始化WordpieceTokenizer对象。

        Args:
          vocab: 词汇表，包含所有词语或子词
          unk_token: 未知词标记，默认值为"[UNK]"
          max_input_chars_per_word: 每个词语的最大字符数，超过此长度的词会被标记为未知词
        """
        self.vocab = vocab  # 词汇表
        self.unk_token = unk_token  # 未知词标记
        self.max_input_chars_per_word = max_input_chars_per_word  # 每个词的最大字符数

    def tokenize(self, text):
        """将文本分解成单词片段（word pieces）。

        使用贪心算法（最长匹配优先）进行分词，依赖于给定的词汇表。

        示例:
          输入 = "unaffable"
          输出 = ["un", "##aff", "##able"]

        Args:
          text: 单个词或通过空格分隔的词列表，应该已经通过 `BasicTokenizer` 进行了预处理。

        Returns:
          返回一个WordPiece的token列表。
        """
        
        # 将文本转换为Unicode格式，保证编码一致性
        text = convert_to_unicode(text)

        output_tokens = []  # 存储最终的分词结果
        # 遍历文本中的每个词，进行分词处理
        for token in whitespace_tokenize(text):
            chars = list(token)  # 将token转换为字符列表
            # 如果字符数超过最大限制，则标记为未知词
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False  # 用于标记是否是无效词
            start = 0  # 当前词的起始位置
            sub_tokens = []  # 存储该词的子token

            # 使用最长匹配优先算法，从词的开头开始查找最大匹配的子token
            while start < len(chars):
                end = len(chars)  # 初始设置子串的结尾位置为整个词的末尾
                cur_substr = None  # 当前匹配到的子串

                # 从当前位置开始，逐步缩小子串的范围，寻找词汇表中存在的子词
                while start < end:
                    substr = "".join(chars[start:end])  # 获取当前子串
                    if start > 0:  # 如果不是第一个子串，添加"##"前缀
                        substr = "##" + substr
                    if substr in self.vocab:  # 如果子串在词汇表中
                        cur_substr = substr  # 记录该子串
                        break  # 找到匹配后退出循环
                    end -= 1  # 缩小子串范围

                # 如果没有找到匹配的子串，标记为无效词
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)  # 将匹配到的子串添加到子token列表
                start = end  # 更新起始位置，继续查找下一个子串

            # 如果是无效词，则使用未知词标记
            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)  # 否则，将所有子token添加到结果中
        return output_tokens  # 返回最终的分词结果


def _is_whitespace(char):
    """检查字符是否为空白字符。"""
    # \t, \n, 和 \r 在技术上是控制字符，但我们将其视为空白字符，因为它们通常被认为是空白字符。
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)  # 获取字符的Unicode类别
    if cat == "Zs":  # 判断是否为空格类字符（Zs类）
        return True
    return False  # 不是空白字符


def _is_control(char):
    """检查字符是否为控制字符。"""
    # \t, \n, 和 \r 在技术上是控制字符，但我们将其视为空白字符。
    if char == "\t" or char == "\n" or char == "\r":
        return False  # 这些字符被视为空白字符
    cat = unicodedata.category(char)  # 获取字符的Unicode类别
    if cat.startswith("C"):  # 如果类别是以"C"开头，则表示控制字符
        return True
    return False  # 不是控制字符


def _is_punctuation(char):
    """检查字符是否为标点符号。"""
    cp = ord(char)  # 获取字符的Unicode码点
    # 我们将所有非字母/数字的ASCII字符视为标点符号。
    # 如"^", "$", "`"这些符号并不属于Unicode标点类，但我们还是将它们视为标点符号。
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)  # 获取字符的Unicode类别
    if cat.startswith("P"):  # 如果类别是以"P"开头，则表示标点符号
        return True
    return False  # 不是标点符号
