# 导入所需的模块
from abc import abstractmethod  # 导入用于定义抽象方法的模块
import json  # 导入json模块，用于处理JSON数据

from transformers.file_utils import ModelOutput  # 导入Transformers库中的ModelOutput类
from transformers.utils.dummy_pt_objects import PreTrainedModel  # 导入用于占位的预训练模型类

from .data_utils import InputFeatures, InputExample  # 导入自定义的InputFeatures和InputExample类
import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
from typing import *  # 导入所有类型注解
from transformers.tokenization_utils import PreTrainedTokenizer  # 导入Transformers库中的预训练Tokenizer类

import numpy as np  # 导入NumPy库
import torch.nn.functional as F  # 导入PyTorch的函数接口模块

'''
相比框架，删减了两个from_config函数。两个函数好像都是从config中加载一些东西的，可能作者直接重写了而不是在配置中写
incorporate_text_example 函数做了修改
process_outputs 函数 做了修改
'''



# 定义一个名为Template的基础类，继承自nn.Module
class Template(nn.Module):
    r'''
    所有模板类的基类。
    大多数方法是抽象的，少数方法用于提供所有模板共享的通用方法，如``loss_ids``, ``save``, ``load``等。

    参数:
        tokenizer (:obj:`PreTrainedTokenizer`): 用于指定词汇表和分词策略的分词器。
        placeholder_mapping (:obj:`dict`): 一个字典，用于表示原始输入文本的占位符。
    '''
    
    # 定义模板类的注册输入标识符，保存了一些常见的标识符
    registered_inputflag_names = ["loss_ids", "shortenable_ids"]

    def __init__(self,
                 tokenizer: PreTrainedTokenizer,  # 初始化时传入一个分词器
                 placeholder_mapping: dict = {'<text_a>':'text_a','<text_b>':'text_b'},  # 占位符映射
                ):
        super().__init__()  # 调用父类的构造函数
        self.tokenizer = tokenizer  # 将分词器赋值给实例变量
        self.placeholder_mapping = placeholder_mapping  # 将占位符映射赋值给实例变量
        self._in_on_text_set = False  # 初始化一个标识符，用于避免递归调用

        self.mixed_token_start = "{"  # 定义混合token的开始标记
        self.mixed_token_end = "}"  # 定义混合token的结束标记

    # 获取模板的默认loss标记索引
    def get_default_loss_ids(self) -> List[int]:
        ''' 获取使用mask的模板的loss索引。 
        例如，当self.text是`'{"placeholder": "text_a"}. {"meta": "word"} is {"mask"}.'`时，输出为 `[0, 0, 0, 0, 1, 0]`。
        返回:
            :obj:`List[int]`: 一个包含[0,1]的整数列表：1表示被mask的token，0表示正常token。
        '''
        return [1 if 'mask' in d else 0 for d in self.text]  # 返回text中mask位置的标记

    # 获取模板的默认短可缩标记索引
    def get_default_shortenable_ids(self) -> List[int]:
        """每个模板需要shortenable_ids，表示模板中哪些部分可以被截断，以适应语言模型的最大序列长度。
        默认：输入文本是可缩短的，而模板文本和其他特殊token不可缩短。

        例如，当self.text是`'{"placeholder": "text_a"} {"placeholder": "text_b", "shortenable": False} {"meta": "word"} is {"mask"}.'`时，输出为 `[1, 0, 0, 0, 0, 0, 0]`。

        返回:
            :obj:`List[int]`: 一个包含[0,1]的整数列表：1表示可缩短的token，0表示不可缩短的token。
        """
        idx = []
        for d in self.text:  # 遍历模板中的文本
            if 'shortenable' in d:  # 如果包含shortenable字段
                idx.append(1 if d['shortenable'] else 0)  # 如果可以缩短，添加1；否则添加0
            else:
                idx.append(1 if 'placeholder' in d else 0)  # 如果是占位符，添加1
        return idx  # 返回所有标记的索引列表

    # 获取默认的软token索引
    def get_default_soft_token_ids(self) -> List[int]:
        r'''
        该函数用于识别哪些token是软token。

        有时候模板中的token不是来自词汇表，而是软token的序列。
        在这种情况下，你需要实现这个方法

        异常:
            NotImplementedError: 如果需要，应该将``soft_token_ids``添加到Template类的``registered_inputflag_names``中，并实现该方法。
        '''
        raise NotImplementedError  # 该方法尚未实现，需子类实现

    # 结合输入的文本示例，将示例的内容填入模板
    def incorporate_text_example(self,
                                 example: InputExample,  # 输入示例对象
                                 text = None,  # 可选的文本内容
                                ):
        if text is None:  # 如果没有传入文本内容，则使用模板的默认文本
            text = self.text.copy()
        else:
            text = text.copy()  # 复制传入的文本内容

        # 遍历文本中的每一项，进行占位符的替换  下面这一段被删改了一部分


        for i, d in enumerate(text):
            #if not callable(d.get("post_processing")):   #（原来有的）
            #    d["post_processing"] = eval(d.get("post_processing", 'lambda x:x'))      #（原来有的）       
            if 'placeholder' in d:
                text[i] = d["add_prefix_space"] + d.get("post_processing", lambda x:x)(getattr(example, d['placeholder']))  # 用示例中的数据替换占位符
            elif 'meta' in d:
                text[i] = d["add_prefix_space"] + d.get("post_processing", lambda x:x)(example.meta[d['meta']])  # 用元数据替换
            elif 'soft' in d:
                #text[i] = '';  # 未使用的软token  原文：
                text[i] = d["soft"]; # unused
            elif 'mask' in d:
                text[i] = '<mask>'  # 替换为mask标记
            elif 'special' in d:
                text[i] = d['special']  # 替换为特殊token
            elif 'text' in d:
                text[i] = d["add_prefix_space"] + d['text']  # 替换文本
            else:
                raise ValueError(f'无法解析{d}')  # 如果遇到无法处理的字段，抛出异常
        return text  # 返回处理后的文本

    # 检查模板格式是否正确
    def _check_template_format(self, ):
        r"""检查模板格式是否正确。
        TODO: 添加更多检查项
        """
        mask_num = 0  # 记录mask的数量
        for i, d in enumerate(self.text):
            if 'mask' in d:
                mask_num += 1  # 统计mask的位置

        if mask_num==0:
            raise RuntimeError(f"模板中没有找到'mask'的位置：{self.text}. 请检查!")  # 如果没有找到mask，抛出异常

    # 解析文本，返回一个字典列表
    def parse_text(self, text: str) -> List[Dict]:
        parsed = []  # 存储解析结果的列表
        i = 0  # 指针初始化
        while i < len(text):  # 遍历文本
            d = {"add_prefix_space": ' ' if (i > 0 and text[i-1] == ' ') else ''}  # 判断当前字符前是否是空格
            while i < len(text) and text[i] == ' ':  # 跳过空格
                d["add_prefix_space"] = ' '
                i = i + 1
            if i == len(text): break  # 如果遍历结束，则退出循环

            if text[i] != self.mixed_token_start:  # 如果当前字符不是混合token的开始
                j = i + 1
                while j < len(text):  # 寻找下一个混合token的开始
                    if text[j] == self.mixed_token_start:
                        break
                    j = j + 1
                d["text"] = text[i:j].rstrip(' ')  # 获取文本部分
                i = j

            else:  # 如果是混合token的开始
                j = i + 1  # 从当前字符的下一个字符开始
                mixed_token_cnt = 1  # 初始化混合token计数器，支持嵌套的token  { {} {} } nested support
                while j < len(text):  # 在文本中寻找匹配的结束token
                    if text[j] == self.mixed_token_end:  # 如果遇到结束token
                        mixed_token_cnt -= 1  # 计数器减一
                        if mixed_token_cnt == 0: break  # 如果计数器为0，说明结束token匹配，跳出循环
                    elif text[j] == self.mixed_token_start:  # 如果遇到开始token
                        mixed_token_cnt += 1  # 计数器加一，支持嵌套token的情况
                    j = j + 1  # 移动到下一个字符

                if j == len(text):  # 如果遍历结束仍然没有找到匹配的结束token
                    # 抛出异常，提示开始token没有找到对应的结束token
                    raise ValueError(f"mixed_token_start {self.mixed_token_start} at position {i} has no corresponding mixed_token_end {self.mixed_token_end}")

                # 从当前位置i+1到j的部分是混合token，去掉开始和结束符号并形成一个新的字符串字典
                dict_str = '{'+text[i+1:j]+'}'  # 构造字典字符串，注意去掉开始和结束的标记符号
                try:
                    val = eval(dict_str)  # 使用eval解析字符串，将其转换为字典
                    if isinstance(val, set):  # 如果解析结果是集合
                        val = {k: None for k in val}  # 将集合转换为字典，键为集合元素，值为None
                    d.update(val)  # 将解析出的字典与当前字典d合并
                except:  # 如果解析过程中出现异常
                    import traceback  # 导入traceback模块，用于输出错误信息
                    print(traceback.format_exc())  # 输出详细的异常堆栈信息
                    print(f"syntax error in {dict_str}")  # 提示错误的字典字符串
                    exit()  # 退出程序

                i = j + 1  # 更新i为j+1，继续处理下一个token

            parsed.append(d)  # 将当前处理的字典d添加到parsed列表中

        return parsed  # 返回最终的解析结果列表
    # @abstractmethod
    def wrap_one_example(self,
                         example: InputExample) -> List[Dict]:
        r'''给定一个包含输入文本的示例，该文本可以通过self.template.placeholder_mapping的值进行引用。
        此函数将该示例处理为一个字典列表，
        每个字典作为一个组，包含样本属性，例如是否可简化、是否为遮盖位置、是否为软token等。
        由于文本将在随后的处理过程中被标记化，这些属性将会传播到标记化后的句子中。

        参数:
            example (:obj:`InputExample`): 一个 :py:class:`~openprompt.data_utils.data_utils.InputExample` 对象，该对象应该有能够填充模板的属性。

        返回:
            :obj:`List[Dict]`: 一个字典列表，其长度与self.text相同，例如``[{"loss_ids": 0, "text": "It was"}, {"loss_ids": 1, "text": "<mask>"}, ]``。
        '''
        
        if self.text is None:  # 如果模板的文本没有初始化
            raise ValueError("template text has not been initialized")  # 抛出错误，文本未初始化
        if isinstance(example, InputExample):  # 确保输入的example是InputExample类型
            text = self.incorporate_text_example(example)  # 将输入示例与文本模板结合，获取最终文本

            not_empty_keys = example.keys()  # 获取InputExample中所有的键
            for placeholder_token in self.placeholder_mapping:  # 遍历模板中的占位符映射
                not_empty_keys.remove(self.placeholder_mapping[placeholder_token])  # 删除已经被占位符处理过的键
            not_empty_keys.remove('meta')  # 删除已经处理过的meta键

            keys, values = ['text'], [text]  # 初始化键和值，'text'是最基本的键，text是相应的值
            for inputflag_name in self.registered_inputflag_names:  # 遍历所有注册的输入标志
                keys.append(inputflag_name)  # 将输入标志名称添加到keys中
                v = None  # 默认值为None
                if hasattr(self, inputflag_name) and getattr(self, inputflag_name) is not None:  # 如果模板对象有该标志，并且其值不为None
                    v = getattr(self, inputflag_name)  # 获取该标志的值
                elif hasattr(self, "get_default_" + inputflag_name):  # 如果模板对象有对应的默认值获取方法
                    v = getattr(self, "get_default_" + inputflag_name)()  # 调用默认值方法并获取值
                    setattr(self, inputflag_name, v)  # 缓存该值
                else:  # 如果既没有默认值，也没有初始化，抛出错误
                    raise ValueError("""
                    Template's inputflag '{}' is registered but not initialized.
                    Try using template.{} = [...] to initialize
                    or create a method get_default_{}(self) in your template.
                    """.format(inputflag_name, inputflag_name, inputflag_name))

                if len(v) != len(text):  # 检查标志的值的长度是否与文本的长度匹配
                    raise ValueError("Template: len({})={} doesn't match len(text)={}."\
                        .format(inputflag_name, len(v), len(text)))  # 如果不匹配，抛出错误
                values.append(v)  # 将值添加到values列表中

            wrapped_parts_to_tokenize = []  # 初始化一个空列表，用于存储需要被标记化的部分
            for piece in list(zip(*values)):  # 遍历values中的元素
                wrapped_parts_to_tokenize.append(dict(zip(keys, piece)))  # 将key和对应的piece组合成字典并添加到列表中

            wrapped_parts_not_tokenize = {key: getattr(example, key) for key in not_empty_keys}  # 生成一个字典，包含所有未被标记化的部分
            return [wrapped_parts_to_tokenize, wrapped_parts_not_tokenize]  # 返回包含标记化部分和未标记化部分的列表
        else:  # 如果example不是InputExample类型
            raise TypeError("InputExample")  # 抛出类型错误

    @abstractmethod
    def process_batch(self, batch):
        r"""模板应该重写此方法，如果需要处理批次输入，如替换嵌入向量。
        """
        return batch  # 默认返回原始batch，未处理

    def post_processing_outputs(self, outputs):
        r"""根据模板的需求对语言模型的输出进行后处理。
        大多数模板不需要后处理，
        但像SoftTemplate这样的模板，在将软模板作为模块附加到输入时（而不是作为输入标记的序列），
        应该删除这些位置的输出，以保持序列长度一致。
        """
        return outputs  # 默认返回未处理的输出

    def save(self,
             path: str,
             **kwargs) -> None:
        r'''
        保存模板的方法API。

        参数:
            path (str): 保存模板的路径。
        '''
        raise NotImplementedError  # 抛出未实现错误，要求子类实现具体的保存逻辑

    @property
    def text(self):
        return self._text  # 获取文本属性

    @text.setter
    def text(self, text):
        self._text = text  # 设置文本属性
        if text is None:  # 如果设置的文本为None
            return
        if not self._in_on_text_set:  # 如果当前没有在调用on_text_set方法
            self.safe_on_text_set()  # 调用safe_on_text_set来避免递归
        self._check_template_format()  # 检查模板格式

    def safe_on_text_set(self) -> None:
        r"""这个包装函数用于确保在``on_text_set()``中设置文本时不会再次触发``on_text_set()``，
            防止产生无限递归。
        """
        self._in_on_text_set = True  # 标记进入设置状态
        self.on_text_set()  # 调用on_text_set方法
        self._in_on_text_set = False  # 设置完成后标记退出状态
    @abstractmethod
    def on_text_set(self):
        r"""
        当模板文本被设置时，执行某些操作的钩子函数。
        模板的设计者应明确知道在模板文本设置时应当执行什么操作。
        """
        raise NotImplementedError  # 抛出未实现错误，要求子类实现此方法

    def from_file(self,
                  path: str,
                  choice: int = 0,
                 ):
        r'''
        从本地文件读取模板。

        参数:
            path (:obj:`str`): 本地模板文件的路径。
            choice (:obj:`int`): 文件中的第choice行。
        '''
        with open(path, 'r') as fin:  # 以只读模式打开文件
            text = fin.readlines()[choice].rstrip()  # 读取文件的第choice行，并去掉行尾的换行符
            #logger.info(f"using template: {text}")  原文有的
        self.text = text  # 将读取到的文本赋值给模板的text属性
        return self  # 返回当前对象，以便链式调用


    #这里被删掉了一个函数  from_config



class Verbalizer(nn.Module):
    r'''
    所有verbalizer的基类。

    参数:
        tokenizer (:obj:`PreTrainedTokenizer`): 一个分词器，用于指定词汇表和分词策略。
        classes (:obj:`Sequence[str]`): 一个包含需要映射的类的序列。
    '''
    def __init__(self,
                 tokenizer: Optional[PreTrainedTokenizer] = None,
                 classes: Optional[Sequence[str]] = None,
                 num_classes: Optional[int] = None,
                ):
        super().__init__()  # 调用父类的构造函数
        self.tokenizer = tokenizer  # 设置分词器
        self.classes = classes  # 设置类序列
        if classes is not None and num_classes is not None:  # 如果既有classes也有num_classes
            assert len(classes) == num_classes, "len(classes) != num_classes, Check you config."  # 确保classes的长度与num_classes一致
            self.num_classes = num_classes  # 设置num_classes为传入的值
        elif num_classes is not None:  # 如果只有num_classes
            self.num_classes = num_classes  # 直接设置num_classes
        elif classes is not None:  # 如果只有classes
            self.num_classes = len(classes)  # 根据classes的长度设置num_classes
        else:
            self.num_classes = None  # 如果没有提供classes和num_classes，num_classes为None
            # raise AttributeError("No able to configure num_classes")  # 可选，抛出错误提示

        self._in_on_label_words_set = False  # 初始化标记，表示当前是否正在设置label_words

    @property
    def label_words(self,):
        r'''
        label words是指通过标签投影到词汇表中的单词。
        例如，如果我们想在情感分类中建立投影：positive :math:`\rightarrow` {`wonderful`, `good`}，
        在这种情况下，`wonderful` 和 `good` 就是label words。
        '''
        if not hasattr(self, "_label_words"):  # 如果没有设置_label_words属性
            raise RuntimeError("label words haven't been set.")  # 抛出运行时错误
        return self._label_words  # 返回label_words属性

    @label_words.setter
    def label_words(self, label_words):
        if label_words is None:  # 如果传入的label_words是None
            return  # 不做处理
        self._label_words = self._match_label_words_to_label_ids(label_words)  # 调用函数匹配标签词与标签ID
        if not self._in_on_label_words_set:  # 如果当前没有在调用设置label_words的方法
            self.safe_on_label_words_set()  # 调用safe_on_label_words_set，避免无限递归

    def _match_label_words_to_label_ids(self, label_words):  # TODO: 新增的函数，文档已写，稍后重命名
        """
        排序label words字典，以匹配类的标签顺序
        """
        if isinstance(label_words, dict):  # 如果传入的是字典类型
            if self.classes is None:  # 如果classes未设置
                raise ValueError(""" 
                Verbalizer的classes属性必须设置，因为给定的label words是字典形式。
                我们需要根据类A的标签索引来匹配标签词。
                """)
            if set(label_words.keys()) != set(self.classes):  # 如果字典的键与类名不一致
                raise ValueError("verbalizer中的类名称与数据集中的类名称不一致")  # 抛出错误
            label_words = [  # 按照类的顺序对label_words进行排序
                label_words[c]
                for c in self.classes
            ]  # 返回按类顺序排序的标签词列表
        elif isinstance(label_words, list) or isinstance(label_words, tuple):  # 如果传入的是列表或元组类型
            pass  # 直接使用，不做任何修改
            # logger.info(""" 
            # 如果label_words是列表类型，默认情况下，列表中的第i个标签词将匹配数据集中的第i类。
            # 请确保它们具有相同的顺序。
            # 您也可以将label_words作为字典传入，映射类名到标签词。
            # """)
        else:  # 如果label_words既不是字典也不是列表或元组
            raise ValueError("Verbalizer的label words必须是列表、元组或字典类型")  # 抛出错误
        return label_words  # 返回排序后的标签词

    def safe_on_label_words_set(self,):
        self._in_on_label_words_set = True  # 设置标记为True，表示正在设置label_words
        self.on_label_words_set()  # 调用on_label_words_set方法
        self._in_on_label_words_set = False  # 设置标记为False，表示设置完成

    def on_label_words_set(self,):
        r"""当文本标签词被设置时执行的钩子函数。
        """
        pass  # 默认实现为空，子类可以重写此方法来定义具体行为
    @property
    def vocab(self,) -> Dict:
        # 定义一个属性vocab，用于获取词汇表
        if not hasattr(self, '_vocab'):  # 如果尚未计算vocab
            # 使用tokenizer将id映射到对应的tokens，构建vocab
            self._vocab = self.tokenizer.convert_ids_to_tokens(np.arange(self.vocab_size).tolist())
        return self._vocab  # 返回词汇表

    @property
    def vocab_size(self,) -> int:
        # 定义一个属性vocab_size，用于获取词汇表的大小
        return self.tokenizer.vocab_size  # 直接返回tokenizer中的vocab_size属性

    @abstractmethod
    def generate_parameters(self, **kwargs) -> List:
        r"""
        verbalizer可以看作是原始预训练模型上的一个额外层。
        在手动verbalizer中，它是一个固定的one-hot向量，维度为vocab_size，
        标签词的位置为1，其他地方为0。
        在其他情况下，参数可以是一个连续的向量，表示每个token的权重。
        此外，参数可以设置为可训练的，以便选择标签词。

        因此，此方法作为抽象方法，用于生成verbalizer的参数，必须在任何派生类中实现。

        注意：参数需要注册为pytorch模块的一部分，
        可以通过将张量包装为``nn.Parameter()``来实现。
        """
        raise NotImplementedError  # 抛出未实现错误，要求子类实现该方法

    def register_calibrate_logits(self, logits: torch.Tensor):
        r"""
        该函数用于注册需要进行校准的logits，并将原始的logits从当前计算图中分离出来。
        """
        if logits.requires_grad:  # 如果logits需要梯度计算
            logits = logits.detach()  # 将logits从计算图中分离
        self._calibrate_logits = logits  # 将logits保存为_calibrate_logits属性

    def process_outputs(self,   #这个函数被改了
                       outputs: torch.Tensor,
                       conn_linear_logits = None, 
                       **kwargs):
        r"""默认情况下，verbalizer将处理PLM（预训练语言模型）的输出logits。

        参数:
            outputs (:obj:`torch.Tensor`): 由预训练语言模型生成的logits。
            conn_linear_logits (:obj:`torch.Tensor`): 可选的附加logits，用于连接线性层。
            kwargs: 其他参数

        原函数：
           return self.process_logits(outputs, batch=batch, **kwargs)
        """
        if conn_linear_logits != None:  # 如果提供了连接线性层的logits
            return self.process_logits(outputs, conn_linear_logits, **kwargs)  # 调用process_logits处理
        else:
            return self.process_logits(outputs, **kwargs)  # 否则只处理outputs

    def gather_outputs(self, outputs: ModelOutput):
        r""" 从整个模型输出中检索对verbalizer有用的输出
        默认情况下，它只会检索logits。

        参数:
            outputs (:obj:`ModelOutput`): 预训练语言模型的输出。

        返回:
            :obj:`torch.Tensor`: 提取出的有用输出，形状应为（batch_size，seq_len，任何）
        """
        return outputs.logits  # 直接返回模型输出中的logits部分

    @staticmethod
    def aggregate(label_words_logits: torch.Tensor) -> torch.Tensor:
        r""" 对多个标签词的logits进行聚合，得到标签的logits
        基本聚合方法：对每个标签词的logits取平均，得到标签的logits
        在高级verbalizer中可以重新实现此方法。

        参数:
            label_words_logits (:obj:`torch.Tensor`): 只有标签词的logits。

        返回:
            :obj:`torch.Tensor`: 由标签词计算得到的最终logits。
        """
        if label_words_logits.dim() > 2:  # 如果label_words_logits的维度大于2
            return label_words_logits.mean(dim=-1)  # 对最后一个维度取平均，进行聚合
        else:
            return label_words_logits  # 否则直接返回标签词logits

    def normalize(self, logits: torch.Tensor) -> torch.Tensor:
        r"""
        给定关于整个词汇表的logits，使用softmax计算标签词集合的概率分布。

        参数:
            logits(:obj:`Tensor`): 整个词汇表的logits。

        返回:
            :obj:`Tensor`: 在标签词集合上的概率分布（和为1）。
        """
        batch_size = logits.shape[0]  # 获取批次大小
        return F.softmax(logits.reshape(batch_size, -1), dim=-1).reshape(*logits.shape)  # 通过softmax归一化logits，返回归一化后的tensor

    @abstractmethod
    def project(self,
                logits: torch.Tensor,
                **kwargs) -> torch.Tensor:
        r"""该方法接收形状为``[batch_size, vocab_size]``的输入logits，并使用
        verbalizer的参数将logits从整个词汇表投影到标签词的logits上。

        参数:
            logits (:obj:`Tensor`): 预训练语言模型生成的整个词汇表的logits，形状为[``batch_size``，``max_seq_length``，``vocab_size``]

        返回:
            :obj:`Tensor`: 每个标签的归一化概率（和为1）。
        """
        raise NotImplementedError  # 抛出未实现错误，要求子类实现此方法
    def handle_multi_token(self, label_words_logits, mask):
        r"""
        支持多种方法处理由tokenizer生成的多token情况。
        如果tokenization的一部分没有意义，建议使用'first'或'max'。
        该方法支持广播到三维张量。

        参数:
            label_words_logits (:obj:`torch.Tensor`): 由模型输出的logits。

        返回:
            :obj:`torch.Tensor`: 经过处理后的logits。
        """
        if self.multi_token_handler == "first":
            # 如果选择了'first'，只保留每个标签的第一个token的logits
            label_words_logits = label_words_logits.select(dim=-1, index=0)
        elif self.multi_token_handler == "max":
            # 如果选择了'max'，通过mask将无效的logits设置为非常小的值（-1000），然后取每个标签的最大logits
            label_words_logits = label_words_logits - 1000 * (1 - mask.unsqueeze(0))  # 用-1000填充mask为0的部分
            label_words_logits = label_words_logits.max(dim=-1).values  # 对每个标签，取最大值
        elif self.multi_token_handler == "mean":
            # 如果选择了'mean'，通过mask对logits进行加权平均
            label_words_logits = (label_words_logits * mask.unsqueeze(0)).sum(dim=-1) / (mask.unsqueeze(0).sum(dim=-1) + 1e-15)
            # 用mask加权logits，对所有token的logits求和，除以mask的总和
        else:
            # 如果multi_token_handler的值不是'first'、'max'或'mean'，抛出异常
            raise ValueError("multi_token_handler {} not configured".format(self.multi_token_handler))
        return label_words_logits  # 返回处理后的logits



    #被删了一个函数  from_config

    @classmethod
    def from_file(self,
                  path: str,
                  choice: Optional[int] = 0 ):
        r"""从文件加载预定义的标签词（verbalizer）。
        当前支持三种文件格式：
        1. .jsonl 或 .json 文件，文件中是单个以字典形式表示的verbalizer。
        2. .jsonl 或 .json 文件，文件中是多个以字典形式表示的verbalizer列表。
        3. .txt 或 .csv 文件，每行列出一个类的标签词，用逗号分隔。用空行开始新的verbalizer。
        如果不确定每个类的名称，推荐使用这种格式。

        verbalizer的详细格式可参考 :ref:`How_to_write_a_verbalizer`。

        参数:
            path (:obj:`str`): 本地文件路径。
            choice (:obj:`int`): 当文件包含多个verbalizer时，选择加载第几个。

        返回:
            Template : `self`对象
        """
        if path.endswith(".txt") or path.endswith(".csv"):
            # 如果文件是txt或csv格式
            with open(path, 'r') as f:
                lines = f.readlines()  # 读取文件的所有行
                label_words_all = []  # 存储所有的标签词
                label_words_single_group = []  # 临时存储单个标签组的标签词
                for line in lines:
                    line = line.strip().strip(" ")  # 去掉每行的前后空白字符
                    if line == "":  # 如果是空行，表示当前标签组结束
                        if len(label_words_single_group) > 0:
                            label_words_all.append(label_words_single_group)  # 将当前标签组添加到标签词集合
                        label_words_single_group = []  # 清空临时标签组
                    else:
                        label_words_single_group.append(line)  # 将当前行的标签词添加到标签组
                if len(label_words_single_group) > 0:  # 如果最后没有空行，仍需将最后一组标签添加
                    label_words_all.append(label_words_single_group)
                if choice >= len(label_words_all):  # 如果选择的verbalizer超出文件中verbalizer的数量
                    raise RuntimeError("choice {} exceed the number of verbalizers {}"
                                .format(choice, len(label_words_all)))

                label_words = label_words_all[choice]  # 获取指定的verbalizer
                label_words = [label_words_per_label.strip().split(",") \
                            for label_words_per_label in label_words]
                # 将每个标签词按逗号分割，并去除空格

        elif path.endswith(".jsonl") or path.endswith(".json"):
            # 如果文件是json或jsonl格式
            with open(path, "r") as f:
                label_words_all = json.load(f)  # 读取json文件
                # 如果文件中包含多个verbalizer
                if isinstance(label_words_all, list):
                    if choice >= len(label_words_all):  # 如果选择的verbalizer超出列表范围
                        raise RuntimeError("choice {} exceed the number of verbalizers {}"
                                .format(choice, len(label_words_all)))
                    label_words = label_words_all[choice]  # 获取指定的verbalizer
                elif isinstance(label_words_all, dict):
                    label_words = label_words_all  # 如果文件中只有一个verbalizer，直接使用它
                    if choice > 0:
                        print("Choice of verbalizer is 1, but the file  \
                        only contains one verbalizer.")  # 如果选择的verbalizer索引大于0，但文件只有一个verbalizer，打印警告

        self.label_words = label_words  # 将标签词赋值给实例的label_words属性
        if self.num_classes is not None:  # 如果num_classes已定义
            num_classes = len(self.label_words)  # 获取标签词的数量
            assert num_classes == self.num_classes, 'verbalizer文件中的类别数与预定义的num_classes不匹配。'  # 验证类别数是否一致
        return self  # 返回当前对象
