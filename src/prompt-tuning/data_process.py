import tokenization_word as tokenization  # 导入自定义的tokenization模块
import os  # 导入操作系统接口模块，用于路径操作
from prompt.data_utils import InputExample  # 导入InputExample，用于存储输入示例
from prompt.pipeline_base import PromptDataLoader  # 导入PromptDataLoader，用于数据加载
import numpy as np  # 导入numpy，用于数值计算
import math  # 导入math模块，进行数学计算


class DataProcessor(object):
    """数据转换器基类，用于处理序列分类数据集。"""

    def get_examples(self, data_dir):
        """获取训练集的InputExample集合，需要子类实现。"""
        raise NotImplementedError()

    def get_labels(self):
        """获取数据集的标签列表，需要子类实现。"""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """读取制表符分隔的文件（tsv）。"""
        file_in = open(input_file, "rb")  # 打开输入文件，以二进制格式读取
        lines = []  # 初始化空列表，用于存储每一行数据
        for line in file_in:
            lines.append(line.decode("utf-8").split("\t"))  # 逐行读取并解码为utf-8，使用制表符分割
        return lines  # 返回文件中所有行的数据


class PromptProcessor(DataProcessor):
    """继承DataProcessor类，用于处理提示数据。"""
    
    def __init__(self):
        self.labels = set()  # 初始化一个空的集合，用于存储标签

    def get_examples(self, data_dir, data_type, num_rels):
        """See base class."""
        """获取数据集的示例，使用_create_examples方法创建样本。"""
        return self._create_examples(
            self._read_tsv(data_dir), data_type, num_rels)  # 调用_create_examples方法

    def get_labels(self):
        """See base class."""
        """获取数据集的标签列表。"""
        
        tmp = list(self.labels)  # 将标签集合转换为列表
        tmp.sort()  # 对标签列表进行排序
        return tmp  # 返回排序后的标签列表

    def _create_examples(self, lines, set_type, num_rels):
        """Creates examples for the training and dev sets."""
        """为训练集和开发集创建示例数据。"""
        examples = []  # 初始化空列表，用于存储所有的示例数据
        for (i, line) in enumerate(lines):  # 遍历每一行数据
            guid = "%s-%s" % (set_type, i)  # 生成每个示例的唯一ID
            text_a = tokenization.convert_to_unicode(line[0])  # 将第1列转换为unicode格式
            label = tokenization.convert_to_unicode(line[2])  # 将第3列转换为unicode格式作为标签
            text_b = tokenization.convert_to_unicode(line[1])  # 将第2列转换为unicode格式
            
            # 创建一个InputExample对象，将数据和标签打包
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b,
                             label=create_multi_label(label, num_rels)))  # 使用create_multi_label函数处理标签
        return examples  # 返回所有示例数据


class DataGenerate(object):
    """数据生成基类，定义获取训练集、开发集、测试集加载器的接口。"""

    def get_train_loader(self):
        raise NotImplementedError()  # 获取训练集加载器，子类需要实现

    def get_dev_loader(self):
        raise NotImplementedError()  # 获取开发集加载器，子类需要实现

    def get_test_loader(self):
        raise NotImplementedError()  # 获取测试集加载器，子类需要实现

    def get_labels(self):
        raise NotImplementedError()  # 获取标签，子类需要实现


class PromptDataGenerate(DataGenerate):
    """继承DataGenerate类，具体实现数据加载的逻辑。"""
    
    def __init__(self, args, prompt_config):
        """初始化数据生成器，包括路径和配置。"""
        self.args = args  # 保存参数配置
        self.processors = PromptProcessor()  # 创建PromptProcessor实例，用于数据处理
        self.train_path = os.path.join(args.data_dir, "train.tsv")  # 训练集路径
        self.dev_path = os.path.join(args.data_dir, "dev.tsv")  # 开发集路径
        self.test_path = os.path.join(args.data_dir, "test.tsv")  # 测试集路径
        self.blind_path = os.path.join(args.data_dir, "blind.tsv")  # Blind集路径
        self.num_train_steps = None  # 训练步数初始化为None
        self.test_data_loader = None  # 测试集数据加载器初始化为空
        self.train_data_loader = None  # 训练集数据加载器初始化为空
        self.dev_data_loader = None  # 开发集数据加载器初始化为空
        self.blind_data_loader = None  # Blind集数据加载器初始化为空
        self.tokenizer =  prompt_config.get_tokenizer()  # 获取tokenizer配置
        self.promptTemplate = prompt_config.get_template()  # 获取模板配置
        self.wrappeer_class = prompt_config.get_wrapperclass()  # 获取数据封装类

    def get_train_loader(self):
        """获取训练集数据加载器。"""
        train_examples = self.processors.get_examples(self.train_path, "train", self.args.num_rels)  # 获取训练集示例数据

        self.num_train_steps = len(train_examples)  # 设置训练步数
        self.train_data_loader = PromptDataLoader(
            dataset=train_examples,  # 训练集数据
            tokenizer=self.tokenizer,  # 使用的tokenizer
            template=self.promptTemplate,  # 模板
            max_seq_length=self.args.max_seq_length,  # 最大序列长度
            tokenizer_wrapper_class=self.wrappeer_class,  # 封装类
            create_token_type_ids=True,  # 是否创建token类型ID
            batch_size=self.args.train_batch_size,  # 批量大小
            shuffle=True  # 是否打乱数据
        )
        return self.train_data_loader  # 返回训练集数据加载器

    def get_dev_loader(self):
        """获取开发集数据加载器。"""
        if self.dev_data_loader is None:  # 如果开发集数据加载器为空
            dev_examples = self.processors.get_examples(self.dev_path, "dev", self.args.num_rels)  # 获取开发集示例数据
            self.dev_data_loader = PromptDataLoader(
                dataset=dev_examples,  # 开发集数据
                tokenizer=self.tokenizer,  # 使用的tokenizer
                template=self.promptTemplate,  # 模板
                max_seq_length=self.args.max_seq_length,  # 最大序列长度
                tokenizer_wrapper_class=self.wrappeer_class,  # 封装类
                create_token_type_ids=True,  # 是否创建token类型ID
                batch_size=self.args.dev_batch_size,  # 批量大小
                shuffle=False  # 不打乱数据
            )
        return self.dev_data_loader  # 返回开发集数据加载器
    
    def get_test_loader(self):
        """获取测试集数据加载器。"""
        if self.test_data_loader is None:  # 如果测试集数据加载器为空
            test_examples = self.processors.get_examples(self.test_path, "test", self.args.num_rels)  # 获取测试集示例数据
            self.test_data_loader = PromptDataLoader(
                dataset=test_examples,  # 测试集数据
                tokenizer=self.tokenizer,  # 使用的tokenizer
                template=self.promptTemplate,  # 模板
                max_seq_length=self.args.max_seq_length,  # 最大序列长度
                tokenizer_wrapper_class=self.wrappeer_class,  # 封装类
                create_token_type_ids=True,  # 是否创建token类型ID
                batch_size=self.args.test_batch_size,  # 批量大小
                shuffle=False  # 不打乱数据
            )
        return self.test_data_loader  # 返回测试集数据加载器
    
    def get_blind_loader(self):
        """获取Blind集数据加载器。"""
        if self.blind_data_loader is None:  # 如果Blind集数据加载器为空
            blind_examples = self.processors.get_examples(self.blind_path, "blind", self.args.num_rels)  # 获取Blind集示例数据
            self.blind_data_loader = PromptDataLoader(
                dataset=blind_examples,  # Blind集数据
                tokenizer=self.tokenizer,  # 使用的tokenizer
                template=self.promptTemplate,  # 模板
                max_seq_length=self.args.max_seq_length,  # 最大序列长度
                tokenizer_wrapper_class=self.wrappeer_class,  # 封装类
                create_token_type_ids=True,  # 是否创建token类型ID
                batch_size=self.args.test_batch_size,  # 批量大小
                shuffle=False  # 不打乱数据
            )
        return self.blind_data_loader  # 返回Blind集数据加载器


def create_multi_label(label_str, num_rels):
    """将标签字符串转换为多标签格式。"""
    label_list = label_str.split('#')  # 按照#符号分割标签
    if num_rels == 4:  # 如果关系数为4
        label_multi = [0, 0, 0, 0]  # 初始化4个标签的多标签格式
    elif num_rels == 11:  # 如果关系数为11
        label_multi = np.zeros(11, dtype=int)  # 初始化11个标签的多标签格式
    else:  # 其他情况，关系数为14
        label_multi = np.zeros(14, dtype=int)  # 初始化14个标签的多标签格式
    for idx in label_list:  # 遍历标签列表
        label_multi[int(idx)] = 1  # 将标签对应位置的值设为1
    return label_multi  # 返回多标签格式的标签
