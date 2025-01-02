import json
import math
import random
random.seed(42)  # 设置随机数种子，确保实验可复现
import os
import torch
from torch.utils.data import TensorDataset, DataLoader  # 导入TensorDataset用于打包数据，DataLoader用于批量加载数据


class GigaProcessor():
    def __init__(self, args, tokenizer):
        self.args = args  # 存储参数对象args，包含数据路径、批处理大小等
        self.raw_lines = []  # 初始化raw_lines为空列表，可能用来存储未处理的文本数据
        self.train_path = os.path.join(self.args.path_datasets, 'train.json')  # 获取训练集的路径
        self.test_path = os.path.join(self.args.path_datasets, 'test.json')  # 获取测试集的路径
        self.tokenizer = tokenizer  # 存储分词器对象tokenizer
        
        # 打开并加载训练集和测试集数据
        self.train_raw_data = self.open_file(self.train_path)
        self.test_raw_data = self.open_file(self.test_path)
        self.vocab_len = len(self.tokenizer.get_vocab())  # 获取分词器的词汇表长度
        # 特殊标记的ID，如[MASK]、[SEP]、[CLS]、[PAD]，用于后续操作时排除这些标记
        self.special_token_id = [self.tokenizer.mask_token_id, self.tokenizer.sep_token_id, self.tokenizer.cls_token_id,
                                 self.tokenizer.pad_token_id]

    # 获取数据加载器，根据传入的type参数（'train'或'test'）来处理相应数据
    def get_data_loader(self, type):
        if type == 'train':  # 若为训练集，则对训练数据进行mask处理
            input_ids, attention_mask, token_type_ids, labels, conns_index  = self.mask_data(self.train_raw_data, type)
        else:  # 否则处理测试数据
            input_ids, attention_mask, token_type_ids, labels, conns_index  = self.mask_data(self.test_raw_data, type)
        
        # 将数据转换为PyTorch张量（tensor）
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        conns_index = torch.tensor(conns_index, dtype=torch.long)
        
        # 使用TensorDataset打包成可迭代的数据集格式
        datasets = TensorDataset(input_ids, attention_mask, token_type_ids, labels, conns_index)
        
        # 根据数据类型（训练集或测试集）决定是否随机打乱数据
        if type == 'train':
            data_loader = DataLoader(datasets, shuffle=True, batch_size=self.args.batch_size)
        else:
            data_loader = DataLoader(datasets, shuffle=False, batch_size=self.args.batch_size)
        
        return data_loader  # 返回数据加载器


    # 对数据进行掩码处理，返回模型所需的各类输入特征
    def mask_data(self, data, type):
        input_ids = []  # 存储输入ID
        attention_mask = []  # 存储注意力掩码
        token_type_ids = []  # 存储token类型ID
        labels = []  # 存储标签
        conns_index = []  # 存储连接词的索引
        for d in data:  # 遍历每条数据，调用mask_single_data进行掩码处理
            format_data = self.mask_single_data(d)
            input_ids.append(format_data["mask_input_ids"])  # 添加处理后的输入ID
            attention_mask.append(format_data["attention_mask"])  # 添加注意力掩码
            token_type_ids.append(format_data["token_type_ids"])  # 添加token类型ID
            labels.append(format_data["label"])  # 添加标签
            conns_index.append(format_data["conn_index"])  # 添加连接词的索引
               
        return input_ids, attention_mask, token_type_ids, labels, conns_index  # 返回所有处理后的数据


    # 对单条句子数据进行掩码处理
    def mask_single_data(self, sent):         
        #L1ngYi 2024/10/21: 这里的sent_text已经是符合ClozePrompt template的了。因为explict_data中train.json的text已经是该模板的样子
        #L1ngYi 2024/10/21: 如果要pre-training阶段使用其他模板，可能可以在这里根据字符串特征重新切割组装，然后替换掉sent_text。目前尚未尝试更换后是否导致pretrain报错
        # "conn_index": 11,
        # "text": "It's absolutely wonderful for the first 15 minutes</s> then</s>It's, 'oh, my god, what do i do with it?'"
        sent_text = sent['text']  # 获取句子文本 
        sent_conn_index = sent['conn_index']  # 获取连接词的索引
        # 对句子进行分词、截断，设定最大长度，并填充至固定长度
        sent_encode_token = self.tokenizer(sent_text, truncation=True, max_length=self.args.sen_max_length, padding='max_length')
        label = [-100 for x in sent_encode_token['input_ids']]  # 初始化标签为-100（BERT的忽略标记）
        attention_mask = sent_encode_token['attention_mask']  # 获取注意力掩码
        token_type_ids = [0 for x in sent_encode_token['input_ids']]  # 初始化token类型ID为0
        raw_input_ids = [x for x in sent_encode_token['input_ids']]  # 获取原始输入ID

        # 如果启用了连接词掩码选项
        if self.args.connective_mask:
            label[sent_conn_index] = raw_input_ids[sent_conn_index]  # 将连接词所在位置的标签设为真实ID
            if random.random() < 0.9:  # 有90%的概率将连接词替换为[MASK]
                raw_input_ids[sent_conn_index] = self.tokenizer.mask_token_id
            
        # 如果启用了MLM（掩码语言模型）选项，则对其余的单词进行掩码操作
        if self.args.mlm:
            raw_input_ids, label = self.mlm(raw_input_ids, sent_conn_index, label) 
        
        # 打包返回格式化后的数据
        format_data = {
            "mask_input_ids": raw_input_ids,  # 掩码后的输入ID
            "attention_mask": attention_mask,  # 注意力掩码
            "token_type_ids": token_type_ids,  # token类型ID
            "label": label,  # 标签
            "conn_index": [sent_conn_index],  # 连接词的索引
        }
        return format_data  # 返回处理后的数据
        
    # 打开并加载JSON文件
    def open_file(self, path):
        with open(path) as f:
            data_lines = json.load(f)  # 使用json模块加载文件内容
        return data_lines  # 返回加载的文件数据


    # 掩码语言模型（MLM）的实现
    def mlm(self, encode_token, conn_index, label):
        # 遍历所有token，对每个token进行掩码处理，排除连接词
        for i in range(0, len(encode_token)):
            if i == conn_index:  # 跳过连接词的索引
                continue
            encode_token[i], label[i] = self.op_mask(encode_token[i])  # 对当前token进行掩码操作
        return encode_token, label  # 返回处理后的token和标签
    
    # 掩码操作的实现细节
    def op_mask(self, token):
        """
        Bert的原始掩码机制。
        (1) 有85%的概率，单词保持不变。
        (2) 有15%的概率，进行如下替换：
            - 80%的概率：将token替换为[MASK]。
            - 10%的概率：用词汇表中的随机token替换当前token。
            - 10%的概率：原始token保持不变。
        """
        if token in self.special_token_id:  # 如果token是特殊标记，跳过掩码处理
            return token, -100

        if random.random() <= 0.15:  # 有15%的概率执行掩码替换
            x = random.random()  # 生成随机数判断具体操作
            label = token  # 标签为当前token
            if x <= 0.80:  # 80%的概率，将token替换为[MASK]
                token = self.tokenizer.mask_token_id
            if x > 0.80 and x <= 0.9:  # 10%的概率，随机替换为词汇表中的其他token
                while True:
                    token = random.randint(0, self.vocab_len - 1)  # 随机选择一个词汇表中的token
                    if token not in self.special_token_id:  # 确保选择的token不是特殊标记
                        break

            return token, label  # 返回新的token及其对应的标签
        return token, -100  # 若不执行掩码，返回原始token，并忽略标签
    

    # 获取数据集的总长度（即训练集的批次数量）
    def get_len(self):
        return math.ceil((len(self.train_raw_data)) / (self.args.batch_size))  # 计算并返回总批次数量
