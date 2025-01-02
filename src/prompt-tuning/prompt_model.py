# 从__future__导入，确保在Python 2.x中使用Python 3.x的功能
from __future__ import absolute_import  # 使导入语句符合Python 3.x的规范
from __future__ import division  # 确保除法操作符合Python 3.x的行为（返回浮点数）
from __future__ import print_function  # 确保print语法符合Python 3.x的规范

# 导入必要的PyTorch库
import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
import numpy as np  # 导入NumPy库
from prompt.pipeline_base import PromptForClassification  # 从prompt库导入分类模型类PromptForClassification

class PromptIDRC(nn.Module):
    """
    定义一个用于文本分类的模型类PromptIDRC，继承自PyTorch的nn.Module。
    """
    def __init__(self, prompt_config):
        """
        初始化PromptIDRC模型，使用给定的prompt_config配置加载分类模型。
        :param prompt_config: 包含配置的对象，提供模板、预训练模型和标签词汇器等信息
        """
        super(PromptIDRC, self).__init__()  # 调用父类nn.Module的构造函数进行初始化

        # 初始化分类模型PromptForClassification
        self.prompt_model = PromptForClassification(
            template=prompt_config.get_template(),  # 获取提示模板
            plm=prompt_config.get_plm(),  # 获取预训练语言模型
            verbalizer=prompt_config.get_verbalizer(),  # 获取标签词汇器
        )
        #self.plm = prompt_config.get_plm()

    def forward(self, input_ids, attention_mask, token_type_ids, loss_ids, label):
        """
        定义前向传播过程，输入包括input_ids、attention_mask、token_type_ids等，输出损失值和softmax后的logits。
        :param input_ids: 输入的token ID序列
        :param attention_mask: 用于指示哪些token是有效的mask
        :param token_type_ids: 用于区分不同句子的ID
        :param loss_ids: 用于计算损失的ID
        :param label: 真实标签
        :return: 损失值和softmax后的logits
        """
        # 使用prompt_model进行前向传播，得到logits
        logits = self.prompt_model(input_ids, attention_mask, token_type_ids, loss_ids)  #logits的计算也要改，但不在这里。

        # 计算损失值
        loss = self.calc_loss(logits, label)  

        # 返回损失值和softmax后的logits
        return loss, torch.nn.functional.softmax(logits, dim=-1)

    def calc_loss(self, logits, label): #损失函数应该会引入新的项，注意修改
        """
        计算交叉熵损失函数，比较预测结果logits和真实标签label之间的差异。
        :param logits: 模型预测的logits（未经softmax处理的原始输出）
        :param label: 真实标签
        :return: 计算出的损失值
        """
        # 计算标签的最大值索引，作为目标类别
        targets = torch.argmax(label, dim=-1)

        # 使用交叉熵损失函数
        loss_cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')  # 定义交叉熵损失，返回平均损失

        # 计算并返回损失
        loss = loss_cross_entropy(logits, targets)
        return loss
