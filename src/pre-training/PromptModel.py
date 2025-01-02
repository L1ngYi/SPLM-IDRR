from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead, RobertaForMaskedLM
from torch.nn import CrossEntropyLoss  # 导入交叉熵损失函数
from transformers.modeling_outputs import ModelOutput  # 导入模型输出类
from typing import Optional, Tuple  # 导入类型注解
import torch.nn.functional as FF  # 导入PyTorch的功能性接口
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch  # 导入PyTorch主模块
import math  # 导入数学模块

# 定义输出类，包含模型的输出
class PromptMaskedLMOutput(ModelOutput):
    mutual_loss: Optional[torch.FloatTensor] = None  # 互信息损失
    loss: Optional[torch.FloatTensor] = None  # 总损失
    logits: torch.FloatTensor = None  # 预测的logits
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # 隐藏层状态
    attentions: Optional[Tuple[torch.FloatTensor]] = None  # 注意力权重

# 定义Scaled Dot Product Attention
class Attention(nn.Module):
    """
    计算'缩放点积注意力'
    """
    def forward(self, query, key, value, mask=None, dropout=None):
        # 计算注意力分数
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # 应用mask，将无效位置的分数设置为负无穷

        p_attn = FF.softmax(scores, dim=-1)  # 计算注意力权重

        if dropout is not None:
            p_attn = dropout(p_attn)  # 应用dropout

        return torch.matmul(p_attn, value), p_attn  # 返回加权值和注意力权重

# 定义多头注意力机制
class MultiHeadedAttention(nn.Module):
    """
    接受模型大小和头的数量
    """
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0  # 确保模型维度可以被头的数量整除

        # 假设d_v始终等于d_k
        self.d_k = d_model // h  # 每个头的维度
        self.h = h  # 头的数量

        # 创建线性层
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)  # 输出线性层
        self.attention = Attention()  # 注意力模块

        self.dropout = nn.Dropout(p=dropout)  # dropout层

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # 将相同的mask应用于所有头
            mask = mask.unsqueeze(1).repeat(1, self.h, 1).unsqueeze(2)

        batch_size = query.size(0)  # 获取批次大小

        # 1) 在批处理中进行所有线性变换
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) 在所有投影向量上应用注意力
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "拼接"并应用最终线性变换
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.output_linear(x)  # 返回最终输出

# 定义最大化互信息的判别器
class Max_Discriminator(nn.Module):
    """
    用于计算互信息最大化的判别器
    """
    def __init__(self, hidden, initrange=None):
        super().__init__()
        self.l1 = nn.Linear(2 * hidden, hidden)  # 第一层
        self.l2 = nn.Linear(hidden, hidden)  # 第二层
        self.l3 = nn.Linear(hidden, 1)  # 输出层
        self.act = FF.relu  # 激活函数

    def forward(self, x1, x2):
        h = torch.cat((x1, x2), dim=1)  # 拼接两个输入
        h = self.l1(h)  # 第一层
        h = self.act(h)  # 激活
        h = self.l2(h)  # 第二层
        h = self.act(h)  # 激活
        h = self.l3(h)  # 输出层
        return h  # 返回输出

# 定义带有提示功能的Roberta模型
class RobertaForPrompt(RobertaPreTrainedModel):
    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]  # 保存时忽略的键
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.weight", r"lm_head.decoder.bias"]  # 加载时缺失的键
    _keys_to_ignore_on_load_unexpected = [r"pooler"]  # 加载时意外的键

    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config, add_pooling_layer=False)  # 初始化Roberta模型
        self.lm_head = RobertaLMHead(config)  # 初始化语言模型头
        self.multi_head_attentention_pooling = MultiHeadedAttention(6, config.hidden_size)  # 初始化多头注意力池化
        self.max_d = Max_Discriminator(config.hidden_size)  # 初始化判别器
        
        # 只在语言模型头的权重与词嵌入绑定时需要特殊处理
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])
        self.init_weights()  # 初始化权重

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        conns_index=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            用于计算掩码语言建模损失的标签。索引应在``[-100, 0, ...,
            config.vocab_size]``（见``input_ids``文档字符串） 设置为``-100``的令牌被忽略
            （掩码），损失仅计算标签在``[0, ..., config.vocab_size]``中的令牌
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            用于隐藏已弃用的遗留参数。
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  # 确定返回字典格式

        # 获取Roberta模型的输出
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]  # 获取序列输出
        mutual_loss = None  # 初始化互信息损失
        masked_lm_loss = None  # 初始化掩码语言模型损失
         
        prediction_scores = self.lm_head(sequence_output)  # 获取预测分数
        
        # 全局逻辑语义增强
        if conns_index is not None:
            # 从序列输出中提取连接词的嵌入
            conn_embedding = torch.gather(sequence_output, 1, conns_index.unsqueeze(-1).expand(-1, -1, sequence_output.size(-1))).squeeze(1)
            new_mask = attention_mask.clone()  # 克隆注意力掩码
            zeros = torch.zeros((new_mask.size(0), new_mask.size(1)), dtype=torch.long, device=new_mask.device)  # 创建零张量
            new_mask.scatter_(1, conns_index, zeros)  # 在新掩码中更新连接词位置
            # 通过多头注意力池化获得逻辑语义
            logic_semantic = self.multi_head_attentention_pooling(conn_embedding, sequence_output, sequence_output, new_mask).squeeze(1)
            neg_logic_semantic = torch.cat((logic_semantic[1:], logic_semantic[0].unsqueeze(0)), dim=0)  # 创建负逻辑语义
            Ej = -FF.softplus(-self.max_d(conn_embedding, logic_semantic)).mean()  # 计算互信息的Ej部分
            Em = FF.softplus(self.max_d(conn_embedding, neg_logic_semantic)).mean()  # 计算互信息的Em部分
            mutual_loss = Em - Ej  # 计算互信息损失
            
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # 创建交叉熵损失函数
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))  # 计算掩码语言模型损失
        
        return PromptMaskedLMOutput(
            mutual_loss=mutual_loss,
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )  # 返回模型输出
