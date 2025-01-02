from typing import List, Optional
from transformers.modeling_utils import PreTrainedModel
from .utils import TokenizerWrapper
from transformers.tokenization_utils import PreTrainedTokenizer
from .mlm import MLMTokenizerWrapper
from transformers import BertConfig, BertTokenizer, BertModel, BertForMaskedLM, \
                         RobertaConfig, RobertaTokenizer, RobertaModel, RobertaForMaskedLM
from collections import namedtuple


'''
get_model_class与其依赖的字典被删除
load_plm_from_config 被删除
load_plm的逻辑改动比较大
'''


# 定义一个名为ModelClass的元组，用于存储模型的相关配置类
ModelClass = namedtuple("ModelClass", ('config', 'tokenizer', 'model','wrapper'))

# 定义加载预训练语言模型(PLM)的函数
def load_plm(model_path, specials_to_add = None):
    r"""一个基于全局配置加载预训练语言模型的函数。它会同时加载模型、tokenizer和配置。

    Args:
        model_path (str): 预训练模型路径
        specials_to_add (List[str], optional): 需要添加的特殊符号列表。默认为None

    Returns:
        model: 加载的预训练模型
        tokenizer: 预训练模型的tokenizer
        model_config: 预训练模型的配置
        wrapper: 使用的wrapper类

    原函数；

    model_class = get_model_class(plm_type = model_name)
    model_config = model_class.config.from_pretrained(model_path)
    # you can change huggingface model_config here
    # if 't5'  in model_name: # remove dropout according to PPT~\ref{}
    #     model_config.dropout_rate = 0.0
    if 'gpt' in model_name: # add pad token for gpt
        specials_to_add = ["<pad>"]
        # model_config.attn_pdrop = 0.0
        # model_config.resid_pdrop = 0.0
        # model_config.embd_pdrop = 0.0
    model = model_class.model.from_pretrained(model_path, config=model_config)
    tokenizer = model_class.tokenizer.from_pretrained(model_path)
    wrapper = model_class.wrapper


    model, tokenizer = add_special_tokens(model, tokenizer, specials_to_add=specials_to_add)

    if 'opt' in model_name:
        tokenizer.add_bos_token=False
    return model, tokenizer, model_config, wrapper
    """
    # 从预训练路径加载Roberta模型的配置
    model_config = RobertaConfig.from_pretrained(model_path)

    # 加载基于Roberta模型的掩码语言模型(MLM)
    model = RobertaForMaskedLM.from_pretrained(model_path, config=model_config)

    # 加载Roberta模型的tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(model_path)

    # 使用MLMTokenizerWrapper作为模型的wrapper
    wrapper = MLMTokenizerWrapper  #这个好像还挺重要的，仔细看一下。引用自mlm.py

    # 调用add_special_tokens函数为模型和tokenizer添加特殊符号
    model, tokenizer = add_special_tokens(model, tokenizer, specials_to_add=specials_to_add)

    # 返回加载的模型、tokenizer、模型配置和wrapper
    return model, tokenizer, model_config, wrapper

# 定义用于添加特殊符号的函数
def add_special_tokens(model: PreTrainedModel,
                       tokenizer: PreTrainedTokenizer,
                       specials_to_add: Optional[List[str]] = None):
    r"""添加特殊符号到tokenizer，如果特殊符号不在现有tokenizer中。

    Args:
        model (PreTrainedModel): 预训练模型，用于在添加特殊符号后调整嵌入层大小
        tokenizer (PreTrainedTokenizer): 预训练模型的tokenizer，用于添加特殊符号
        specials_to_add (List[str], optional): 需要添加的特殊符号列表

    Returns:
        model: 添加特殊符号后调整过嵌入层大小的模型
        tokenizer: 添加了特殊符号的tokenizer
    """
    # 如果没有特殊符号需要添加，直接返回原始模型和tokenizer
    if specials_to_add is None:
        return model, tokenizer

    # 遍历每个需要添加的特殊符号
    for token in specials_to_add:
        # 如果符号中包含"pad"，并且当前tokenizer没有pad符号
        if "pad" in token.lower():
            if tokenizer.pad_token is None:
                # 添加pad符号到tokenizer中
                tokenizer.add_special_tokens({'pad_token': token})
                # 调整模型的嵌入层大小以适应新添加的符号
                model.resize_token_embeddings(len(tokenizer))
                #logger.info("pad token is None, set to id {}".format(tokenizer.pad_token_id)) 原函数有的
    # 返回调整后的模型和tokenizer
    return model, tokenizer
