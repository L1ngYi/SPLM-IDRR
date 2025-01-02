from pickle import FALSE
from torch.utils.data.sampler import RandomSampler
from transformers.configuration_utils import PretrainedConfig
from transformers.generation_utils import GenerationMixin
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import *
from .data_utils import InputExample, InputFeatures
from torch.utils.data._utils.collate import default_collate
from tqdm.std import tqdm
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils.dummy_pt_objects import PreTrainedModel
from .utils import TokenizerWrapper
from .prompt_base import Template, Verbalizer
from collections import defaultdict
import inspect
from collections import namedtuple
import numpy as np
from torch.utils.data import DataLoader
'''
原框架中有这个class,它自己重写了两个forward函数
'''




def signature(f):   ###这函数是utils/utils.py中的一段
    """
    获取函数f的输入参数信息, 这在一些需要将函数参数动态传入多个函数的场景非常实用

    参数:
        f (function): 需要获取输入参数的函数

    返回:
        namedtuple: 包含函数参数信息，包括args、default、varargs、keywords等
    """
    sig = inspect.signature(f)
    args = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    varargs = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.VAR_POSITIONAL
    ]
    varargs = varargs[0] if varargs else None
    keywords = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.VAR_KEYWORD
    ]
    keywords = keywords[0] if keywords else None
    defaults = [
        p.default for p in sig.parameters.values()
        if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        and p.default is not p.empty
    ] or None
    argspec = namedtuple('Signature', ['args', 'defaults',
                                        'varargs', 'keywords'])
    return argspec(args, defaults, varargs, keywords) 

class PromptDataLoader(object):
    """
    PromptDataLoader封装了原始数据集, 首先使用模板处理数据, 然后通过封装的tokenizer进行分词操作.

    参数:
        dataset (Dataset or List): 可以是Dataset对象或包含输入数据的列表.
        template (Template): 模板，用于包装输入数据.
        tokenizer (PretrainedTokenizer): 预训练的分词器.
        tokenizer_wrapper_class (TokenizerWrapper): 封装分词器的类.
        max_seq_length (int, optional): 最大的序列长度，用于截断句子.
        batch_size (int, optional): 批次大小.
        teacher_forcing (bool, optional): 是否使用教师强制策略，训练生成模型时设置为True.
        decoder_max_length (int, optional): 编码器-解码器模型的解码器最大长度.
        predict_eos_token (bool, optional): 是否预测<eos>符号，生成任务建议设置为True.
        truncate_method (str, optional): 截断方法，可以选择 'head', 'tail', 'balanced' 等.
        kwargs : 传递给tokenizer封装器的其他参数.
    """
    def __init__(self,
                 dataset: Union[Dataset, List],
                 template: Template,
                 tokenizer_wrapper: Optional[TokenizerWrapper] = None,
                 tokenizer: PreTrainedTokenizer = None,
                 tokenizer_wrapper_class = None,
                 verbalizer: Optional[Verbalizer] = None,
                 max_seq_length: Optional[str] = 512,
                 batch_size: Optional[int] = 1,
                 shuffle: Optional[bool] = False,
                 teacher_forcing: Optional[bool] = False,
                 decoder_max_length: Optional[int] = -1,
                 predict_eos_token: Optional[bool] = False,
                 truncate_method: Optional[str] = "tail",
                 drop_last: Optional[bool] = False,
                 **kwargs,
                ):

        assert hasattr(dataset, "__iter__"), f"The dataset must have __iter__ method. dataset is {dataset}"
        assert hasattr(dataset, "__len__"), f"The dataset must have __len__ method. dataset is {dataset}"
        self.raw_dataset = dataset

        # 初始化wrapped_dataset和tensor_dataset用于保存包装后的数据和张量化的数据
        self.wrapped_dataset = []
        self.tensor_dataset = []
        self.template = template
        self.verbalizer = verbalizer
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.teacher_forcing = teacher_forcing

        if tokenizer_wrapper is None:
            if tokenizer_wrapper_class is None:
                raise RuntimeError("Either wrapped_tokenizer or tokenizer_wrapper_class should be specified.")
            if tokenizer is None:
                raise RuntimeError("No tokenizer specified to instantiate tokenizer_wrapper.")

            # 准备初始化tokenizer_wrapper的参数
            tokenizer_wrapper_init_keys = signature(tokenizer_wrapper_class.__init__).args
            prepare_kwargs = {
                "max_seq_length" : max_seq_length,
                "truncate_method" : truncate_method,
                "decoder_max_length" : decoder_max_length,
                "predict_eos_token" : predict_eos_token,
                "tokenizer" : tokenizer,
                **kwargs,
            }

            to_pass_kwargs = {key: prepare_kwargs[key] for key in prepare_kwargs if key in tokenizer_wrapper_init_keys}
            self.tokenizer_wrapper = tokenizer_wrapper_class(**to_pass_kwargs)
        else:
            self.tokenizer_wrapper = tokenizer_wrapper

        # 检查template是否具有wrap_one_example方法
        assert hasattr(self.template, 'wrap_one_example'), "Your prompt has no function variable \
                                                         named wrap_one_example"

        # 开始处理数据
        self.wrap()
        self.tokenize()

        # 如果需要随机打乱数据，则使用RandomSampler
        if self.shuffle:
            sampler = RandomSampler(self.tensor_dataset)
        else:
            sampler = None

        # 使用DataLoader加载数据，InputFeatures的collate_fct负责将数据整理为一个batch
        self.dataloader = DataLoader(
            self.tensor_dataset,
            batch_size = self.batch_size,
            sampler= sampler,
            collate_fn = InputFeatures.collate_fct,
            drop_last = drop_last,
        )


    def wrap(self):
        """ 
        包装函数，将数据集中的每一个示例都传入模板中进行包装。
        """
        if isinstance(self.raw_dataset, Dataset) or isinstance(self.raw_dataset, List):
            assert len(self.raw_dataset) > 0, 'The dataset to be wrapped is empty.'
            for idx, example in enumerate(self.raw_dataset):
                if self.verbalizer is not None and hasattr(self.verbalizer, 'wrap_one_example'): # 如果verbalizer具有wrap_one_example方法，调用此方法
                    example = self.verbalizer.wrap_one_example(example)
                wrapped_example = self.template.wrap_one_example(example)  # 使用模板包装示例
                self.wrapped_dataset.append(wrapped_example)  # 将包装后的示例添加到wrapped_dataset中
        else:
            raise NotImplementedError

    def tokenize(self) -> None:
        """ 
        对wrapped_dataset进行分词操作，结果存入tensor_dataset中。
        """
        for idx, wrapped_example in tqdm(enumerate(self.wrapped_dataset),desc='tokenizing'):
            inputfeatures = InputFeatures(**self.tokenizer_wrapper.tokenize_one_example(wrapped_example, self.teacher_forcing), **wrapped_example[1]).to_tensor()
            self.tensor_dataset.append(inputfeatures)

    def __len__(self):
        return  len(self.dataloader)

    def __iter__(self,):
        return self.dataloader.__iter__()
class PromptModel(nn.Module):
    r'''``PromptModel`` 是 ``Template`` 和 ``pre-trained model`` 的封装类，
    使用 OpenPrompt，可以灵活地组合这些模块。该类是 ``PromptForClassification`` 和 ``PromptForGeneration`` 的基类。

    参数:
        plm (:obj:`PreTrainedModel`): 用于当前 prompt-learning 任务的预训练语言模型。
        template (:obj:`Template`): ``Template`` 对象，用于包装输入数据。
        freeze_plm (:obj:`bool`): 是否冻结预训练语言模型的参数。
        plm_eval_mode (:obj:`bool`): 比 freeze_plm 更强的冻结模式，关闭模型的 dropout，无论其他部分是否设置为训练模式。
    '''
    def __init__(self,
                 plm: PreTrainedModel,
                 template: Template,
                 freeze_plm: bool = False,
                 plm_eval_mode: bool=False,
                ):
        super().__init__()
        self.plm = plm  # 赋值预训练模型
        self.template = template  # 赋值模板对象
        self.freeze_plm = freeze_plm  # 赋值是否冻结模型
        self.plm_eval_mode = plm_eval_mode  # 赋值是否采用强冻结模式
        
        # 如果需要冻结模型参数，将 plm 中的所有参数 requires_grad 设置为 False
        if freeze_plm:
            for param in self.plm.parameters():
                param.requires_grad = False
        
        # 如果 plm_eval_mode 为 True，将模型设置为 eval 模式，并冻结所有参数
        if plm_eval_mode:
            self.plm.eval()
            for param in self.plm.parameters():
                param.requires_grad = False

        # 获取模型前向方法中的参数关键词
        self.forward_keys = signature(self.plm.forward).args

        # 初始化模型的主输入名
        self._prepare_main_input_name()

    def _prepare_main_input_name(self):
        '''
        检查模型的 main_input_name 属性，优先从 encoder 获取，否则从模型本体获取。
        如果模型未定义 main_input_name,则使用默认值 "input_ids"。
        将最终结果保存到实例属性 self.main_input_name。
        '''
        # 获取模型的主输入名（如 input_ids）
        model = self.plm
        if hasattr(model, "encoder") and hasattr(model.encoder, "main_input_name"):
            print("get main_input_name and encoder. checked at pipeline_base ")#l1ngyi :在这里上个log，看看有没有覆写的encoder
            if model.encoder.main_input_name != model.main_input_name:
                main_input_name = model.encoder.main_input_name
            else:
                main_input_name = model.main_input_name
        else:
            main_input_name = getattr(model, "main_input_name", "input_ids")
        # 设置主输入名
        self.main_input_name = main_input_name


        

    def train(self, mode: bool = True):
        # 设置模型为训练或评估模式
        if not isinstance(mode, bool):
            raise ValueError("训练模式的参数应为布尔值")
        self.training = mode
        # 遍历模型的子模块，根据条件设置训练模式
        for name, module in self.named_children():
            if not (self.plm_eval_mode and 'plm' in name and mode):
                module.train(mode)
        return self

    def forward(self, input_ids, attention_mask, token_type_ids) -> torch.Tensor:  ##这个函数被改了。
        r"""
        这是前向方法，将包装的输入数据传入模型并返回输出 logits，通常用于预测 ``<mask>`` 位置的内容。

        参数:
            batch (:obj:`Union[Dict, InputFeatures]`): 包含数据序列的输入特征。

        返回:
            :obj:`torch.Tensor`: 处理后的输出。

        原函数写法：
        batch = self.template.process_batch(batch)
        input_batch = {key: batch[key] for key in batch if key in self.forward_keys}
        outputs = self.plm(**input_batch, output_hidden_states=True)
        outputs = self.template.post_processing_outputs(outputs)
        return outputs
        """
        # 将输入传入 plm 的 forward 方法，获取输出并启用隐藏状态输出
        outputs = self.plm(input_ids, attention_mask, token_type_ids, output_hidden_states=True)
        # 对输出进行模板后处理
        outputs = self.template.post_processing_outputs(outputs)
        return outputs

class PromptForClassification(nn.Module):
    r'''``PromptModel`` 在其基础上添加分类头。分类头将序列中所有位置的 logits（``PromptModel`` 的返回值）
    映射为标签的 logits，使用 verbalizer 对标签进行投影。

    参数:
        plm (:obj:`PretrainedModel`): 用于分类的预训练模型，如 BERT。
        template (:obj:`Template`): 用于包装分类输入文本的 ``Template`` 对象。
        verbalizer (:obj:`Verbalizer`): 用于将标签投影为标签词的 ``Verbalizer`` 对象。
        freeze_plm (:obj:`bool`): 是否冻结预训练模型的参数。
        plm_eval_mode (:obj:`bool`): 是否采用强冻结模式，关闭模型 dropout。
    '''
    def __init__(self,
                 plm: PreTrainedModel,
                 template: Template,  #定义在Prompt base里
                 verbalizer: Verbalizer,#定义在Prompt base里
                 freeze_plm: bool = False,
                 plm_eval_mode: bool=False,
                 multi_task: bool=False
                ):
        super().__init__()
        # 创建一个 PromptModel 实例
        self.prompt_model = PromptModel(plm, template, freeze_plm, plm_eval_mode)
        self.verbalizer = verbalizer  # 赋值 verbalizer 对象
        self.multi_task = multi_task  # 赋值是否多任务标志   #原来没有这一句

    @property
    def plm(self):
        # 返回 plm 模型
        return self.prompt_model.plm

    @property
    def template(self):
        # 返回模板对象
        return self.prompt_model.template

    @property
    def device(self,):
        r"""注册 device 参数。"""
        return self.plm.device  # 返回设备信息

    def extract_at_mask(self, outputs: torch.Tensor, loss_ids):
        r"""获取输出在 <mask> 位置的值，
        例如，将形状为（batch_size，max_seq_length，vocab_size）的 logits 映射到形状为
        （batch_size，num_mask_token，vocab_size）或（batch_size，vocab_size）。

        参数:
            outputs (:obj:`torch.Tensor`): 序列中所有位置的输出。
            loss_ids: 损失计算的 id

        返回:
            :obj:`torch.Tensor`: ``<mask>`` 位置的输出。
        """
        outputs = outputs[torch.where(loss_ids > 0)]
        outputs = outputs.view(loss_ids.shape[0], -1, outputs.shape[1])
        # 如果只有一个 mask 位置，调整形状
        if outputs.shape[1] == 1:
            outputs = outputs.view(outputs.shape[0], outputs.shape[2])
        return outputs

    def forward(self, input_ids, attention_mask, token_type_ids, loss_ids) -> torch.Tensor:  #如果你要改logits，应该是从这里下手 #这个函数被改动了

        r"""
        获取标签词的 logits。

        参数:
            batch (:obj:`Union[Dict, InputFeatures]`): 原始 batch 数据

        返回:
            :obj:`torch.Tensor`: 通过 verbalizer 获取的标签词 logits。


        原函数：
        outputs = self.prompt_model(batch)
        outputs = self.verbalizer.gather_outputs(outputs)
        if isinstance(outputs, tuple):
            outputs_at_mask = [self.extract_at_mask(output, batch) for output in outputs]
        else:
            outputs_at_mask = self.extract_at_mask(outputs, batch)
        label_words_logits = self.verbalizer.process_outputs(outputs_at_mask, batch=batch)
        return label_words_logits
        """
        # 获取 prompt_model 的输出
        outputs = self.prompt_model(input_ids, attention_mask, token_type_ids)#outputs = self.prompt_model(batch)
        linear_logits = None
        # 如果是多任务，记录线性层 logits
        if self.multi_task:
            linear_logits = outputs.linear_logits
            print("multi_task has been set")
        # 通过 verbalizer 收集输出
        outputs = self.verbalizer.gather_outputs(outputs)#outputs = self.verbalizer.gather_outputs(outputs)
        
        # 处理 outputs，如果是 tuple 结构，提取 mask 位置
        if isinstance(outputs, tuple):
            outputs_at_mask = [self.extract_at_mask(output, loss_ids) for output in outputs]# outputs_at_mask = [self.extract_at_mask(output, batch) for output in outputs]
        else:
            outputs_at_mask = self.extract_at_mask(outputs, loss_ids)#outputs_at_mask = self.extract_at_mask(outputs, batch)
        
        # 获取标签词 logits
        label_words_logits = self.verbalizer.process_outputs(outputs_at_mask)#label_words_logits = self.verbalizer.process_outputs(outputs_at_mask, batch=batch)

        # 返回标签词 logits，若多任务则返回额外的线性 logits
        if self.multi_task:
            return label_words_logits, linear_logits
        else:
            return label_words_logits

    @property
    def tokenizer(self):
        r'''便捷属性，用于更方便地获取 tokenizer'''
        return self.verbalizer.tokenizer

    def parallelize(self, device_map=None):
        r"""将模型并行化到多个设备上"""
        if hasattr(self.plm, "parallelize"):
            self.plm.parallelize(device_map)
            self.device_map = self.plm.device_map
            self.template.cuda()
            self.verbalizer.cuda()
        else:
            raise NotImplementedError("该 plm 不支持 parallelize 方法。")

    def deparallelize(self):
        r"""将模型的并行化解除"""
        if hasattr(self.plm, "deparallelize"):
            self.plm.deparallelize()
            self.device_map = None
            self.template.cpu()
            self.verbalizer.cpu()
        else:
            raise NotImplementedError("该 plm 不支持 deparallelize 方法。")
