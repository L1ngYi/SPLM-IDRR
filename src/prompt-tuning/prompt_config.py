from prompt import load_plm  # 导入load_plm函数，用于加载预训练语言模型
from prompt.manual_verbalizer import ManualVerbalizer  # 导入ManualVerbalizer类，用于手动创建标签词汇器
from prompt.ptuning_prompts import PtuningTemplate
import torch  # 导入PyTorch库，用于深度学习模型的加载和操作

class PromptConfig(object):
    """
    定义一个配置类，用于初始化和管理语言模型、tokenizer、提示模板和标签词汇器。
    """
    def __init__(self, args):
        """
        初始化函数，加载预训练模型、tokenizer，并设置提示模板和标签词汇器。
        :param args: 包含各种参数的配置对象
        """
        # 加载预训练模型，tokenizer，模型配置和包装类
        plm, tokenizer, model_config, WrapperClass = load_plm(args.pretrain_file)
        self.plm = plm  # 预训练语言模型
        self.tokenizer = tokenizer  # tokenizer，用于文本的编码和解码
        self.wrapperclass = WrapperClass  # 包装类，用于封装模型和tokenizer的交互

        # 如果使用预训练模型，则加载预训练模型的状态字典
        if args.use_pretrain:
            net_state = torch.load(args.plse_pretrain_file)['net']  # 加载预训练模型的参数
            self.plm.load_state_dict(net_state, strict=False)  # 将参数加载到模型中，允许部分匹配
        

        # 创建提示模板，定义输入格式
        # self.promptTemplate = ManualTemplate(
        #     text='{"placeholder":"text_a"}{"special": "</s>"}{"mask"}{"special": "</s>"}{"placeholder":"text_b"}',
        #     tokenizer=tokenizer
        # )
        """
        该模板表示：
        text_a 和 text_b 是输入文本，<mask> 是占位符，用于生成关系预测的提示
        </s> 是分隔符，用于分隔不同的文本部分
        """
        self.promptTemplate = PtuningTemplate(
            text='{"soft"}{"soft"}{"soft"}{"soft"}{"placeholder":"text_a"}{"mask"}{"placeholder":"text_b"}',
            tokenizer=tokenizer,
            model = self.plm
        )



        # 根据输入的关系数（num_rels）来设置类别和对应的标签词汇
        if args.num_rels == 4:
            # pdtb2 4分类任务
            classes = ["Comparision", "Contingency", "Expansion", "Temporal"]
            label_words = {
                "Comparision": ["but", "however", "although", "though"],  # 比较关系
                "Contingency": ["because", "so", "thus", "therefore", "consequently"],  # 假设关系
                "Expansion": ["instead", "rather", "and", "also", "furthermore", "example", "instance", "fact", "indeed", "particular", "specifically"],  # 扩展关系
                "Temporal": ["then", "before", "after", "meanwhile", "when"],  # 时间关系
            }
        elif args.num_rels == 11:
            # pdtb2 11分类任务
            classes = list(range(11))  # 11个类别
            label_words = {
                0: ['although', 'though', 'however'],  # 转折关系
                1: ["but"],  # 转折关系
                2: ["because", 'so', 'thus', 'consequently', "therefore"],  # 因果关系
                3: ['since', 'as'],  # 因果关系
                4: ["instead", 'rather'],  # 替代关系
                5: ["and", "also", "furthermore", 'fact'],  # 并列关系
                6: ["example", 'instance'],  # 示例关系
                7: ["finally"],  # 结论关系
                8: ["specifically", 'particular', 'indeed'],  # 具体化关系
                9: ["then", 'before', 'after'],  # 时间关系
                10: ["meanwhile", "when"]  # 时间关系
            }
        else:
            # conll 14分类任务
            classes = list(range(14))  # 14个类别
            label_words = {
                0: ['however', 'although'],  # 转折关系
                1: ["but"],  # 转折关系
                2: ["because"],  # 因果关系
                3: ['so', 'thus', 'consequently', 'therefore'],  # 因果关系
                4: ["if"],  # 条件关系
                5: ["unless"],  # 条件关系
                6: ["instead", 'rather'],  # 替代关系
                7: ["and", "also", "furthermore", 'fact'],  # 并列关系
                8: ["except"],  # 排除关系
                9: ["example", 'instance'],  # 示例关系
                10: ["specifically", 'particular', 'indeed'],  # 具体化关系
                11: ["then", 'before'],  # 时间关系
                12: ['after'],  # 时间关系
                13: ["meanwhile", 'when']  # 时间关系
            }

        # 创建标签词汇器，将类别和标签词汇与tokenizer结合
        self.promptVerbalizer = ManualVerbalizer(
            classes=classes,  # 类别
            label_words=label_words,  # 标签词汇
            tokenizer=tokenizer,  # tokenizer
            post_log_softmax=False  # 是否在softmax之后执行操作，False表示不做
        )
    
    def get_plm(self):
        """ 返回预训练语言模型 """
        return self.plm
    
    def get_tokenizer(self):
        """ 返回tokenizer """
        return self.tokenizer
    
    def get_wrapperclass(self):
        """ 返回模型的包装类 """
        return self.wrapperclass
    
    def get_template(self):
        """ 返回提示模板 """
        return self.promptTemplate

    def get_verbalizer(self):
        """ 返回标签词汇器 """
        return self.promptVerbalizer
