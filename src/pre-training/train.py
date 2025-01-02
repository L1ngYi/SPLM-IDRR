import os  # 导入os模块用于文件操作
import time  # 导入time模块用于时间处理
import logging  # 导入logging模块用于日志记录
import math  # 导入math模块用于数学计算
import argparse  # 导入argparse模块用于命令行参数解析
import numpy as np  # 导入numpy用于数值计算
import pandas as pd  # 导入pandas用于数据处理
import torch  # 导入PyTorch
from tqdm.auto import tqdm  # 导入tqdm用于进度条显示
from transformers import AdamW, get_cosine_schedule_with_warmup  # 导入优化器和学习率调度器
from transformers import RobertaForMaskedLM  # 导入RoBERTa模型
from PromptModel import RobertaForPrompt  # 导入自定义的Prompt模型
from transformers import RobertaTokenizer  # 导入RoBERTa分词器
from data_processor import GigaProcessor  # 导入数据处理模块
import json  # 导入json模块用于文件读写



###
# CUDA_VISIBLE_DEVICES=0 python ./src/pre-training/train.py \
#   --num_epochs 2 \
#   --batch_size 64 \    ---CHANGE TO 16
#   --learning_rate 5e-6 \
#   --num_warmup_ratio 0.1 \
#   --sen_max_length 256 \
#   --initial_pretrain_model ./pretrain_model/base/roberta-base \
#   --path_model_save ./pretrain_model/plse/plse_pretrain_model \
#   --path_datasets ./src/data/explicit_data \
#   --lse \
#   --mlm \
#   --connective_mask
###
def train(args, logger):
    logger.info('training start')  # 记录训练开始的日志
    save_config(args, logger)  # 保存配置文件
    device = torch.device('cuda')  # 设置使用GPU设备

    # 数据加载
    logger.info('data loading')
    tokenizer = RobertaTokenizer.from_pretrained(args.initial_pretrain_model)  # 初始化分词器
    data_processor = GigaProcessor(args, tokenizer)  # 初始化数据处理器

    # 初始化模型
    logger.info('model loading')
    if args.lse:  # 根据参数选择模型
        model = RobertaForPrompt.from_pretrained(args.initial_pretrain_model)
        logger.info('RobettaForPrompt is selected') #l1ngyi
    else:
        model = RobertaForMaskedLM.from_pretrained(args.initial_pretrain_model)
        logger.info('RobettaForMaskedLM is selected') #l1ngyi

    state_dict = None  # 初始化状态字典
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)  # 初始化优化器

    num_training_steps = args.num_epochs * data_processor.get_len()  # 计算总训练步数
    lr_scheduler = get_cosine_schedule_with_warmup(  # 初始化学习率调度器
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_ratio * num_training_steps,
        num_training_steps=num_training_steps
    )
    
    # 分布式训练
    model.to(device)  # 将模型转移到GPU
    if torch.cuda.device_count() > 1:  # 检查是否有多张GPU
        model = torch.nn.DataParallel(model)  # 使用数据并行
    logger.info('start to train')  # 记录训练开始
    model.train()  # 设置模型为训练模式
    
    loss_best = math.inf  # 初始化最佳损失为无穷大

    start_epoch = 0  # 起始轮次
    end_epoch = args.num_epochs  # 结束轮次
    cur_training_steps = end_epoch * data_processor.get_len()  # 当前训练步数
        
    progress_bar = tqdm(range(cur_training_steps))  # 初始化进度条
    eval_dl = data_processor.get_data_loader('test')  # 获取测试数据加载器

    for epoch in range(start_epoch, end_epoch):  # 遍历每个训练轮次
        train_dl = data_processor.get_data_loader('train')  # 获取训练数据加载器
        logger.info("Load dataloader Finished")  # 记录数据加载完成
        for i, batch in enumerate(train_dl):  # 遍历训练数据
            input_ids, attention_mask, token_type_ids, labels, conns_index = batch  # 解包批次数据
            input_ids = input_ids.to(device)  # 转移到GPU
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            labels = labels.to(device)
            conns_index = conns_index.to(device)

            if args.lse:  # 根据参数选择模型输出
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, conns_index=conns_index)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            mask_loss = outputs.loss  # 获取掩码损失
            if args.lse:  # 如果使用LSE，获取互信息损失
                mutual_loss = outputs.mutual_loss
            if torch.cuda.device_count() > 1:  # 如果使用多GPU，计算均值
                mask_loss = mask_loss.mean()
                if args.lse:
                    mutual_loss = mutual_loss.mean()
                
            if args.lse:  # 计算总损失
                loss = mask_loss + mutual_loss
            else:
                loss = mask_loss
            
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            lr_scheduler.step()  # 更新学习率
            optimizer.zero_grad()  # 清空梯度
            progress_bar.update(1)  # 更新进度条

            if i % 250 == 0:  # 每250个步骤记录一次日志
                if args.lse:
                    logger.info('epoch:{0}  iter:{1}/{2}  loss:{3}  mutual_loss:{4}  mask_loss:{5}'.format(epoch, i, len(train_dl), loss.item(), mutual_loss.item(),  mask_loss.item()))
                else:
                    logger.info('epoch:{0}  iter:{1}/{2}  loss:{3}'.format(epoch, i, len(train_dl), loss.item()))


        ###
        # now l1ngyi start ...
        logger.info('ready to eval ..   saving first') #l1ngyi
        path = args.path_model_save + 'pretrain_state_epoch_{}without eval.ckpt'.format(epoch)  # 保存路径
        model_save = model.module if torch.cuda.device_count() > 1 else model  # 处理多GPU模型
        state_dict = {  # 保存状态字典
            'epoch': epoch,
            'net': model_save.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr': lr_scheduler.state_dict()
        }
        torch.save(state_dict, path)  # 保存模型
        logger.info(f"Save epoch-{epoch} model state without eval")  # 记录模型保存状态
        # l1ngyi end
        ###

        current_loss = eval(eval_dl, args, model, epoch, logger, device)  # 评估当前损失
        if current_loss < loss_best:  # 如果当前损失优于最佳损失，保存模型
            loss_best = current_loss
            logger.info('saving model')
            path = args.path_model_save + 'pretrain_state_epoch_{}.ckpt'.format(epoch)  # 保存路径
  
            model_save = model.module if torch.cuda.device_count() > 1 else model  # 处理多GPU模型
            state_dict = {  # 保存状态字典
                'epoch': epoch,
                'net': model_save.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr': lr_scheduler.state_dict()
            }
            torch.save(state_dict, path)  # 保存模型
            logger.info(f"Save epoch-{epoch} model state")  # 记录模型保存状态
            
def eval(eval_dataloader, args, model, epoch, logger, device):
    logger.info(f"start epoch_{epoch} test")  # 记录测试开始
    mutual_losses = []  # 存储互信息损失
    mask_losses = []  # 存储掩码损失
    losses = []  # 存储总损失
    model.eval()  # 设置模型为评估模式
    for step, batch in enumerate(tqdm(eval_dataloader)):  # 遍历评估数据
        input_ids, attention_mask, token_type_ids, labels, conns_index = batch  # 解包批次数据
        input_ids = input_ids.to(device)  # 转移到GPU
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        labels = labels.to(device)
        conns_index = conns_index.to(device)
        with torch.no_grad():  # 禁用梯度计算
            if args.lse:  # 根据参数选择模型输出
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, conns_index=conns_index)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        mask_loss = outputs.loss  # 获取掩码损失
        if args.lse:  # 如果使用LSE，获取互信息损失
            mutual_loss = outputs.mutual_loss
        
        if torch.cuda.device_count() > 1:  # 如果使用多GPU，计算均值
            mask_loss = mask_loss.mean()
            if args.lse:
                mutual_loss = mutual_loss.mean()
    
        if args.lse:  # 计算总损失
            loss = mask_loss + mutual_loss
        else:
            loss = mask_loss
        
        if args.lse:  # 存储损失
            mutual_losses.append(mutual_loss)
            mask_losses.append(mask_loss)
        losses.append(loss)
        
    if args.lse:  # 转换为张量
        mutual_losses = torch.tensor(mutual_losses, device=device)   #####出现报错 l1ngyi  这里在转换交互损失  mutual_losses = torch.tensor(mutual_losses,device)
        mask_losses = torch.tensor(mask_losses, device=device)  ###l1ngyi   mask_losses = torch.tensor(mask_losses, device)
    losses = torch.tensor(losses, device=device)  ###l1ngyi  losses = torch.tensor(losses, device)
    
    if args.lse:  # 计算平均损失
        mutual_losses_avg = torch.mean(mutual_losses)
        mask_losses_avg = torch.mean(mask_losses)
    losses_avg = torch.mean(losses)
    if args.lse:
        logger.info('eval {0}: loss:{1}  mutual_loss:{2}  mask_loss:{3}'.format(epoch, losses_avg.item(), mutual_losses_avg.item(), mask_losses_avg.item()))
    else:
        logger.info('eval {0}: loss:{1}'.format(epoch, losses_avg.item()))
    return losses_avg  # 返回平均损失

def save_config(args, logger):
    logger.info("save config")  # 记录配置保存
    run_conf = {  # 配置字典
        'lr' : args.learning_rate,
        'batch_size': args.batch_size,
        'max_length': args.sen_max_length,
        'mlm': args.mlm,
        'lse': args.lse,
        'connective_mask': args.connective_mask,
    }
    json.dump(run_conf, open(os.path.join(args.path_model_save, "train_config.json"), 'w'), ensure_ascii=False, indent=4)  # 保存配置到json文件

if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # 创建参数解析器
    parser.add_argument("--seed", default=42, type=int)  # 随机种子
    parser.add_argument("--num_epochs", default=2, type=int)  # 训练轮数
    parser.add_argument("--batch_size", default=64, type=int)  # 批次大小
    parser.add_argument("--learning_rate", default=5e-6, type=float)  # 学习率
    parser.add_argument("--num_warmup_ratio", default=0.1, type=float)  # 预热比例
    parser.add_argument("--sen_max_length", default=256, type=int)  # 最大句子长度
    parser.add_argument("--initial_pretrain_model", default='', type=str)  # 初始预训练模型路径
    parser.add_argument("--path_model_save", default='', type=str)  # 模型保存路径
    parser.add_argument("--path_datasets", default='', type=str)  # 数据集路径
    parser.add_argument("--path_log", default='', type=str)  # 日志路径
    parser.add_argument("--lse", action="store_true", help="global logical semantics enhancement")  # 是否启用逻辑语义增强
    parser.add_argument("--mlm", action="store_true")  # 是否启用掩码语言模型
    parser.add_argument("--connective_mask", action="store_true")  # 是否启用连接词掩码
    args = parser.parse_args()  # 解析参数
    
    torch.manual_seed(args.seed)  # 设置PyTorch随机种子
    np.random.seed(args.seed)  # 设置numpy随机种子
    
    path_log = os.path.join(args.path_model_save, 'logs/')  # 日志文件夹路径
    if not os.path.exists(args.path_model_save): 
        os.mkdir(args.path_model_save)  # 创建模型保存路径
    if not os.path.exists(path_log):
        os.mkdir(path_log)  # 创建日志路径
        
    logger = logging.getLogger()  # 获取日志记录器
    logger.setLevel(logging.INFO)  # 设置日志级别
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%Y/%m/%d %H:%M:%S')  # 设置日志格式
    console = logging.StreamHandler()  # 控制台输出
    console.setFormatter(fmt)  # 设置控制台格式
    logger.addHandler(console)  # 添加控制台处理器
    logfile = logging.FileHandler(os.path.join(path_log, f"pretrain_{time.time()}.log"), 'a')  # 创建日志文件处理器
    logfile.setFormatter(fmt)  # 设置文件格式
    logger.addHandler(logfile)  # 添加文件处理器
    train(args, logger)  # 开始训练
