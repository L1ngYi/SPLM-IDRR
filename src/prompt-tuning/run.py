import os
import shutil
import time
import torch
import argparse
import logging
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from torch.optim import AdamW
import json


from data_process import PromptDataGenerate
from tqdm import tqdm, trange
from prompt_model import PromptIDRC
from evaluate import evaluate_accuracy, evaluate_precision_recall_f1
from prompt_config import PromptConfig


# str2list 函数将逗号分隔的字符串转换为列表
def str2list(x):
    results = []
    for x in x.split(","):
        x = x.strip()  # 去除两边的空格
        try:
            x = eval(x)  # 尝试将字符串转换为对应的数据类型
        except:
            pass
        results.append(x)
    return results

# 打印模型评估结果
def print_model_result(result, data_type='train'):
    for key in sorted(result.keys()):
        print(" \t %s = %-5.5f" % (key, float(result[key])), end="")

# 模型评估函数
def model_eval(model, args, data_loader, data_type='dev', epoch_num=-1, metric=None, device=None):
    result_sum = {}  # 存储评估结果
    nm_batch = 0  # 批次计数
    labels_pred = list()  # 存储预测标签
    multi_labels_true = list()  # 存储真实标签
    total_cnt = 0  # 总计数
    total_loss = 0.0  # 总损失
    # 遍历数据加载器中的批次
    for step, batch in enumerate(tqdm(data_loader)):

        labels = batch.label  # 获取真实标签
        bsz = len(labels)  # 获取批次大小
        total_cnt += bsz  # 累加样本数量

        input_ids = batch.input_ids
        attention_mask = batch.attention_mask
        token_type_ids = batch.token_type_ids
        loss_ids = batch.loss_ids
        
        # 将数据移到指定的设备（如 GPU）
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        loss_ids = loss_ids.to(device)
        label = labels.to(device)
        
        model.eval()  # 设置模型为评估模式
        with torch.no_grad():  # 禁止梯度计算
            loss, pred = model(input_ids, attention_mask, token_type_ids, loss_ids, label)
        
        # 如果使用多GPU进行训练，计算均值
        if torch.cuda.device_count() > 1:
            loss = loss.mean()
            
        total_loss += loss.item() * bsz  # 累加损失
        pred = np.argmax(pred.detach().cpu().numpy(), axis=1)  # 获取预测结果
        labels_pred.extend(pred.tolist())  # 将预测结果添加到列表中
        multi_labels_true.extend(labels.tolist())  # 将真实标签添加到列表中

        nm_batch += 1  # 批次计数器加1

    # 计算评估指标
    result_sum["loss"] = total_loss / total_cnt
    result_sum["accuracy"] = evaluate_accuracy(np.array(labels_pred), np.array(multi_labels_true))
    f1_results = evaluate_precision_recall_f1(np.array(labels_pred), np.array(multi_labels_true))
    result_sum["f1-detail"] = f1_results
    result_sum["f1"] = f1_results["overall"][-1]
    print(result_sum["f1"])
    
    # 将结果保存到文件中
    with open(os.path.join(args.output_dir, 'Discourage_' + data_type + '_result.txt'), 'a+',
              encoding='utf-8') as writer:
        print("***** Eval results in " + data_type + "*****")
        if data_type == 'dev':
            writer.write(f"======{data_type} Epoch {epoch_num}=========\n")
        else:
            writer.write(f"======{data_type} Best {metric}=========\n")
        for key in sorted(result_sum.keys()):
            print("%s = %s" % (key, str(result_sum[key])))
            writer.write("%s = %s\n" % (key, str(result_sum[key])))
        writer.write('\n')

    return result_sum

# 保存最佳模型（基于损失值或F1分数）
def save_best_model(model, args, v, optimizer=None, data_type='dev', eval_best = None, train_best = None, use_f1=False, logger = None):
     # 使用损失值作为评估指标
    if not use_f1 and data_type == 'dev':
        if eval_best > v:
            eval_best = v
            state = {
                'prompt_model': model.prompt_model.state_dict(),
                'optimizer': optimizer
            }
            save_path = os.path.join(args.output_dir, 'Discourage' + '_state_dict_' +
                                 data_type + '_loss_' + str(v) + '.model')
            torch.save(state, save_path)
            logger.info(f"========Save best loss model {save_path}=========\n")
            train_best = save_path
            # 复制最佳F1分数和最佳损失的模型文件到输出目录
            logger.info(f"========best loss model changed to {eval_best}=======\n")
            shutil.copy(save_path, os.path.join(args.output_dir, 'best_loss_model.bin'))
        return eval_best, train_best

    # 使用F1分数作为评估指标
    if use_f1 and data_type == 'dev':
        if eval_best < v:
            eval_best = v
            state = {
                'prompt_model': model.prompt_model.state_dict(),
                'optimizer': optimizer
            }
            save_path = os.path.join(args.output_dir, 'Discourage' + '_state_dict_'
                                     + data_type + '_f1_' + str(v) + '.model')
            torch.save(state, save_path)
            logger.info(f"========Save best f1 model {save_path}=========\n")
            train_best = save_path
            logger.info(f"========best f1 model changed to {eval_best}=======\n")
            shutil.copy(save_path, os.path.join(args.output_dir, 'best_f1_model.bin'))
        return eval_best, train_best

# 保存每个epoch的模型
def save_epoch_model(model, epoch, args, logger):
    model.eval()
    state = {'prompt_model': model.prompt_model.state_dict()}
    save_path = os.path.join(args.output_dir, 'Discourage' + '_state_dict_' + '_epoch_' + str(epoch) + '.model')
    logger.info(f"Save Epoch {epoch} Model\n")
    torch.save(state, save_path)

# 保存训练配置到文件
def save_config(args, prompt_config, logger):
    logger.info("save config")
    run_conf = {
        'lr' : args.learning_rate,
        'train_batch_size':args.train_batch_size,
        'template': prompt_config.get_template().text,
        'verbalizer': prompt_config.get_verbalizer().label_words,
        'use_pretrain': args.use_pretrain,
        'pretrain_file': args.plse_pretrain_file
    }
    json.dump(run_conf,open(os.path.join(args.output_dir,"train_config.json"),'w'),ensure_ascii=False,indent=4)

# 处理模型（若使用DataParallel，返回模块）
def inner_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model

# 训练函数
def train(model, args, data_generator, prompt_config, logger,  device):
    
    dev_loader = data_generator.get_dev_loader()  # 获取验证集数据加载器
    train_loader = data_generator.get_train_loader()  # 获取训练集数据加载器
    
    # 如果使用多GPU，使用DataParallel包装模型
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        
    model = model.to(device)  # 将模型移动到设备（GPU或CPU）

    global_step = 0  # 初始化全局步骤计数器
    
    #optimizer = AdamW(model.parameters(), lr=args.learning_rate)  # 使用AdamW优化器
    # 获取 Soft Token 的参数
    soft_token_parameters = []
    soft_token_parameters.append(model.prompt_model.template.new_embedding.parameters())  # Soft Token embedding 参数

    # 如果使用了 LSTM 编码器
    if hasattr(model.prompt_model.template, "new_lstm_head"):
        soft_token_parameters.append(model.prompt_model.template.new_lstm_head.parameters())  # LSTM 权重
    # 如果使用了 MLP
    if hasattr(model.prompt_model.template, "new_mlp_head"):
        soft_token_parameters.append(model.prompt_model.template.new_mlp_head.parameters())  # MLP 权重

    # 展开所有 Soft Token 参数
    soft_token_parameters = list(param for group in soft_token_parameters for param in group)
    soft_token_parameters = list(set(soft_token_parameters)) # 去重
    # 获取预训练模型的参数
    pretrained_model_parameters = model.prompt_model.plm.parameters()

    optimizer = AdamW([
    {"params": pretrained_model_parameters, "lr": args.learning_rate},  # 预训练模型学习率
    {"params": soft_token_parameters, "lr": args.p_tuning_learning_rate},        # Soft Token学习率
    ])




    save_config(args, prompt_config, logger)  # 保存训练配置

    eval_best_loss = 999  # 最佳损失初始化为较大值
    eval_best_f1 = 0  # 最佳F1分数初始化为0
    train_best_loss_model = None  # 初始化最佳损失模型路径
    train_best_f1_model = None  # 初始化最佳F1模型路径
    # 进行多轮训练
    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        model.train()  # 设置模型为训练模式
        for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
            
            labels = batch.label  # 获取标签
            input_ids = batch.input_ids
            attention_mask = batch.attention_mask
            token_type_ids = batch.token_type_ids
            loss_ids = batch.loss_ids
            # 将数据移到设备
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            loss_ids = loss_ids.to(device)
            label = labels.to(device)

            # 计算损失和输出
            loss, output = model(input_ids, attention_mask, token_type_ids, loss_ids, label)
            if n_gpu > 1:
                loss = loss.mean()  # 使用多GPU时，求均值

            loss.backward()  # 反向传播
            optimizer.step()  # 更新优化器
            optimizer.zero_grad()  # 清空梯度
            global_step += 1  # 更新全局步骤计数器
        if args.do_eval:
            model.eval()  # 将模型设置为评估模式
            print("\nepoch:{} global:{}\t".format(epoch, global_step))  # 打印当前epoch和全局步数
            # 评估模型在开发集（dev）的表现
            eval_result = model_eval(inner_model(model), args, dev_loader, data_type='dev', epoch_num=epoch, metric=None, device=device)

            # 按照损失最小保存最佳模型
            eval_best_loss, train_best_loss_model = save_best_model(inner_model(model), args, eval_result['loss'], optimizer.state_dict(), eval_best=eval_best_loss, data_type='dev', logger=logger)
            # 按照F1分数最大保存最佳模型
            eval_best_f1, train_best_f1_model = save_best_model(inner_model(model), args, eval_result['f1'], optimizer.state_dict(), eval_best=eval_best_f1, data_type='dev',
                                use_f1=True, logger=logger)

        # 保存每个epoch的模型
        save_epoch_model(inner_model(model), epoch, args, logger)

    



def eval_test(model, args, data_generator, logger, device):
    model.eval()  # 将模型设置为评估模式
    # 设置最佳模型的文件路径列表
    best_model_path = [os.path.join(args.output_dir, 'best_f1_model.bin'),
                       os.path.join(args.output_dir, 'best_loss_model.bin')]
    # 逐个加载最佳模型文件并进行测试评估
    for best_model in best_model_path:
        checkpoint = torch.load(best_model)['prompt_model']  # 加载模型权重
        model.prompt_model.load_state_dict(checkpoint, strict=False)  # 加载到模型的prompt部分
        model = model.to(device)  # 将模型加载到设备
        test_loader = data_generator.get_test_loader()  # 获取测试集数据加载器
        
        logger.info("\n********" + best_model + "********")  # 记录日志
        # 在测试集上评估模型表现，若模型名称包含“loss”则使用损失作为度量，否则使用F1分数
        model_eval(model, args, test_loader, data_type='test', metric='loss' if 'loss' in best_model else 'f1', device=device)
        # 如果关系类别为14，进一步在盲集（blind）上评估
        if args.num_rels == 14:
            blind_loader = data_generator.get_blind_loader()  # 获取盲集加载器
            model_eval(model, args, blind_loader, data_type='blind', metric='loss' if 'loss' in best_model else 'f1', device=device)
    pass  # pass语句用于防止空语句块报错


if __name__ == '__main__':
#     
#     train_pdtb_4.sh
#
#     CUDA_VISIBLE_DEVICES=0 python ./src/prompt-tuning/run.py \
#   --seed 42 \
#   --gpu_ids 0 \
#   --num_train_epochs 10 \
#   --train_batch_size 64 \
#   --dev_batch_size 64 \
#   --test_batch_size 64 \
#   --learning_rate 1e-5 \
#   --pretrain_file ./pretrain_model/base/roberta-base \
#   --data_dir ./src/data/pdtb2_4 \
#   --output_dir ./checkpoint/plse_model_pdtb_4 \
#   --max_seq_length 256 \
#   --num_rels 4 \
#   --plse_pretrain_file ./pretrain_model/plse/plse_pretrain_model/pretrain_state_epoch_1.ckpt \
#   --do_train \
#   --do_eval \
#   --do_test \
#   --use_pretrain
#    
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器
    parser.add_argument("--seed", default=42, type=int)  # 设置随机种子
    parser.add_argument("--gpu_ids", type=str2list, default=None, help="gpu ids")  # 设置GPU ID列表
    parser.add_argument("--num_train_epochs", default=10, type=int, help="number of train epochs")  # 训练总epoch数
    parser.add_argument("--train_batch_size", default=64, type=int)  # 训练批次大小
    parser.add_argument("--test_batch_size", default=64, type=int)  # 测试批次大小
    parser.add_argument("--dev_batch_size", default=64, type=int)  # 开发集批次大小
    parser.add_argument("--learning_rate", default=1e-5, type=float)  # 学习率
    parser.add_argument("--pretrain_file", default='', type=str)  # 预训练文件路径
    parser.add_argument("--data_dir", default='', type=str)  # 数据目录
    parser.add_argument("--output_dir", default='', type=str)  # 输出目录
    parser.add_argument("--plse_pretrain_file", default='', type=str)  # PLSE预训练文件路径
    parser.add_argument("--max_seq_length", default=256, type=int)  # 最大序列长度
    parser.add_argument("--num_rels", type=int, default=4, choices=[4, 11, 14], help="how many relations are computed")  # 设置关系数目
    parser.add_argument("--train_best_f1_model", default='', type=str)  # 最佳F1模型文件路径    
    parser.add_argument("--train_best_loss_model", default='', type=str)  # 最佳损失模型文件路径 
    parser.add_argument("--do_train", action="store_true")  # 是否进行训练
    parser.add_argument("--do_eval", action="store_true")  # 是否进行评估
    parser.add_argument("--do_test", action="store_true")  # 是否进行测试
    parser.add_argument("--use_pretrain", action="store_true")  # 是否使用预训练
    parser.add_argument("--p_tuning_learning_rate",default=1e-3,type=float)
    args = parser.parse_args()  # 解析命令行参数

    start_time = time.time()  # 记录起始时间
    os.makedirs(args.output_dir, exist_ok=True)  # 创建输出目录

    logger = logging.getLogger()  # 设置日志记录器
    logger.setLevel(logging.INFO)  # 设置日志级别为INFO
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%Y/%m/%d %H:%M:%S')  # 设置日志格式
    console = logging.StreamHandler()  # 创建控制台日志处理器
    console.setFormatter(fmt)  # 设置控制台日志格式
    logger.addHandler(console)  # 添加控制台处理器到日志记录器
    logfile = logging.FileHandler(os.path.join(args.output_dir, "Prompt.log"), 'a')  # 创建文件日志处理器
    logfile.setFormatter(fmt)  # 设置文件日志格式
    logger.addHandler(logfile)  # 添加文件处理器到日志记录器

    prompt_config = PromptConfig(args)  # 初始化Prompt配置     ##引用PromptConif.py
    model = PromptIDRC(prompt_config)  # 初始化PromptIDRC模型   ##引用Promptmodel.py
    
    torch.manual_seed(args.seed)  # 设置随机种子
    np.random.seed(args.seed)  # 设置numpy随机种子
    data_generator = PromptDataGenerate(args, prompt_config)  # 初始化数据生成器   ##引用dataProcess
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择设备，优先使用GPU
    n_gpu = len(args.gpu_ids)  # 获取GPU数量
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))  # 记录设备信息
    # 如果有多个GPU可用，则使用指定的GPU
    if n_gpu > 0:
        device = torch.device("cuda:%d" % args.gpu_ids[0])  # 设置使用的GPU ID
        torch.cuda.set_device(device)  # 设置GPU设备
        
    if args.do_train:
        train(model, args, data_generator, prompt_config, logger, device)  # 训练模型
    if args.do_test:
        eval_test(model, args, data_generator, logger, device)  # 测试模型
    end_time = time.time()  # 记录结束时间
    print("Time Cost：%d m" % int((end_time - start_time) / 60))  # 输出总耗时
    pass  # pass用于防止空语句块
