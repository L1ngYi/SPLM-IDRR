import numpy as np  # 导入numpy库，用于数值计算和数组操作
from sklearn.metrics import precision_recall_fscore_support, accuracy_score  # 导入sklearn的评估函数，计算精准率、召回率、F1分数和准确率


# 我们遵循CoNLL 2015共享任务的评估标准
# 在一个样本中，如果预测的关系与目标相同，则认为是正确的
# 否则，多个关系会合并成一个优选的关系
# 参考链接：https://github.com/attapol/conll15st/blob/master/scorer.py

#OK

def evaluate_accuracy(pred, target, prefered_target=None):
    """
    计算预测结果的准确率。
    :param pred: 预测结果，形状为(num_examples, num_classes)，每行是一个预测类别索引
    :param target: 真实标签，形状为(num_examples, num_classes)，每行是一个目标标签的one-hot编码
    :param prefered_target: 优选目标标签（可选），如果为None，则使用真实标签的最大值作为目标标签
    :return: 包含各类别和整体准确率的字典
    """
    num_examples, num_classes = target.shape  # 获取样本数量和类别数量

    correct = 0  # 初始化正确预测的数量
    pred_refined = np.zeros_like(target)  # 初始化预测标签的精细化数组
    target_refined = np.zeros_like(target)  # 初始化目标标签的精细化数组

    # 遍历每一个样本
    for i in range(num_examples):
        j = pred[i]  # 获取当前样本的预测标签
        pred_refined[i, j] = 1  # 在预测的标签位置标记为1
        if target[i, j]:  # 如果预测标签与真实标签相符
            correct += 1  # 增加正确预测计数
            target_refined[i, j] = 1  # 在目标标签的位置标记为1
        else:
            if prefered_target is not None:  # 如果提供了优选目标
                j = prefered_target[i]  # 使用优选目标标签
            else:
                j = target[i].argmax()  # 否则使用目标标签的最大值索引作为目标
            target_refined[i, j] = 1  # 在目标标签的位置标记为1

    # 删除没有标签的类别，并将这些预测标签映射到一个虚拟标签
    cnt = target_refined.sum(axis=0)  # 统计每个类别的标签总和
    real_labels = cnt > 0  # 标记真实存在标签的类别
    result = {}  # 初始化结果字典

    # 遍历每个类别
    for c in range(num_classes):
        if real_labels[c]:  # 如果该类别有真实标签
            result[c] = accuracy_score(target_refined[:, c], pred_refined[:, c])  # 计算该类别的准确率
        else:
            result[c] = 1.0  # 如果该类别没有真实标签，准确率为1

    result["overall"] = correct / num_examples  # 计算整体的准确率
    return result  # 返回结果字典，包含各类别和整体准确率


def evaluate_precision_recall_f1(pred, target, prefered_target=None, average="macro"):
    """
    计算预测结果的精确度、召回率和F1分数。
    :param pred: 预测结果，形状为(num_examples, num_classes)，每行是一个预测类别索引
    :param target: 真实标签，形状为(num_examples, num_classes)，每行是一个目标标签的one-hot编码
    :param prefered_target: 优选目标标签（可选），如果为None，则使用真实标签的最大值作为目标标签
    :param average: 计算平均方式，可以是'macro'、'micro'或'binary'
    :return: 包含各类别及整体评估结果的字典
    """
    num_examples, num_classes = target.shape  # 获取样本数量和类别数量

    # 如果使用二分类的平均方式
    if average == "binary":
        if len(pred.shape) == 2:  # 如果预测结果是二维数组（每个样本有多个类的预测概率）
            pred_refined = pred.argmax(axis=1)  # 获取每个样本的最大预测值的索引作为最终类别
        else:
            pred_refined = pred  # 否则直接使用预测结果
        target_refined = target[:, 1]  # 获取目标标签的第二列（假设为二分类标签）

        # 计算并返回二分类的精准率、召回率、F1分数
        result = {
            "overall": tuple(precision_recall_fscore_support(target_refined, pred_refined, average="binary")[0:3])}
    else:
        # 初始化精细化的预测标签和目标标签
        pred_refined = np.zeros_like(target)
        target_refined = np.zeros_like(target)
        for i in range(num_examples):
            j = pred[i]  # 获取当前样本的预测标签
            pred_refined[i, j] = 1  # 在预测标签的位置标记为1
            if target[i, j]:  # 如果预测标签和真实标签匹配
                target_refined[i, j] = 1  # 将目标标签对应位置标记为1
            else:
                if prefered_target is not None:  # 如果提供了优选目标
                    j = prefered_target[i]  # 使用优选目标标签
                else:
                    j = target[i].argmax()  # 否则使用目标标签的最大值作为目标
                target_refined[i, j] = 1  # 将目标标签对应位置标记为1

    # 删除没有标签的类别，并将这些预测标签映射到一个虚拟标签
    cnt = target_refined.sum(axis=0)  # 统计每个类别的标签总和
    real_labels = cnt > 0  # 标记真实标签存在的类别
    result = {}  # 初始化结果字典

    # 遍历每个类别，计算精准率、召回率和F1分数
    for c in range(num_classes):
        if real_labels[c]:  # 如果该类别有真实标签
            result[c] = tuple(
                precision_recall_fscore_support(target_refined[:, c], pred_refined[:, c], average="binary")[0:3])
        else:
            result[c] = (0.0, 0.0, 0.0)  # 如果该类别没有真实标签，返回0的精准率、召回率和F1分数

    # 如果没有没有标签的类别，计算整体的评估结果
    if pred_refined[:, cnt <= 0].sum() == 0:
        result["overall"] = tuple(
            precision_recall_fscore_support(target_refined[:, real_labels], pred_refined[:, real_labels],
                                            average=average)[0:3])  # 计算整体的精准率、召回率和F1分数
    else:
        result["overall"] = tuple(precision_recall_fscore_support(
            np.concatenate([target_refined[:, real_labels], np.zeros((num_examples, 1), dtype=target_refined.dtype)],
                           axis=1),  # 拼接没有标签的类别，创建虚拟标签
            np.concatenate(
                [pred_refined[:, real_labels], pred_refined[:, cnt <= 0].sum(axis=1).reshape(num_examples, 1)], axis=1),
            average=average)[0:3])  # 计算并返回整体的评估结果（精准率、召回率、F1分数）

    return result  # 返回评估结果
