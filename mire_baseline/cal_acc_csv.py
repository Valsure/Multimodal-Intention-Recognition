
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import json


def calculate_metrics(y_true, y_pred):
    """
    计算多标签分类任务的指标: 加权F1, 总体acc, precision, recall。

    参数:
    y_true (List[List[int]]): 真实标签
    y_pred (List[List[int]]): 预测标签

    返回:
    dict: 各指标的分数
    """

    # 计算指标
    metrics = {
        "f1": float(f1_score(y_true, y_pred, average="weighted")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted")),
        "recall": float(recall_score(y_true, y_pred, average="weighted")),
    }

    return metrics


def get_label_pred(test_file, pred_file):
    """获取测试集标签以及预测结果

    Args:
        test_file (_type_): 带ground_truth标签的测试集文件
        pred_file (_type_): 对应的预测结果文件
    """
    test_data = pd.read_csv(test_file)
    id_test = test_data['id']
    labels = test_data['output']

    pred_data = pd.read_csv(pred_file)
    id_pred = pred_data['id']
    preds = pred_data['predict']
    
    assert id_test.equals(id_pred), "id_test and id_pred not match"

    return labels, preds


def cal_acc(test_file, pred_file):
    labels, preds = get_label_pred(test_file, pred_file)
    metrics = calculate_metrics(y_true=labels, y_pred=preds)
    return metrics


def badcase_analysis(test_file, pred_file):
    labels, preds = get_label_pred(test_file, pred_file)
    case = [(item_true, item_pred, 1 if item_true == item_pred else 0) for item_true, item_pred in zip(labels, preds)]
    case = pd.DataFrame(case, columns=["label", "pred", "correct"])
    case.to_csv("../data/badcase.csv")
    
if __name__ == "__main__":
    test_file = "../data/output_demo.csv"
    pred_file = "../data/prediction_demo.csv"
    metrics = cal_acc(test_file, pred_file) #修改了4个
    badcase_analysis(test_file, pred_file)

    print(metrics)
