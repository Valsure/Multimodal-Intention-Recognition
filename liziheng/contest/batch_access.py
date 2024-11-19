import requests
import csv
import json

# API 地址
url = "http://127.0.0.1:8005/batch_predict"

# 定义逐条写入预测结果的函数
def process_and_save_individual_predictions(json_file_path, output_csv_path):
    with open(json_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    # 打开 CSV 文件（追加模式）
    with open(output_csv_path, mode="a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        
        # 如果文件为空，写入标题
        if csv_file.tell() == 0:
            writer.writerow(["id", "predict"])

        # 逐条读取数据
        for item in data["items"]:
            request_data = {
                "items": [item],  # 只发送一条数据
                "image_folder": data["image_folder"]
            }
            
            # 发送请求
            response = requests.post(url, json=request_data)
            
            # 检查响应状态
            if response.status_code == 200:
                result = response.json()
                
                # 清理 predict 字段，去除开头和结尾多余的引号或三重引号
                cleaned_predict = result[0]["predict"]
                if cleaned_predict.startswith('"""') and cleaned_predict.endswith('"""'):
                    cleaned_predict = cleaned_predict[3:-3]
                elif cleaned_predict.startswith('"') and cleaned_predict.endswith('"'):
                    cleaned_predict = cleaned_predict[1:-1]
                
                # 将清理后的结果写入 CSV
                writer.writerow([result[0]["id"], cleaned_predict.strip()])
                print(f"写入 {result[0]['id']} 的结果到 CSV")
            else:
                print(f"请求失败: {response.status_code}, {response.text}")
                continue  # 跳过失败的请求，继续下一条数据

# 调用函数处理大数据集
process_and_save_individual_predictions("/hy-tmp/project/WWW2025/datasets/test1/test1.json", "prediction_results.csv")
