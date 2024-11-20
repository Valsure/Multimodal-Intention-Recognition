import requests
import csv
import json
import re

request_url = "http://127.0.0.1:8005/single_predict"
# input_path = "/hy-tmp/project/WWW2025/datasets/test_mini/test_mini.json"
input_path = "/hy-tmp/project/WWW2025/datasets/test1/test1.json"
output_path = "/hy-tmp/project/WWW2025/datasets/prediction.csv"

def strip_quotes(s):
    # 使用正则表达式去掉字符串首尾的引号
    return re.sub(r'^[\'"]+|[\'"]+$', '', s)

def process_single_task(input_path ,output_path):
    with open(input_path, "r", encoding = "utf-8" ) as input_file:
        json_file = json.load(input_file)
        image_folder = json_file["image_folder"]
    
    with open(output_path, "a", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)

        if csv_file.tell() == 0:
            writer.writerow(["id", "predict"])
        
        for item in json_file["items"][5394:]:
            request_data = {
                "id": item["id"],
                "instruction": item["instruction"],
                "image": item["image"],
                "image_folder": image_folder
            }
            try:
                response = requests.post(request_url, json = request_data)            
                response.raise_for_status() 
                result = response.json()
                
                #有一些多余的引号，需要去除：
                cleaned_predict = result["predict"]
                if cleaned_predict.startswith('"""') and cleaned_predict.endswith('"""'):
                    cleaned_predict = cleaned_predict[3:-3]
                elif cleaned_predict.startswith('"') and cleaned_predict.endswith('"'):
                    cleaned_predict = cleaned_predict[1:-1]
                writer.writerow([item['id'], strip_quotes(cleaned_predict)])
                
                print(f"完成: {item['id']}")
            except requests.exceptions.HTTPError as http_err:
                print(f"HTTP error occurred: {http_err}")
            except json.JSONDecodeError as json_err:
                print(f"JSON decode error: {json_err}")
            except Exception as e:
                print(f"Request failed for {item['id']}: {e}")
        print("done")
if __name__ == "__main__":
    process_single_task(input_path, output_path)