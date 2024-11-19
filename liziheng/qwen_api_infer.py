import requests

url = "http://127.0.0.1:8005/generate"
data = {
    "image_path": "/hy-tmp/project/WWW2025/liziheng/documents/image2.png",
    "text_prompt": "重大工程材料服役安全研究评价设施包含哪些呢"
}

response = requests.post(url, json=data)
print(response.json())
