import requests

url = "http://127.0.0.1:8005/generate"
data = {
    "image_path": "/hy-tmp/project/WWW2025/liziheng/documents/image_test.jpg",
    "text_prompt": "描述一下图片"
}

response = requests.post(url, json=data)
print(response.json())
