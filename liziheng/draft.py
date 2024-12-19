import json

# 读取 JSON 文件
with open('/hy-tmp/project/WWW2025/datasets/test1/test1.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 添加前缀到每个 image 的路径
prefix = "/hy-tmp/project/WWW2025/datasets/test1/images"
for item in data:
    if "image" in item:
        item["image"] = [f"{prefix}/{image}" for image in item["image"]]
        # item["image"] = item["image"][:2]    

# 保存修改后的 JSON 文件
with open('/hy-tmp/project/WWW2025/datasets/test1/test_with_abs_path.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

print("前缀已成功添加并保存到 output.json 文件中。")
