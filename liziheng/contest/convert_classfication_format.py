import json

# 设置图片的前缀
image_prefix = "/hy-tmp/project/WWW2025/datasets/train/images/"

# 读取原始数据
with open('/hy-tmp/project/WWW2025/datasets/train/train.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 处理数据
modified_data = []
for item in data[301:]:
    instruction = item["instruction"]
    
    system_content = instruction.split("Picture 1: <image>")[1].strip()  # 获取图片描述后的内容
    
    # 用户的内容是 "Picture 1: <image>"
    user_content = "Picture 1: <image>"
    
    # 生成新的结构
    modified_item = {
        "messages": [
            {
                "content": system_content,  # system部分是instruction去除图片后的内容
                "role": "system"
            },
            {
                "content": user_content,  # 用户部分是图片描述
                "role": "user"
            },
            {
                "content": item["output"],  # assistant部分是原output
                "role": "assistant"
            }
        ],
        "images": [image_prefix + item["image"][0]]  # 为图片路径加上前缀
    }
    modified_data.append(modified_item)

# 写入修改后的数据到新的 JSON 文件
with open('/hy-tmp/project/WWW2025/datasets/train/train_sharegpt_format.json', 'w', encoding='utf-8') as f:
    json.dump(modified_data, f, ensure_ascii=False, indent=4)

print("数据已成功转换并保存到 output.json 文件中！")
