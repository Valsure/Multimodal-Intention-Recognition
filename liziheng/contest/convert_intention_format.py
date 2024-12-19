import json

# 设置图片的前缀
image_prefix = "/hy-tmp/project/WWW2025/datasets/train/images/"

# 读取原始数据
with open('/hy-tmp/project/WWW2025/datasets/train/train.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 处理数据
modified_data = []
for item in data[:300]:
    instruction = item["instruction"]
    
    # 提取出system和user的对话内容
    start_idx = instruction.find("<用户与客服的对话 START>")
    end_idx = instruction.find("<用户与客服的对话 END>") + len("<用户与客服的对话 END>")
    system_content = instruction[:start_idx] + instruction[end_idx:].strip()
    
    user_content = instruction[start_idx + len("<用户与客服的对话 START>"):end_idx - len("<用户与客服的对话 END>")].strip()
    
    # 为images路径加上前缀
    images_with_prefix = [image_prefix + img for img in item["image"]]
    
    # 生成新的结构
    modified_item = {
        "messages": [
            {
                "content": system_content.strip(),
                "role": "system"
            },
            {
                "content": user_content.strip(),
                "role": "user"
            },
            {
                "content": item["output"],
                "role": "assistant"
            }
        ],
        "images": images_with_prefix  # 加上前缀后的图片路径
    }
    modified_data.append(modified_item)

# 写入修改后的数据到新的 JSON 文件
with open('/hy-tmp/project/WWW2025/liziheng/documents/draft.json', 'w', encoding='utf-8') as f:
    json.dump(modified_data, f, ensure_ascii=False, indent=4)

print("数据已成功转换并保存到 output.json 文件中！")
