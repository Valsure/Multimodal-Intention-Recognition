import os
import torch
import torch.distributed as dist
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# 设置环境变量，用于torch.distributed
os.environ['MASTER_ADDR'] = 'localhost'  # 设置主节点地址，如果是分布式训练，将其设置为主节点IP
os.environ['MASTER_PORT'] = '12355'  # 设置一个未被占用的端口

# 初始化分布式进程组
dist.init_process_group("nccl", rank=0, world_size=3)  # 设置rank和world_size

local_model_path = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/51c47430f97dd7c74aa1fa6825e68a813478097f")

# 设置设备映射，确保模型张量在多卡间并行分布
model = Qwen2VLForConditionalGeneration.from_pretrained(
    local_model_path, torch_dtype="auto", device_map="balanced",  # 可以改成 "sequential" 或 "auto" 根据需要调优
)

processor = AutoProcessor.from_pretrained(local_model_path)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "描述一下图片"},
        ],
    }
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# 使用张量并行生成推理结果
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=128)

# 清理特殊token并解码输出
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)

# 清理分布式进程组
dist.destroy_process_group()
