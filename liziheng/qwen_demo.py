from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import torch
from accelerate import infer_auto_device_map
local_model_path = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/51c47430f97dd7c74aa1fa6825e68a813478097f")

model = Qwen2VLForConditionalGeneration.from_pretrained(local_model_path, torch_dtype="auto", device_map="auto")

processor = AutoProcessor.from_pretrained(local_model_path)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/hy-tmp/project/WWW2025/liziheng/documents/image_test.jpg",
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

# 记录显存分配情况
def print_memory_usage():
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_allocated() / (1024 ** 2)
    print(f"Allocated Memory: {allocated:.2f} MB")
    print(f"Reserved Memory: {reserved:.2f} MB")

print("Before inference:")
print_memory_usage()

# 生成推理结果
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)

print("After inference:")
print_memory_usage()
