from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import torch
from accelerate import infer_auto_device_map
local_model_path = os.path.expanduser("/hy-tmp/project/WWW2025/models/qwen2_by_liziheng")

model = Qwen2VLForConditionalGeneration.from_pretrained(local_model_path, torch_dtype="auto", device_map="auto")

processor = AutoProcessor.from_pretrained(local_model_path)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/hy-tmp/project/WWW2025/datasets/train/images/0a856d3d-6939-4a3b-b80c-46ff0ddb324d-461-0.jpg",
            },
            # {"type": "text", "text": "描述一下图片"},
        ],
        
        # "content":
        #     [
        #         {
        #             "type": "text", "text": "你是谁"
        #         }
        #     ]
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


# 生成推理结果
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)



