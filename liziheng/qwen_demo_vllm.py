from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import os
 
MODEL_PATH = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/51c47430f97dd7c74aa1fa6825e68a813478097f")
IMAGE_PATH = '/hy-tmp/project/WWW2025/liziheng/documents/image_test.jpg'
 
llm = LLM(
    model=MODEL_PATH,
    limit_mm_per_prompt={'image': 10, 'video': 10},
    tensor_parallel_size=2
)
 
sampling_params = SamplingParams(
    temperature=0.1, top_p=0.001, repetition_penalty=1.05, max_tokens=256,
    stop_token_ids=[],
)
 
messages = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': [
        {
            'type': 'image',
            'image': IMAGE_PATH,
 
            # min_pixels & max_pixels are optional
            'max_pixels': 12845056,
        },
 
        {
            'type': 'text',
            'text': '描述一下图片',
        },
    ]},
]
 
processor = AutoProcessor.from_pretrained(MODEL_PATH)
prompt = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
)
image_inputs, video_inputs = process_vision_info(messages)
 
mm_data = {}
if image_inputs is not None:
    mm_data['image'] = image_inputs
if video_inputs is not None:
    mm_data['video'] = video_inputs
 
llm_inputs = {
    'prompt': prompt,
    'multi_modal_data': mm_data,
}
 
outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
generated_text = outputs[0].outputs[0].text
 
print(generated_text)