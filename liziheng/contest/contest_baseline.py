from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import os
import json

app = FastAPI()

MODEL_PATH = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/51c47430f97dd7c74aa1fa6825e68a813478097f")

# Initialize model
llm = LLM(
    model=MODEL_PATH,
    limit_mm_per_prompt={'image': 10, 'video': 10},
    tensor_parallel_size=2
)

processor = AutoProcessor.from_pretrained(MODEL_PATH)

# Define request schema
class GenerateRequest(BaseModel):
    id: str
    image: str
    instruction: str
    image_folder: str

class BatchRequest(BaseModel):
    items: list[GenerateRequest]
    image_folder: str

class PredictionResult(BaseModel):
    id: str
    predict: str

@app.post("/single_predict", response_model=PredictionResult)
def singe_predict(request: GenerateRequest):
    message = [
        {'role':'system', 'content':'You are a helpful assistant.'},
        {'role':'user', 'content':[
            {'type': 'image', 'image':os.path.join(request.image_folder, request.image)},
            {'type': 'text', 'text': request.instruction}
        ]}
    ]
    prompt = processor.apply_chat_template(message, tokenize = False, add_generation_prompt = True)
    image_inputs ,_ = process_vision_info(message)
    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    
    sampling_params = SamplingParams(
        temperature=0.1, top_p=0.001, repetition_penalty=1.05, max_tokens=256, stop_token_ids=[]
    )
    llm_inputs = {'prompt': prompt, 'multi_modal_data': mm_data}
    outputs = llm.generate([llm_inputs], sampling_params = sampling_params)
    generated_text = outputs[0].outputs[0].text.strip()
    # print(generated_text)
    return PredictionResult(id = request.id, predict = generated_text)
    
    
@app.post("/batch_predict", response_model=list[PredictionResult])
async def batch_predict(request: BatchRequest):
    results = []
    for item in request.items:
        # Build the prompt for each item
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': [
                {'type': 'image', 'image': os.path.join(request.image_folder, item.image[0])},
                {'type': 'text', 'text': item.instruction},
            ]},
        ]

        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)

        mm_data = {}
        if image_inputs is not None:
            mm_data['image'] = image_inputs

        sampling_params = SamplingParams(
            temperature=0.1, top_p=0.001, repetition_penalty=1.05, max_tokens=256, stop_token_ids=[]
        )

        llm_inputs = {'prompt': prompt, 'multi_modal_data': mm_data}
        outputs = llm.generate([llm_inputs], sampling_params=sampling_params)

        if outputs and outputs[0].outputs:
            # Process generated text to extract predicted label
            generated_text = outputs[0].outputs[0].text.strip()
            results.append(PredictionResult(id=item.id, predict=generated_text))
        else:
            raise HTTPException(status_code=500, detail=f"Failed to generate text for item ID {item.id}")

    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
