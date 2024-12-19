from byaldi import RAGMultiModalModel
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from pdf2image import convert_from_path

input_file = "/hy-tmp/project/WWW2025/liziheng/documents/rag_book.pdf"
rag_engine = RAGMultiModalModel.from_pretrained("/root/.cache/huggingface/hub/models--vidore--colpali-v1.2/snapshots/fe2e8900a38a5891530bd08f7c0407471042096d")
model = Qwen2VLForConditionalGeneration.from_pretrained("/root/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/51c47430f97dd7c74aa1fa6825e68a813478097f",
                                                        torch_dtype=torch.bfloat16,
                                                        device_map="auto")

rag_engine.index(
    input_path=input_file,
    index_name="index",
    store_collection_with_index=False,
    overwrite=True
)

text_query = "张彬接着说:“从你身上，我又看到了年轻时的自己，我尽最大的努力去阻止你走这条危险的路，但知道这没有用，你还会在这条路上走下去的。我要告诉你的是，我已经做完我所能做的了。” 说完他疲倦地坐到一个纸箱上。"

results = rag_engine.search(text_query, k=3)
print(results)

images = convert_from_path(input_file)
image_index = results[0]["page_num"] - 1

processor = AutoProcessor.from_pretrained("/root/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/51c47430f97dd7c74aa1fa6825e68a813478097f", max_pixels = 1024 * 28 * 28)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": images[image_index],
            },
            {"type": "text", "text": "这段话发生的情景是什么，接下来怎么样了"},
            {"type": "text", "text": text_query},
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


generated_ids = model.generate(**inputs, max_new_tokens=512)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)