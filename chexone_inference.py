import os

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

print("Loading CheXOne model (first run downloads ~8GB, may take several minutes)...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "StanfordAIMI/CheXOne", torch_dtype="auto", device_map="auto"
)
print(f"Model loaded on {next(model.parameters()).device}")

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image scenarios.
# Requires CUDA; not supported on Mac MPS.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "StanfordAIMI/CheXOne",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
# processor = AutoProcessor.from_pretrained("StanfordAIMI/CheXOne")

# The default range for the number of visual tokens per image in the model is 4-16384.
# We recommand to set max_pixels=512*512 to align with the training setting.
min_pixels = 256*28*28
max_pixels = 512*512
print("Loading processor...")
processor = AutoProcessor.from_pretrained("StanfordAIMI/CheXOne", min_pixels=min_pixels, max_pixels=max_pixels)

# Inference Mode: Reasoning
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://raw.githubusercontent.com/YBZh/CheXOne/main/asset/cxr.jpg",
            },
            {"type": "text", "text": "Write an example findings section for the CXR. Please reason step by step, and put your final answer within \\boxed{{}}."},
        ],
    }
]

# Inference Mode: Instruct
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": "https://raw.githubusercontent.com/YBZh/CheXOne/main/asset/cxr.jpg",
#             },
#             {"type": "text", "text": "Write an example findings section for the CXR."},
#         ],
#     }
# ]

print("Preparing inputs...")
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
device = next(model.parameters()).device
inputs = inputs.to(device)

print(f"Running inference on {device}...")
generated_ids = model.generate(**inputs, max_new_tokens=256)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print("\n--- CheXOne output ---")
print(output_text[0] if output_text else output_text)
