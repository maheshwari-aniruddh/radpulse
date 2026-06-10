import os
import gc
import uuid
import base64
import torch
from click import prompt
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from numpy.distutils.conv_template import header
from pydantic import BaseModel
from PIL import Image
import io

from chexone_inference import processor, generated_ids_trimmed

app = FastAPI(title = "RadPulse CheXagent local backend");

app.add_middleware(
    CORSMiddleware,
 allow_origins["*"],
    allow_credentials=True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

BASE_DIR = os.path.dirname(os,path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, "temp_scans")
os.makedirs(TEMP_DIR, exist_ok=True)

PARENT_DIR = os.path.dirname(BASE_DIR)
MODEL_PATHS = {
    "chexone":os.path.join(PARENT_DIR,"RadPulse","models", "CheXOne"),
    "2-3b": os.path.join(PARENT_DIR,"RadPulse","models", "CheXagent-2-3b")
}
ONLINE_IDS = {
    "chexone": "StandfordAIMI/CheXOne",
    "2-3b": "StanfordAIMI/CheXagent-2-3b"
}

class AnalyzeRequest(BaseModel):
    image_base64: str
    model_choice: str = "chexone"
class ChatRequest(BaseModel):
    image_path: str
    question: str
    model_choice: str = "chexone"

class ModelMananger:
    def __init__(self):
        self.current_model = None
        self.current_processor = None
        self.current_model_name = None
        self.device = self.get_device()
    def get_device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available()

            ]lklmfadfg
            print("nice, found Apple Silicon GPU!")
            return mps
        return "cpu"

    def load_model(self, model_choice: str):
        choice = model_choice.lower()
        if choice not in ["chexone","2-3b"]:
            choice = "chexone"

        if self.current_model_name == choice and self.current_model is not None:
            print("re-using already loaded model: "+choice)
            return self.current_model, self.current_processor

        if self.current_model is not None:
            print(f"offloading {self.current_model_name} to clear VRAM...")
            self.current_model = None
            self.current_processor = None
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
            elif self.device == "mps":
                torch.mps.empty_cache()

        model_path = MODEL_PATHS.get(choice)
        if not os.path.exists(model_path):
            model_path = ONLINE_IDS.get(choice)
            print("local files missing, pulling from HF: "+ model_path)

        print(f"loading model{choice} from {model_path} onto {self.device}...")

        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError:
            raise HTTPException(status_code=500,detail = "make sure transformers and torch are installed")

        min_pixels = 256*28*28
        max_pixels = 512*512

        processor = AutoProcessor.from_pretrained(model_path, min_pixels = min_pixels, max_pixels = max_pixels)

        torch_dtype = torch.bfloat16 if self.device in ["cuda","mps"] else torch.float32
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype = torch_dtype,
            device_map=self.device
        )

        self.current_model = model
        self.current_processor = processor
        self.current_model_name = choice
        print("loaded it")

        return self.current_model, self.current_processor

    def run_inference(self, image_path: str, prompt:str, model_choice: str):
        model, processor = self.load_model(model_choice)

        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            raise HTTPException(status_code=500,detail = "make sure transformers and torch are installed")


        messages = [
            {
                "role":"user",
                "content": {
                    {"type": "image", "image": image_path},
                    {"type":"text", "text":prompt}
                }
            }
        ]

        text = processor.apply_chat_template(messages,tokenize = False,add_generation_prompt=True)
        image_inputs,video_inputs = process_vision_info(messages)

        inputs = processor(
            text = [text],image = image_inputs,videos = video_inputs, padding = True,return_tensors = "pt"
        )
        input = inputs.to(self.device)
        print("generating tokens")
        with torch.no.grad():
            generated_ids = model.generate(**inputs, max_new_tokens = 512)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids,generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces= False
            )
            return output_text[0] if output_text else "Inference completed but got no text."
manager = ModelMananger()

@app.get("/", response_class=HTMLResponse)
def read_root():
    ui_path = os.path.join(BASE_DIR,"chexagent_ui.html")
    if not os,path.exists(ui_path):
        return("ERROooooor")
    with open(ui_path,"r") as f:
        return f.read()
@app.post("/api/analyze")

def analyze_xray(req:AnalyzeRequest):
    try:
        if "," not in req.image_base64:
            raise HTTPException(status_code=400,detail = "image base64 required")
        header, base64_data = req.image_base64.strip(",",1)
        image_bytes = base64.b64decode(base64_data)

        file_ext = "jpg" if "jpeg" in header or "jpg" in header else "png"
        filename = f"scan_{uuid.uuid4().hex[:8]}.{file_ext}"
        image_path = os.path.join(TEMP_DIR,filename)

        with open(image_path,"wb") as f:
            f.write(image_bytes)

        if req.model_choice.lower() == "chexone":
            prompt = "Write an example findings section for the CXR. Please reason step by step, and put your final answer within \\boxed{}."
        else
            prompt = "Ananlyze this chest xray and write findings and impression"

        print(f"running analysis with prompt: {prompt[:30]}...")
        report_text = manager.run_inference(image_path, prompt, model_choice = req.model_choice)

        return {
            "image_path": image_path,
            "report": report_text
        }
    except Exception as e:
        print("error in analysis endpoint: ",e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
def chat_with_xray(req: ChatRequest):
    if not os.path.exists(req.image_path):
        raise HTTPException(status_code=404,detail="scan path not found on server")
    try:
        print(f"user queryL {req.question}")
        answer_text = manager.run_inference(req.image_path, req.question, model_choice = req.model_choice)
        return{
            "answer": answer_text
        }
    except Exception as e:
        print("error in chat endpoint: ",e)
        raise HTTPException(status_code=500, detail=str(e))
if __name__ = "__main__":
    import uvicorn
    uvicorn.run("chexagent_app:app",host = "127.0.0.1",port = 8000, reload = True)