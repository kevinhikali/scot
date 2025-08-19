from fastapi import FastAPI, UploadFile, Form
from pydantic import BaseModel
from typing import Optional
from PIL import Image
import io
from typing import List, Union, Optional
from PIL import Image
import io
import torch
import requests
import base64
from transformers import Blip2Processor, Blip2ForConditionalGeneration

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = '/ossfs/workspace/kaiwen/model/blip2'
processor = Blip2Processor.from_pretrained(model_path)
model = Blip2ForConditionalGeneration.from_pretrained(model_path, device_map="auto" if device=="cuda" else None)
model.to(device)

model_name = 'KevinBlip'

class ContentItem(BaseModel):
    type: str  # "text" 或 "image_url" 或 "image_base64"
    text: Optional[str] = None
    image_url: Optional[dict] = None  # {"url": "..."}
    image_base64: Optional[str] = None
    alt_text: Optional[str] = None

class Message(BaseModel):
    role: str  # "user" / "assistant" / "system"
    content: Union[str, List[ContentItem]]

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    messages = req.messages
    question = messages[1].content[0].text
    image_base64 = messages[1].content[1].image_url['url']
    if image_base64.startswith('data:image'): image_base64 = image_base64.split(',', 1)[1]
    missing_padding = len(image_base64) % 4
    if missing_padding: image_base64 += '=' * (4 - missing_padding)
    image_binary = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_binary)).convert("RGB")
    inputs = processor(image, question, return_tensors="pt").to(device)
    outputs = model.generate(**inputs)
    answer = processor.decode(outputs[0], skip_special_tokens=True)

    return {
        "id": f"chatcmpl-{model_name}",
        "object": "chat.completion",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": answer
                }
            }
        ],
        "model": f"{model_name}"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9122)