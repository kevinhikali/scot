import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
up_dir = parent_dir
for i in range(3):
    sys.path.append(up_dir)
    up_dir = os.path.dirname(up_dir)
from kutils import DEBUG, INFO, WARN, ERROR
import utils as u
from PIL import Image
import base64
from io import BytesIO
import requests
from flask import Flask, request
from transformers import Blip2Processor, Blip2ForConditionalGeneration

class BLIPWebService:
    def __init__(self, config):
        self.app = Flask(__name__)
        self.config = config
        model_path = self.config['model_path']
        self.processor = Blip2Processor.from_pretrained(model_path)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_path, device_map="cuda")

        self._initialize()

    def _initialize(self):
        @self.app.route('/', methods=['GET', 'POST'])
        def home():
            if request.method == 'POST':
                data = request.get_json()
                sys_prompt = data.get('sys_prompt', '')
                prompt = data.get('prompt', '')
                image_file = data.get('image_file', '')
                decoded_bytes = base64.b64decode(image_file)
                byte_stream = BytesIO(decoded_bytes)
                raw_image = Image.open(byte_stream).convert('RGB')
                inputs = self.processor(raw_image, prompt, return_tensors="pt").to("cuda")
                out = self.model.generate(**inputs)
                res = self.processor.decode(out[0], skip_special_tokens=True)
                return res 
            else:
                # 处理 GET 请求
                return f"Error"

    def run(self):
        self.app.run(host='0.0.0.0', port=self.config.get('port'))


if __name__ == '__main__':
    if len(sys.argv) == 2:
        port = int(sys.argv[1])
    else:
        port = 9122
    config = {
        'port': port,
        'model_path': f'{u.get_nas()}/model/blip2',
    }
    server = BLIPWebService(config)
    server.run()


