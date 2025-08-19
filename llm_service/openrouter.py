import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
up_dir = parent_dir
for i in range(3):
    sys.path.append(up_dir)
    up_dir = os.path.dirname(up_dir)
from kutils import DEBUG, INFO, WARN, ERROR
import utils as u
from openai import OpenAI
import os
from PIL import Image
from ais_requestor import KevinAISRequestor

if __name__ == "__main__":
    API_KEY = os.getenv("OPENROUTER_KEY")
    kq_config = {
        'api_key': API_KEY,
        'model': 'openai/gpt-5-chat',
        'base_url': "https://openrouter.ai/api/v1",
        'temperature': 0.0,
        'max_tokens': 4096,
    }

    kq = KevinAISRequestor(kq_config)
    prompt = ""
    image_file = f'{u.get_git()}/0_saved_methods/qwenvl25/cookbooks/assets/universal_recognition/unireco_bird_example.jpg'
    img = Image.open(image_file)
    response = kq.infer('', prompt, img)
    DEBUG(response)