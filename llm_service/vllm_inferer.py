import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
up_dir = parent_dir
for i in range(3):
    sys.path.append(up_dir)
    up_dir = os.path.dirname(up_dir)
from kutils import DEBUG, INFO, WARN, ERROR
import utils as u
from dataclasses import asdict
from typing import NamedTuple, Optional
from PIL.Image import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLProcessor
if u.get_os() != 'mac':
    from vllm import LLM, SamplingParams
    from vllm import LLM, EngineArgs, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.multimodal.utils import fetch_image

class VLLMInferer:
    def __init__(self, model_type, model_path, seed = 0, force_bos_token = None, limit_mm_per_prompt = 5) -> None:
        import os
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        self.seed = seed
        engine_args = EngineArgs(
            model=model_path,
            max_model_len=32768 if process_vision_info is None else 4096,
            max_num_seqs=5,
            limit_mm_per_prompt={"image": limit_mm_per_prompt},
        )

        engine_args = asdict(engine_args) | {"seed": seed}
        self.llm = LLM(**engine_args)
        self.sampling_params = SamplingParams(
            temperature = 0.0,
            max_tokens = 4096,
        )

        if model_type == 'qwen_2_5_vl':
            # self.processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
            self.processor = AutoProcessor.from_pretrained(model_path)
        else: self.processor = AutoProcessor.from_pretrained(model_path)

    def infer_prompt(self, sys_prompt, prompt, img_files):
        return self.infer(sys_prompt, prompt, img_files)
    
    def infer_vl2(self, messages):
        raise ValueError('Not implement')
                    
    def infer_messages(self, messages, img_files):
        raise ValueError('Not implement')

    def infer(self, sys_prompt: str, prompt: str, image_urls: list[str]):
        placeholders = [{"type": "image", "image": url} for url in image_urls]
        messages = [{
            "role": "system",
            "content": sys_prompt
        }, {
            "role":
            "user",
            "content": [
                *placeholders,
                {
                    "type": "text",
                    "text": prompt
                },
            ],
        }]

        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

        if process_vision_info is None:
            image_data = [fetch_image(url) for url in image_urls]
        else:
            image_data, _ = process_vision_info(messages, return_video_kwargs=False)

        outputs = self.llm.generate(
            {
                "prompt": prompt,
                "multi_modal_data": {
                    "image": image_data
                },
            },
            sampling_params=self.sampling_params,
            lora_request=None,
        )

        generated_text = outputs[0].outputs[0].text
        return generated_text, messages

if __name__ == '__main__':
    if u.get_os() == 'mac': NAS_PATH = f'{u.get_home()}/data/'
    else: NAS_PATH = '/mnt/agent-s1/common/public/kevin/'
    model_rel_path = 'output'
    model_name = '20250610153514_Qwen2.5-VL-7B-Instruct-SCOT_sft_sw0_pt/checkpoint-100'
    model_path = f'{NAS_PATH}/{model_rel_path}/{model_name}'
    img_file = '/ossfs/workspace/kevin_git/qwenvl25/qwen-vl-finetune/demo/images/COCO_train2014_000000580957.jpg'

    QUESTION = "图片里有什么"
    image_urls = [img_file]
    vi = VLLMInferer('qwen2_5_vl', model_path)
    response, messages = vi.infer('You are a helpful assistant.', QUESTION, image_urls)
    DEBUG(response)
