from typing import List

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
# from transformers import (
#     Blip2ForConditionalGeneration,
#     Blip2Processor,
# )
from llm_service.ais_requestor import KevinAISRequestor


def get_captioning_fn(caption_model) -> callable:
    kq_config = {
        'api_key': '123',
        'model': caption_model,
        'base_url': "https://agi-pre.alipay.com/api",
        'temperature': 0.0,
        'max_tokens': 4096,
    }
    kq = KevinAISRequestor(kq_config)

    def caption_images(
        images: List[Image.Image],
        prompt: List[str] = None,
        max_new_tokens: int = 32,
    ) -> List[str]:

        prompt = len(images) * ['']

        assert len(images) == len(prompt), "Number of images and prompts must match, got {} and {}".format(len(images), len(prompt))

        captions = []
        for question, img in zip(prompt, images):
            caption = kq.infer('', question, img)
            captions.append(caption)

        return captions

    return caption_images


def get_image_ssim(imageA, imageB):
    # Determine the size to which we should resize
    new_size = max(imageA.size[0], imageB.size[0]), max(
        imageA.size[1], imageB.size[1]
    )

    # Resize images
    imageA = imageA.resize(new_size, Image.LANCZOS)
    imageB = imageB.resize(new_size, Image.LANCZOS)

    # Convert images to grayscale
    grayA = imageA.convert("L")
    grayB = imageB.convert("L")

    # Convert grayscale images to numpy arrays for SSIM computation
    grayA = np.array(grayA)
    grayB = np.array(grayB)

    # Compute the Structural Similarity Index (SSIM) between the two images
    score, _ = ssim(grayA, grayB, full=True)
    return score
