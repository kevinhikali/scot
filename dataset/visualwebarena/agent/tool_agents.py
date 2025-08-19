import asyncio
import io
from playwright.async_api import async_playwright
import os
import imagehash
from PIL import Image
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
	BaseMessage,
	HumanMessage,
	SystemMessage,
)
import base64
import requests
from PIL import Image
from io import BytesIO

class SimJudger:
    def __init__(self,
                 LLM_MODEL_NAME="zg-qw72b-h4",
                 LLM_API_KEY="xx",
                 LLM_BASE_URL="https://agi.alipay.com/api",
                 temperature=0.3,
                 ):
        self.LLM_MODEL_NAME=LLM_MODEL_NAME
        self.LLM_API_KEY=LLM_API_KEY
        self.LLM_BASE_URL=LLM_BASE_URL
        self.temperature=temperature

    def phash_similarity(self, img1, img2):
        h1 = imagehash.phash(img1)
        h2 = imagehash.phash(img2)
        return 1 - (h1 - h2) / 64.0   # 0~1，越接近 1 越相似

    def ahash_similarity(self, img1, img2):
        h1 = imagehash.average_hash(img1)
        h2 = imagehash.average_hash(img2)
        return 1 - (h1 - h2) / 64.0

    def llm_similarity(self, img1,img2):
        llm=ChatOpenAI(
            model=self.LLM_MODEL_NAME,
            api_key=self.LLM_API_KEY,
            base_url=self.LLM_BASE_URL,
            temperature=self.temperature,
        )
        image_data1 = self.image_to_base64(img1)
        image_data2 = self.image_to_base64(img2)
        
        resp=""
        try:
            message = HumanMessage(
                content=[
                    {"type": "text", "text": "给出这两张图片的相似度，直接输出0~1.0中的值"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{image_data1}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{image_data2}"}}
                ]
            )
            input_messages=[message]
            resp=float(llm.invoke(input_messages).content)
        except Exception as e:
            print(e)

        return resp
    
    def image_to_base64(self, img):
        buffered = io.BytesIO()
        img.save(buffered, format=img.format)
        img_byte = buffered.getvalue()
        img_base64 = base64.b64encode(img_byte).decode('utf-8')
        return img_base64

    def url_to_pil_image(self, url):
        response = requests.get(url)
        response.raise_for_status()
        image_data = BytesIO(response.content)
        image = Image.open(image_data)

        return image

    def get_item(self, query_img, page, sim_method="llm_similarity"):
        rows = page.locator('//div[@class="submission__row"]')
        count = rows.count()
        items_li = []
        for idx in range(count):
            row = rows.nth(idx)
            if row.locator('xpath=.//div[@class="submission__inner"]/header/div/h1/a').is_visible():
                title_text   = row.locator('xpath=.//div[@class="submission__inner"]/header/div/h1/a').inner_text()
                title_url    = row.locator('xpath=.//div[@class="submission__inner"]/header/div/h1/a').get_attribute('href')
            else:
                title_text, title_url = None, None
            if row.locator('xpath=..//div[@class="submission__inner"]/header/div/h1/a/img').is_visible():
                item_img     = row.locator('xpath=..//div[@class="submission__inner"]/header/div/h1/a/img').get_attribute('src')
            else:
                item_img = None
            if row.locator('xpath=.//nav[@class="submission__nav"]/ul/li[1]/a/strong').is_visible():
                comments_txt = row.locator('xpath=.//nav[@class="submission__nav"]/ul/li[1]/a/strong').inner_text()
                comments_url = row.locator('xpath=.//nav[@class="submission__nav"]/ul/li[1]/a').get_attribute('href')
            else:
                comments_txt, comments_url = None, None
            if row.locator('xpath=.//div[@class="submission__vote"]/form/span[1]').is_visible():
                vote_text    = row.locator('xpath=.//div[@class="submission__vote"]/form/span[1]').inner_text()
            else:
                vote_text = None
            if row.locator('xpath=.//button[contains(@class,"vote__up")]').is_visible():
                vote_up_bbox   = row.locator('xpath=.//button[contains(@class,"vote__up")]').bounding_box()
            else:
                vote_up_bbox = None
            if row.locator('xpath=.//button[contains(@class,"vote__down")]').is_visible():
                vote_down_bbox = row.locator('xpath=.//button[contains(@class,"vote__down")]').bounding_box()
            else:
                vote_down_bbox = None
            items_li.append({
                    "idx": idx,
                    "title_text": title_text,
                    "title_url": title_url,
                    "item_img": item_img,
                    "comments_text": comments_txt,
                    "comments_url": comments_url,
                    "vote_text": vote_text,
                    "vote_up_bbox": vote_up_bbox,
                    "vote_down_bbox": vote_down_bbox,
                })

        max_sims=0
        max_item=None
        for item in items_li:
            if item["item_img"] is None:
                continue
            # item_img_path="http://localhost:9999"+item["item_img"]
            item_img_path=item["title_url"]
            img2=self.url_to_pil_image(item_img_path)
            if sim_method=="phash_similarity":
                sims=self.phash_similarity(query_img,img2)
            elif sim_method=="ahash_similarity":
                sims=self.ahash_similarity(query_img,img2)
            else:
                sims=self.llm_similarity(query_img,img2)
            item["sims"]=float(sims)
            if item["sims"]>max_sims:
                max_sims=item["sims"]
                max_item=item

        return max_sims, max_item


    
    
