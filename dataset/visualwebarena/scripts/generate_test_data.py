import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
up_dir = parent_dir
for i in range(4):
    sys.path.append(up_dir)
    up_dir = os.path.dirname(up_dir)
from kutils import DEBUG, INFO, WARN, ERROR
import utils as u

"""Replace the website placeholders with website domains from env_config
Generate the test data"""
import json
import os

from browser_env.env_config import *

if __name__ == "__main__":
    DATASET = os.environ["DATASET"]
    if DATASET == "webarena":
        print("DATASET: webarena")
        print(f"REDDIT: {REDDIT}")
        print(f"SHOPPING: {SHOPPING}")
        print(f"SHOPPING_ADMIN: {SHOPPING_ADMIN}")
        print(f"GITLAB: {GITLAB}")
        print(f"WIKIPEDIA: {WIKIPEDIA}")
        print(f"MAP: {MAP}")
        print(f"HOMEPAGE: {HOMEPAGE}")
        inp_paths = ["config_files/wa/test_webarena.raw.json"]
        replace_map = {
            "__REDDIT__": REDDIT,
            "__SHOPPING__": SHOPPING,
            "__SHOPPING_ADMIN__": SHOPPING_ADMIN,
            "__GITLAB__": GITLAB,
            "__WIKIPEDIA__": WIKIPEDIA,
            "__MAP__": MAP,
            "__HOMEPAGE__": HOMEPAGE,
        }
    elif DATASET == "visualwebarena":
        print("DATASET: visualwebarena")
        print(f"CLASSIFIEDS: {CLASSIFIEDS}")
        print(f"REDDIT: {REDDIT}")
        print(f"SHOPPING: {SHOPPING}")
        print(f"HOMEPAGE: {HOMEPAGE}")
        inp_paths = [
            "config_files/vwa/test_classifieds.raw.json", 
            "config_files/vwa/test_shopping.raw.json", 
            "config_files/vwa/test_reddit.raw.json",
        ]
        replace_map = {
            "__REDDIT__": REDDIT,
            "__SHOPPING__": SHOPPING,
            "__WIKIPEDIA__": WIKIPEDIA,
            "__CLASSIFIEDS__": CLASSIFIEDS,
            "__HOMEPAGE__": HOMEPAGE,
        }
    else:
        raise ValueError(f"Dataset not implemented: {DATASET}")

    data_path = f'{u.get_nas()}/gui_dataset/visualwebarena/'
    auth_path = f'{data_path}/auth/'
    u.mkdir(auth_path)

    for inp_path in inp_paths:
        inp_path = data_path + inp_path
        output_dir = inp_path.replace('.raw.json', '')
        os.makedirs(output_dir, exist_ok=True)
        with open(inp_path, "r") as f:
            raw = f.read()
        for k, v in replace_map.items():
            raw = raw.replace(k, v)

        with open(inp_path.replace(".raw", ""), "w") as f:
            f.write(raw)
        data = json.loads(raw)

        for idx, item in enumerate(data):
            with open(os.path.join(output_dir, f"{idx}.json"), "w") as f:
                if 'image' in item.keys():
                    image_path = item['image']
                    if isinstance(image_path, str):
                        item['image'] = data_path + image_path[image_path.find('static'):]
                    elif isinstance(image_path, list):
                        if image_path:
                            for i in range(len(image_path)):
                                item['image'][i] = data_path + image_path[i][image_path[i].find('static'):]
                storage_state = item['storage_state']
                storage_name = u.get_name(storage_state)
                storage_state = f'{auth_path}/{storage_name}'
                item['storage_state'] = storage_state
                json.dump(item, f, indent=2)
