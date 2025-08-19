import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
up_dir = parent_dir
for i in range(3):
    sys.path.append(up_dir)
    up_dir = os.path.dirname(up_dir)
from kutils import DEBUG, INFO, WARN, ERROR
import utils as u
from tqdm import tqdm
import pyarrow.json as pajson
import pandas as pd
import json
import ast
from datasets import load_dataset

if __name__ == "__main__":
    dataset_name = 'MultiModel_Mind2Web'
    main_path = f'{u.get_home()}/kaiwen/gui_dataset/{dataset_name}'

    ds = load_dataset(main_path)
    DEBUG(ds)

    col_names = ds.column_names
    DEBUG(col_names)

    output_path = f'{main_path}/raw/'
    u.mkdir(output_path)
    annotations = []
    raw_data = []
  
    for split, content in tqdm(ds.items()):
        if split != 'test_task': continue
        img_path = f'{output_path}/imgs/'
        u.mkdir(img_path)
        cols = col_names[split]
        del cols[1]
        del cols[1]
        del cols[9]
        del cols[3]
        raw_data = {}
        raw_file = f'{output_path}/raw_{split}.json'
        if u.is_file_exist(raw_file): raw_data = u.read_json(raw_file)
        for item in tqdm(content, split):
            anno_id = item['annotation_id']
            action_uid = item['action_uid']
            if anno_id in raw_data.keys(): continue
            img = item['screenshot']
            del item['screenshot']
            if action_uid == '61c2ca43-6663-4af1-8966-be9e621b41a4':
                DEBUG(item.keys())
                u.print_json(item)
                exit()
            if not img: continue
            raw_data[anno_id] = {}
            img_file = f'{img_path}/{anno_id}_{action_uid}.png'
            img.save(img_file)
            for col in cols:
                raw_data[anno_id][col] = item[col]
            u.write_json(raw_file, raw_data, indent = 4, encoding = 'utf-8')
