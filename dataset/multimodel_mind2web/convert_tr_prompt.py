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
import pandas as pd
import copy
import json
from infer_bbox_250714 import prompt_f


def convert_250714(annos):
    new_annos = copy.deepcopy(annos)
    for i in tqdm(range(len(annos))):
        ori_prompt = annos[i]['conversations'][0]['value']
        task = u.extract_text(ori_prompt, 'Your primary task:', '\nYour action history:')[0]
        action_history = u.extract_text(ori_prompt, 'Your action history:', '\nYou should output the best action that is most likely to complete the task.')[0]
        action_history = json.loads(action_history)
        action_history_str = ''
        for i, step_content in enumerate(action_history):
            action_history_str += f'step {i}: {step_content}\n'
        action_history_str = action_history_str[:-1]
        new_annos[i]['conversations'][0]['value'] = prompt_f.format(task, action_history_str)
    return new_annos


if __name__ == "__main__":
    train_path = f'{u.get_nas()}/gui_dataset/MultiModel_Mind2Web/train/'
    anno_name = 'anno_250624_4o'
    save_name = 'anno_250714_4o'
    u.mkdir(f'{train_path}/{save_name}')

    ori_file = f'{train_path}/{anno_name}/train.json'
    annos = u.read_json(ori_file)
    new_annos = convert_250714(annos)
    new_file = f'{train_path}/{save_name}/train.json'
    u.write_json(new_file, new_annos, encoding='utf-8')

    ori_file = f'{train_path}/{anno_name}/val.json'
    annos = u.read_json(ori_file)
    new_annos = convert_250714(annos)
    new_file = f'{train_path}/{save_name}/val.json'
    u.write_json(new_file, new_annos, encoding='utf-8')
