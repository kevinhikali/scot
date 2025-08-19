import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
up_dir = parent_dir
for i in range(3):
    sys.path.append(up_dir)
    up_dir = os.path.dirname(up_dir)
from kutils import DEBUG, INFO, WARN, ERROR
import utils as u
from datasets import load_dataset
from tqdm import tqdm
import pyarrow.json as pajson
from PIL import Image
import pandas as pd
import re
import json
import copy
from dataset.multimodel_mind2web.infer import MMInfer

def convert_action_to_string(action):
    action_type = action.get("action_type")

    if action_type == "click":
        x, y = action['x'], action['y']
        return f"click(start_box='<|box_start|>({x},{y})<|box_end|>')"

    elif action_type == "long_press":
        x, y = action['x'], action['y']
        return f"long_press(start_box='<|box_start|>({x},{y})<|box_end|>')"

    elif action_type == "type_text":
        text = action['text']
        return f"type(content='{text}')"

    elif action_type == "scroll":
        direction = action['direction']
        return f"scroll(direction='{direction}')"

    elif action_type == "open_app":
        app_name = action['app_name']
        return f"open_app(app_name='{app_name}')"

    elif action_type == "navigate_back":
        return "press_back()"

    elif action_type == "wait":
        return "wait()"

    elif action_type == "status":
        return "finished()"

    else:
        return "# Unknown action"

def find2(s):
    match = re.search(r'<\|box_start\|>\(([^,]+),([^)]+)\)<\|box_end\|>', s)
    if match:
        x1 = match.group(1)
        y1 = match.group(2)
        return [int(float(x1)), int(float(y1))]
    else:
        return [0, 0]

def build_multi_turn_prompt(task, action_desc_history, action_history, img_files):
    sys_prompt = {
        "role": "system",
        "content": "You are a helpful assistant."
    }

    user_prompt = {
        "role": "user",
        "content": \
f"""You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format

Thought: ...
Action: ...

## Action Space
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
long_press(start_box='<|box_start|>(x1,y1)<|box_end|>', time='')
type(content='', start_box='<|box_start|>(x1,y1)<|box_end|>')
scroll(direction='down or up or right or left')
open_app(app_name='')
press_back()
press_home()
wait()
finished() # Submit the task regardless of whether it succeeds or fails.


## Note
- Use English in Thought part.
- Summarize your next action (with its target element) in one sentence in Thought part.

## User Instruction
{task}"""
    }

    conversation = [sys_prompt, user_prompt]

    if len(action_history) != len(action_desc_history):
        DEBUG(img_files[-1])
        ERROR(f'len(action_history) != len(action_desc_history)')

    for i in range(len(action_history)):
        thought = action_desc_history[i]
        action_str = action_history[i]
        img_file = img_files[i]
        conversation.append({
            "role": "user",
            "content": [{"type": "image", "image": f"file:///{img_file}"}],
        })
        conversation.append({
            "role": "assistant",
            "content": f"Thought: {thought}\nAction: {action_str}"
        })

    conversation.append({
        "role": "user",
        "content": [{"type": "image", "image": f"file:///{img_files[-1]}"}],
    })

    return sys_prompt, user_prompt, conversation

def parse_response(response):
    pred_action_history = ''
    pred_action_description = ''
    pred_action = ''
    pred_action_type = 'UNKNOWN'
    pred_bbox = [0, 0, 0, 0]
    pred_type_value = ''
    pred_click_point = [0, 0]

    error_msg = ''
    try: 
        pred_action_description = u.extract_text(response, 'Thought: ', '\nAction: ')[0]
        pred_action = u.extract_text(response, 'Action: ', '<|im_end|>')[0]
    except Exception as e: error_msg += f'{response}, error at extract text {e}\n'
    try: 
        pred_click_point = find2(pred_action)
        pred_click_point[0] = int(pred_click_point[0] * 1.28)
    except Exception as e: error_msg += f'{response}, error at find2 {e}\n'

    try:
        if pred_action.startswith('click'):
            pred_action_type = 'CLICK'
            pred_type_value = ''
        elif pred_action.startswith('select'):
            pred_action_type = 'SELECT'
            pred_type_value = pred_action.split('content=')[1].split("'")[1]
        elif pred_action.startswith('type'):
            pred_action_type = 'TYPE'
            pred_type_value = pred_action.split('content=')[1].split("'")[1]
        else:
            pred_action_type = 'UNKNOWN'
            pred_type_value = ''
    except:
        pred_action_type = 'UNKNOWN'
        pred_type_value = ''

    res = {
        'pred_action_history': pred_action_history,
        'pred_action_description': pred_action_description,
        'pred_action': pred_action,
        'pred_action_type': pred_action_type,
        'pred_bbox': pred_bbox,
        'pred_type_value': pred_type_value,
        'pred_click_point': pred_click_point,
        'parse_error_msg': error_msg,
    }
    return res

if __name__ == "__main__":
    mmi = MMInfer(sys.argv)
    mmi.infer('vl2', parse_response, 
        {
            'message_builder': build_multi_turn_prompt
        }
    )

#     response = \
# f'''Thought: "COMPARE" was clicked
# Action: click(start_box='<|box_start|>(785,273)<|box_end|>')<|im_end|>'''
# #     response = \
# # f'''Thought: "NEXT" was clicked
# # Action: click(start_box='<|box_start|>(960.5,433)<|box_end|>')<|im_end|>'''
# #     response = \
# # f'''Thought: "Search" was clicked
# # Action: click(start_box='<|box_start|>(871.0,316)<|box_end|>')<|im_end|>'''
#     res = parse_response(response)
#     u.print_json(res)
