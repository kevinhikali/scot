import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
up_dir = parent_dir
for i in range(3):
    sys.path.append(up_dir)
    up_dir = os.path.dirname(up_dir)
from kutils import DEBUG, INFO, WARN, ERROR
import utils as u
from infer_utils import parse_type
from dataset_utils import find4, find2

pu_bbox_prompt_f = \
'''You are in a sequential mission to help the user to finish a multi-step task. You will be given serveral GUI screenshots including current screenshot, a primary task and your previous action history.

The actions you can take are listed below:
click(start_box='<|box_start|>[x1, y1, x2, y2]<|box_end|>') # click the element in box x1, y1, x2, y2
type(content='', start_box='<|box_start|>[x1, y1, x2, y2]<|box_end|>') # type in the input box at coordinate x1, y1, x2, y2
select(content='', start_box='<|box_start|>[x1, y1, x2, y2]<|box_end|>') # select content in the radio button at coordinate x1, y1, x2, y2

Your primary task:
{}

Your action history:
{}

You should output the best action that is most likely to complete the task.
```'''

pu_pxxt_prompt_f = \
'''You are in a sequential mission to help the user to finish a multi-step task. You will be given serveral GUI screenshots including current screenshot, a primary task and your previous action history.

The actions you can take are listed below:
click(start_box='<|box_start|>[x, y]<|box_end|>') # click the element at point x, y
type(content='', start_box='<|box_start|>[x, y]<|box_end|>') # type in the input box at coordinate x, y
select(content='', start_box='<|box_start|>[x, y]<|box_end|>') # select content in the radio button at coordinate x, y

The screenshot resolution is [1280x1000].

Your primary task:
{}

Your action history:
{}

You should output the best action that is most likely to complete the task.
```'''

ori_bbox_prompt_f = \
'''You are in a sequential mission to help the user to finish a multi-step task. You will be given serveral GUI screenshots including current screenshot, a primary task and your previous action history.

The actions you can take are listed below:
click(start_box='<|box_start|>[x1, y1, x2, y2]<|box_end|>') # click the element in box x1, y1, x2, y2
type(content='', start_box='<|box_start|>[x1, y1, x2, y2]<|box_end|>') # type in the input box at coordinate x1, y1, x2, y2
select(content='', start_box='<|box_start|>[x1, y1, x2, y2]<|box_end|>') # select content in the radio button at coordinate x1, y1, x2, y2

Your primary task:
{}

Your action history:
{}

You should output the best action that is most likely to complete the task.
Output your results in the following json format:
```json
{{
    "action_description": "",
    "action": ""
}}
```'''

ori_pxxt_prompt_f = \
'''You are in a sequential mission to help the user to finish a multi-step task. You will be given serveral GUI screenshots including current screenshot, a primary task and your previous action history.

The actions you can take are listed below:
click(start_box='<|box_start|>[x, y]<|box_end|>') # click the element at point x, y
type(content='', start_box='<|box_start|>[x, y]<|box_end|>') # type in the input box at coordinate x, y
select(content='', start_box='<|box_start|>[x, y]<|box_end|>') # select content in the radio button at coordinate x, y

The screenshot resolution is [1280x1000].

Your primary task:
{}

Your action history:
{}

You should output the best action that is most likely to complete the task.
Output your results in the following json format:
```json
{{
    "action_description": "",
    "action": ""
}}
```'''

sys_prompt = '''You are a helpful assistant.'''

job_desc = \
'You are in a sequential mission to finish a multi-step task or answer a question by the browser. You will be given current screenshot, a primary task and your previous action history.'

format_desc = \
'''You should output the best action that is most likely to complete the task.
Output your results in the following json format:
```json
{{
    "action_description": "",
    "action": ""
}}
```'''

format_desc_vwa = \
'''Output the best action that is most likely to complete the task and contents you want to write in your memo (contents you think may help yourself in future steps) in the following json format:
```json
{{
    "action_description": "",
    "action": "",
    "content_to_memo": ""
}}
```'''

action_space_bbox_mmmw = \
'''The actions you can take are listed below:
click(start_box='<|box_start|>[x1, y1, x2, y2]<|box_end|>') # click the element in box x1, y1, x2, y2
type(content='', start_box='<|box_start|>[x1, y1, x2, y2]<|box_end|>') # type in the input box at coordinate x1, y1, x2, y2
select(content='', start_box='<|box_start|>[x1, y1, x2, y2]<|box_end|>') # select content in the radio button at coordinate x1, y1, x2, y2
scroll(direction='') # scroll the page, direction can only be down or up
back() # return to the last visited page
wait() # sleep for 1s when you see blank or corrupted page
finish(answer='') # finish the task, give the answer if you are asked something'''

action_space_pxxt_mmmw = \
'''The actions you can take are listed below:
goto(url='') # open the target website
click(start_box='<|box_start|>[x, y]<|box_end|>') # click the element at point x, y
type(content='', start_box='<|box_start|>[x, y]<|box_end|>') # type in the input box at coordinate x, y
select(content='', start_box='<|box_start|>[x, y]<|box_end|>') # select content in the radio button at coordinate x, y
scroll(direction='') # scroll the page, direction can only be down or up
back() # return to the last visited page
wait() # sleep for 1s
finish(answer='') # finish the task, give the answer if you are asked something'''

action_space_bbox_vwa = \
'''The actions you can take are listed below:
click(start_box='<|box_start|>[x1, y1, x2, y2]<|box_end|>') # click the element in box x1, y1, x2, y2
type(content='', start_box='<|box_start|>[x1, y1, x2, y2]<|box_end|>') # type in the input box at coordinate x1, y1, x2, y2
scroll(direction='') # scroll the page, direction can only be down or up
back() # return to the last visited page
wait() # sleep for 1s when you see blank or corrupted page
finish(answer='') # finish the task, give the answer if you are asked something'''

action_space_pxxt_vwa = \
'''The actions you can take are listed below:
goto(url='') # open the target website
click(start_box='<|box_start|>[x, y]<|box_end|>') # click the element at point x, y
type(content='', start_box='<|box_start|>[x, y]<|box_end|>') # type in the input box at coordinate x, y
scroll(direction='') # scroll the page, direction can only be down or up
back() # return to the last visited page
wait() # sleep for 1s
finish(answer='') # finish the task, give the answer if you are asked something'''

prompt_bbox_250714 = \
f'''{job_desc}

{action_space_bbox_mmmw}

Your primary task:
{{}}

Your action history:
{{}}

{format_desc}'''

prompt_pxxt_250714 = \
f'''{job_desc}

{action_space_pxxt_mmmw}

Your primary task:
{{}}

Your action history:
{{}}

{format_desc}'''


job_desc_vwa = \
'You are a browser-use agent and is completing a task. You will be given the task instruction (some tasks are attached with an input image), current webpage screenshot, the actions you can take, your action history, the information in your memo, tabs of the browser and a hint of current page if neccessary.'

prompt_bbox_vwa = \
f'''{job_desc_vwa}

{action_space_bbox_vwa}

Your task:
{{}}

Your action history:
{{}}

Info in your memo:
{{}}

Tabs of the browser:
{{}}

Hint: {{}}

{format_desc_vwa}'''

prompt_pxxt_vwa = \
f'''{job_desc_vwa}

{action_space_pxxt_vwa}

Your task:
{{}}

Your action history:
{{}}

Info in your memo:
{{}}

Tabs of the browser:
{{}}

Hint: {{}}

{format_desc_vwa}'''

def pu_bbox_parse_response(response):
    pred_action_history = [] # TODO
    pred_action_description = ''
    pred_action = response
    pred_action_type = 'UNKNOWN'
    pred_bbox = [0, 0, 0, 0]
    pred_type_value = ''
    pred_click_point = [0, 0]

    error_msg = ''
    try: pred_bbox = find4(response)
    except Exception as e: error_msg += f'{response}, error at find4 {e}\n'
    DEBUG(pred_bbox)
    try: pred_click_point = [(pred_bbox[0] + pred_bbox[2]) / 2, (pred_bbox[1] + pred_bbox[3]) / 2]
    except Exception as e: error_msg += f'{response}, error at pred click point{e}\n'

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

def pu_pxxt_parse_response(response):
    pred_action_history = [] # TODO
    pred_action_description = ''
    pred_action = ''
    pred_action_type = 'UNKNOWN'
    pred_bbox = [0, 0, 0, 0]
    pred_type_value = ''
    pred_click_point = [0, 0]

    error_msg = ''
    try: pred_action = str(pred_action)
    except Exception as e: error_msg += f'{response}, error at action {e}\n'
    try: pred_click_point = find2(pred_action)
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

def parse_response_bbox_250610(response):
    pred_action_history = [] # TODO
    pred_action_description = ''
    pred_action = ''
    pred_action_type = 'UNKNOWN'
    pred_bbox = [0, 0, 0, 0]
    pred_type_value = ''
    pred_click_point = [0, 0]

    error_msg = ''
    try: response_results = u.extract_text(response, '```json', '```')[0]
    except Exception as e: error_msg += f'{response}, error at extract text {e}\n'
    try: pred_action_description = u.extract_text(response_results, '\"action_description\": \"', '\",')[0]
    except Exception as e: error_msg += f'{response}, error at action description {e}\n'
    try: pred_action = u.extract_text(response_results, '\"action\": \"', '\"\n}')[0]
    except Exception as e: error_msg += f'{response}, error at action {e}\n'
    try: pred_bbox = find4(pred_action)
    except Exception as e: error_msg += f'{response}, error at find4 {e}\n'
    try: pred_click_point = [(pred_bbox[0] + pred_bbox[2]) / 2, (pred_bbox[1] + pred_bbox[3]) / 2]
    except Exception as e: error_msg += f'{response}, error at pred click point{e}\n'

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

def parse_response_pxxt_250610(response):
    pred_action_history = [] # TODO
    pred_action_description = ''
    pred_action = ''
    pred_action_type = 'UNKNOWN'
    pred_bbox = [0, 0, 0, 0]
    pred_type_value = ''
    pred_click_point = [0, 0]

    error_msg = ''
    try: response_results = u.extract_text(response, '```json', '```')[0]
    except Exception as e: error_msg += f'{response}, error at extract text {e}\n'
    try: pred_action_description = u.extract_text(response_results, '\"action_description\": \"', '\",')[0]
    except Exception as e: error_msg += f'{response}, error at action description {e}\n'
    try: pred_action = u.extract_text(response_results, '\"action\": \"', '\"\n}')[0]
    except Exception as e: error_msg += f'{response}, error at action {e}\n'
    try: pred_click_point = find2(pred_action)
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

def parse_response_bbox_250714(response):
    pred_action_history = [] # TODO
    pred_action_description = ''
    pred_action = ''
    pred_action_type = 'UNKNOWN'
    pred_bbox = [0, 0, 0, 0]
    pred_type_value = ''
    pred_click_point = [0, 0]

    error_msg = ''
    try: response_results = u.extract_text(response, '```json', '```')[0]
    except Exception as e: error_msg += f'{response}, error at extract text {e}\n'
    try: pred_action_description = u.extract_text(response_results, '\"action_description\": \"', '\",')[0]
    except Exception as e: error_msg += f'{response}, error at action description {e}\n'
    try: pred_action = u.extract_text(response_results, '\"action\": \"', '\"\n}')[0]
    except Exception as e: error_msg += f'{response}, error at action {e}\n'

    pred_action_type, pred_type_value = parse_type(pred_action)
    if pred_action_type == 'CLICK' or pred_action_type == 'SELECT' or pred_action_type == 'TYPE':
        try: pred_bbox = find4(pred_action)
        except Exception as e: error_msg += f'{response}, error at find4 {e}\n'
        try: pred_click_point = [(pred_bbox[0] + pred_bbox[2]) / 2, (pred_bbox[1] + pred_bbox[3]) / 2]
        except Exception as e: error_msg += f'{response}, error at pred click point{e}\n'

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

def parse_response_pxxt_250714(response):
    pred_action_history = [] # TODO
    pred_action_description = ''
    pred_action = ''
    pred_action_type = 'UNKNOWN'
    pred_bbox = [0, 0, 0, 0]
    pred_type_value = ''
    pred_click_point = [0, 0]

    error_msg = ''
    try: response_results = u.extract_text(response, '```json', '```')[0]
    except Exception as e: error_msg += f'{response}, error at extract text {e}\n'
    try: pred_action_description = u.extract_text(response_results, '\"action_description\": \"', '\",')[0]
    except Exception as e: error_msg += f'{response}, error at action description {e}\n'
    try: pred_action = u.extract_text(response_results, '\"action\": \"', '\"\n}')[0]
    except Exception as e: error_msg += f'{response}, error at action {e}\n'
    try: pred_click_point = find2(pred_action)
    except Exception as e: error_msg += f'{response}, error at find2 {e}\n'

    pred_action_type, pred_type_value = parse_type(pred_action)
        
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

def parse_response_vwa(response, mode = 'bbox'):
    pred_action_history = [] # TODO
    pred_action_description = ''
    pred_action = ''
    pred_action_type = 'UNKNOWN'
    pred_bbox = [0, 0, 0, 0]
    pred_type_value = ''
    pred_click_point = [0, 0]
    content_to_memo = ''

    error_msg = ''
    try: response_results = u.extract_text(response, '```json', '```')[0]
    except Exception as e: error_msg += f'{response}, error at extract text {e}\n'
    try: pred_action_description = u.extract_text(response_results, '\"action_description\": \"', '\",')[0]
    except Exception as e: error_msg += f'{response}, error at action description {e}\n'
    try: pred_action = u.extract_text(response_results, '\"action\": \"', '\",')[0]
    except Exception as e: error_msg += f'{response}, error at action {e}\n'
    try: content_to_memo = u.extract_text(response_results, '\"content_to_memo\": \"', '\"\n}')[0]
    except Exception as e: error_msg += f'{response}, error at content_to_memo {e}\n'

    pred_action_type, pred_type_value = parse_type(pred_action)
    if pred_action_type == 'CLICK' or pred_action_type == 'TYPE':
        if mode == 'bbox':
            try: pred_bbox = find4(pred_action)
            except Exception as e: error_msg += f'{response}, error at find4 {e}\n'
            try: pred_click_point = [(pred_bbox[0] + pred_bbox[2]) / 2, (pred_bbox[1] + pred_bbox[3]) / 2]
            except Exception as e: error_msg += f'{response}, error at pred click point{e}\n'
        elif mode == 'pxxt':
            try: pred_click_point = find2(pred_action)
            except Exception as e: error_msg += f'{response}, error at find2 {e}\n'

    res = {
        'pred_action_history': pred_action_history,
        'pred_action_description': pred_action_description,
        'pred_action': pred_action,
        'pred_action_type': pred_action_type,
        'pred_bbox': pred_bbox,
        'pred_type_value': pred_type_value,
        'pred_click_point': pred_click_point,
        'parse_error_msg': error_msg,
        'content_to_memo': content_to_memo,
    }
    return res

if __name__ == "__main__":
    a = prompt_bbox_250714.format('123', '456')
    a = ori_bbox_prompt_f.format('123', '456')
    DEBUG(a)