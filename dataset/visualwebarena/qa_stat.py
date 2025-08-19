import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
up_dir = parent_dir
for i in range(3):
    sys.path.append(up_dir)
    up_dir = os.path.dirname(up_dir)
from kutils import DEBUG, INFO, WARN, ERROR
import utils as u

def count_elements_with_indices(data):
    result = {}
    
    for idx, sublist in enumerate(data):
        for element in sublist:
            if element not in result.keys():
                result[element] = {'count': 0, 'indices': []}
            result[element]['count'] += 1
            if idx not in result[element]['indices']:
                result[element]['indices'].append(idx)
    
    return result

if __name__ == "__main__":
    data_path = f'{u.get_git()}/dataset/visualwebarena/config_files/vwa/'
    domains = ['reddit', 'classifieds', 'shopping']
    eval_types_stat = {}
    eval_types_stat_file = f'{data_path}/eval_types_stat.json'
    for domain in domains:
        intent_templates = []
        intents = []
        all_eval_types = []

        intent_template_file = f'{data_path}/intent_template_{domain}.json'
        intent_file = f'{data_path}/intent_{domain}.json'
        all_eval_types_file = f'{data_path}/eval_types_{domain}.json'

        config_path = f'{data_path}/test_{domain}/'
        config_files = u.list_files(config_path)
        config_files = sorted(config_files, key=lambda x: int(x.split('.')[0]))
        for config_file in config_files:
            config = u.read_json(f'{config_path}/{config_file}')
            task_id = config['task_id']

            intent_template = config['intent_template']
            intent = config['intent']
            eval_types = config['eval']['eval_types']

            intent_templates.append(intent_template)
            intents.append(intent)
            all_eval_types.append(eval_types)
        eval_types_stat[domain] = count_elements_with_indices(all_eval_types)

        u.write_json(intent_template_file, intent_templates)
        u.write_json(intent_file, intents)
        u.write_json(all_eval_types_file, all_eval_types)

    u.write_json(eval_types_stat_file, eval_types_stat)
