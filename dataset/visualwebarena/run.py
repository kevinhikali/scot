import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
up_dir = parent_dir
for i in range(3):
    sys.path.append(up_dir)
    up_dir = os.path.dirname(up_dir)
from kutils import DEBUG, INFO, WARN, ERROR
import utils as u
import argparse
import copy
import multiprocessing as mp
from agent.prompts import *
from llm_service.ais_requestor import check_providers
from vwa_tester import VWATester, VWAPathHandler, VWAConfig

def test(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args.action_set_tag = "som" # action type

    if args.mode == 'vision':
        args.instruction_path = f'{args.vwa_code_path}/agent/prompts/jsons/vision.json'
        args.observation_type = "image"
    elif args.mode == 'som':
        args.instruction_path = f'{args.vwa_code_path}/agent/prompts/jsons/som.json'
        args.observation_type = "image_som"
    elif args.mode == 'mas':
        args.instruction_path = f'{args.vwa_code_path}/agent/prompts/jsons/mas.json'
        args.observation_type = "image_som"

    tester = VWATester(args)
    tester.test()

if __name__ == "__main__":
    args = VWAConfig()

    # u.execute(f'bash {u.get_git()}/dataset/visualwebarena/docker.sh')

    args.vwa_code_path = f'{u.get_git()}/dataset/visualwebarena/'
    args.vwa_data_path = f'{u.get_nas()}/gui_dataset/visualwebarena/'
    args.oss_bucket = 'antsys-aworldspace-prod'
    args.oss_rel_path = '/ml001/browser_agent/vwa/'
    caption_models = [
        # 'KevinBlip',
        'KevinBlip2',
        'KevinBlip3',
        'KevinBlip4',
        # 'KevinBlip',
        'KevinBlip2',
        'KevinBlip3',
        'KevinBlip4',
    ]

    providers = [
        # 'KevinQwen', 
        # 'KevinQwen2', 
        # 'KevinQwen3', 
        # 'KevinQwen4', 
        'zg-qw72b-h1', 
        # 'zg-qw72b-h2', 
        # 'zg-qw72b-h3', 
        # 'zg-qw72b-h4'
    ]
    multi_process = True
    args.model = 'qwen25vl72b'
    args.mode = 'mas' # som, vision, mas
    domains = ['reddit']
    args.enable_oss = False
    args.render = False

    # for debug
    providers = [ 'KevinQwen2' ]
    multi_process = False
    # args.model = 'gpt4o'
    args.model = 'qwen25vl72b'
    args.mode = 'mas'
    domains = ['reddit']
    args.enable_oss = False
    args.print_time = True
    args.output_response = True
    args.render = True
    args.test_start_idx = 0
    args.test_end_idx = 0

    if not check_providers(providers): exit()
    if not check_providers(caption_models): exit()
    n_chunks = len(providers)
    for domain in domains:
        args.domain = domain
        ph = VWAPathHandler(args)
        metrics_files = u.list_files(ph.metrics_path)

        test_config_base_dir = f'{args.vwa_data_path}/config_files/vwa/test_{domain}'
        test_config_files = u.list_files(test_config_base_dir)
        test_config_files = sorted(test_config_files, key=lambda x: int(x.split('.')[0]))
        if not args.flush: test_config_files = [a for a in test_config_files if a not in metrics_files]
        test_config_files = [test_config_base_dir + '/' + a for a in test_config_files]

        params = []
        for i_chunk in range(n_chunks):
            n_all = len(test_config_files)
            n_sep = int(n_all / n_chunks)
            if i_chunk == (n_chunks - 1):
                chunk_files = test_config_files[i_chunk * n_sep:]
            else:
                chunk_files = test_config_files[i_chunk * n_sep: (i_chunk + 1) * n_sep]

            args.test_config_files = chunk_files
            args.provider = providers[i_chunk]
            args.caption_model = caption_models[i_chunk]

            chunk_files_int = [int(u.get_name(a).replace(".json", "")) for a in chunk_files]
            INFO(f'{i_chunk} {args.provider} {args.caption_model} {chunk_files_int}')

            chunk_args = copy.deepcopy(args)
            params.append(chunk_args)
            if multi_process: continue
            test(chunk_args)

        if not multi_process: exit()
        with mp.Pool(processes = len(params)) as pool:
            results = pool.map(test, params)
