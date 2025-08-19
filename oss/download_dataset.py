import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
up_dir = parent_dir
for i in range(3):
    sys.path.append(up_dir)
    up_dir = os.path.dirname(up_dir)
from kutils import DEBUG, INFO, WARN, ERROR
import utils as u
import oss_utils as ou

if __name__ == "__main__":
    home = u.get_home()
    exe = home + '/ossutil64'
    dataset_name = sys.argv[1:][0]

    bucket = f'antsys-ssdata-dataset/antcia/huggingface/{dataset_name}'
    
    # command = f"{exe} du oss://{bucket} --block-size GB"
    # u.execute(command)

    command = f"{exe} cp oss://{bucket} /ossfs/workspace/kaiwen/gui_dataset/ -r --config-file {u.get_home()+'/.ossutilconfig'}"
    u.execute(command)