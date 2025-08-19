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
    folder_path = 'antsys-aworldspace-prod/ml001/browser_agent/traces/0721_infer_browsecomp_som_qw72b/'
    output_path = f'{u.get_nas()}/gui_dataset/browsecomp/zhuige/'
    u.mkdir(output_path)
    home = u.get_home()
    exe = home + '/ossutilmac64'
    command = f"{exe} cp oss://{folder_path} {output_path} -r --config-file {u.get_home()+'/.ossutilconfig_zhuige'}"
    u.execute(command)