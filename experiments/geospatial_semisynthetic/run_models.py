import os
import sys
sys.path.insert(0, '.')
from subprocess import call

JOB_ID = int(os.environ['SLURM_ARRAY_TASK_ID'])
offset = 0

job_dict = {
    1+offset: 'linear_without_U.py', 
    2+offset: 'f_cnn_g_mlp_without_U.py', 
    3+offset: 'f_cnn_g_dk_mlp_without_U.py', 
    4+offset: 'f_gcn_g_mlp_without_U.py', 
    5+offset: 'f_gcn_g_dk_mlp_without_U.py', 
    6+offset: 'f_unet_g_mlp_without_U.py', 
    7+offset: 'f_unet_g_dk_mlp_without_U.py', 
    8+offset: 'linear_with_U.py', 
    9+offset: 'f_cnn_g_mlp_with_U.py', 
    10+offset: 'f_cnn_g_dk_mlp_with_U.py', 
    11+offset: 'f_gcn_g_mlp_with_U.py', 
    12+offset: 'f_gcn_g_dk_mlp_with_U.py', 
    13+offset: 'f_unet_g_mlp_with_U.py', 
    14+offset: 'f_unet_g_dk_mlp_with_U.py'
}

if JOB_ID not in job_dict:
    print(f"{JOB_ID} not here!")
    quit()

call(["python", job_dict[JOB_ID]])
