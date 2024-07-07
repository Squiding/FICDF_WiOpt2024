import sys
from datetime import datetime
import pytz
import os
import logging
import torch

# Set environment variables to suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set logging level to DEBUG
logging.basicConfig(level=logging.DEBUG)

# Check CUDA and PyTorch setup
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("CUDA devices:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")

# Set the timezone to US Eastern Standard Time
timezone = pytz.timezone("US/Eastern")
current_time = datetime.now(timezone).strftime("%Y%m%d_%H%M%S")

# v1 is the draft one, can only run one task, and will get stuck in the beginning of task 2
# v1.2 fixed the incremental part but haven't added iCaRL
# v1.3 added timer
# v1.4 local epoch = 2
# v1.5 non-disjoint

# v2.1 PID 2202021
# v2.2 PID 2210999 fixed IndexError: list index out of range
# v2.3 PID         testing the dataloader

# v1.2.1 iid all clients same classes
# v1.2.2 iid some clients same classes
# v1.2.3 one client

# v3.3 add exemplar function
# v3.4 add exemplars to new task
# v3 final version

# v4.1 w/o noise 2361332 cuda:0

# v5.1 PID 2361555 cuda:1

# cuda:0 2024
# v5.5.1 20 exemplars
# v5.5.2 10 exemplars
# v5.5.3  5 exemplars
# v5.5.4  0 exemplars

# cuda:1 2025
# v5.6.1 20 exemplars
# v5.6.2 10 exemplars
# v5.6.3  5 exemplars
# v5.6.4  0 exemplars

# v5.7.1 20 exemplars tasks 5

# cuda:2 2025, g_round 30, l_round 20
# v5.8.1 20 exemplars

# cuda:1 2024, g_round 30, l_round 20
# v5.9.1 20 exemplars

# cuda:0
# v6.1 PID 1472241; 20 exemplars; 2024; 30 5; [4, 8, 12, 16, 20, 24, 27]
# v6.2 PID 1472433

# v6.2 PID 1472552 2025

#v10.1.2 PID 1472889
#v10.1.1 PID 1472999


# 6.3.1 7tasks(27) 2024 PID 1326243 k-mean, distillation
# 6.3.1.2 7tasks 2024 PID 1326243 k-mean, distillation
# 6.3.2 5tasks 2024 PID 4095709 k-mean, distillation
# 6.3.3 6tasks 2024 PID 4095709 k-mean, distillation

# 6.4.1 fast dynamic round tasks = [6, 12, 28, 24, 30] 2024  PID 2182132

# v7.1 dynamic training data 50*20centroids TESTING VERSION

# v7.1.2 real [6, 12, 24, 28, 30]

# v8.1.1 [6, 12, 24, 28, 30] SqueezeNet TESTING PID 215945
# v8.1.2 [6, 12, 24, 28, 30] SqueezeNet PID 1354219

# 6.10.1

# v7.2.1 300 PID 2918172
# v7.2.2 300 PID 1100279

# 7.9.1

# version = 'v1.2_3c6t2024'
# version = 'v2.3_3c6t3024'  # PID
# version = 'v3.3_3c4t2024'  # PID
# version = 'v4.3_4c6t2024'  # PID

# version = 'v1.9.2_3c6t2024'  # PID 3599668
# version = 'v2.9.2_3c6t3024'  # PID
# version = 'v3.9.2_3c4t2024'  # PID
# version = 'v4.9.2_4c6t2024'  # PID

# version = 'v1.10.1_3c6t2024'  # PID
# version = 'v2.10.1_3c6t3024'  # PID
# version = 'v3.10.1_3c4t2024'  # PID
# version = 'v4.10.1_4c6t2024'  # PID

# version = 'v1.9.1.1_3c6t2024'  # PID 3601879

# version = 'v1.9.1.2_3c6t2024'  # PID 3602565
# version = 'v2.9.1.2_3c6t3024'  # PID 3602656
# version = 'v3.9.1.2_3c4t2024'  # PID 3602747
version = 'v4.9.1.2_4c6t2024'  # PID 3602935

# Define file paths.2
script_path = "/home/ding393/projects/WiOpt2024/FFcil/FFcil_v9.1.py"
output_dir = "/home/ding393/projects/WiOpt2024/FFcil/final_output/terminal_output/"
output_file = f"final_FFcil_{version}_{current_time}.txt"
nohup_file = "nohup.out"

# Clean or truncate the nohup.out file
if os.path.exists(nohup_file):
    with open(nohup_file, 'w') as file:
        pass  # Just open and close the file to truncate it

# Construct the command with CUDA paths and environment variables
command = (
    f"nohup sh -c '"
    f"export PATH=/usr/local/cuda-11.8/bin${{PATH:+:${{PATH}}}}; "
    f"export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${{LD_LIBRARY_PATH:+:${{LD_LIBRARY_PATH}}}}; "
    f"source ~/.bashrc; "  # Or source ~/.zshrc if you are using zsh
    f"CUDA_LAUNCH_BLOCKING=1 {sys.executable} {script_path} > {output_dir}{output_file}' &"
)

print(f"Executing command: {command}")

# Execute the command
os.system(command)
