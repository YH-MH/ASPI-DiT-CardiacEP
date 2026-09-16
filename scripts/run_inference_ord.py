import os
import sys
from utils.misc import name_with_datetime

inference_path = ".\\inference.py" # 运行主文件
yaml_path = ".\\configs\\ord1_train.yaml" # 配置文件
model_path = ".\\save\\diffusion_modules\\checkpoints\\xxxx.pth"

data_phase = "test4" 
output_path = ".\\save" # 存储文件夹
num_generated = 100
batch_size = 512
name = "xxxxxx"
string = f"python {inference_path} --name {name} --config_file {yaml_path} --output {output_path}\
      --load_path {model_path} --data_phase {data_phase} --num_generated {num_generated} --batch_size {batch_size} --num_node 1 --tensorboard "

# os.system(string)

import subprocess
for p in [inference_path, yaml_path, model_path]:
    if not os.path.exists(p):
        print("❌ Not found:", p)
if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)

cmd = [
    sys.executable,              
    inference_path,
    "--name", name,
    "--config_file", yaml_path,
    "--output", output_path,
    "--load_path", model_path,
    "--data_phase", data_phase,
    "--num_generated", str(num_generated),
    "--batch_size", str(batch_size),
    "--num_node", "1",
    "--tensorboard",
]

print("Running:", " ".join(f'"{c}"' if " " in c else c for c in cmd))
subprocess.run(cmd)  


