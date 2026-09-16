import os
import sys
import subprocess
import random
from utils.misc import name_with_datetime

train_path = ".\\train.py" # 运行主文件
yaml_path = ".\\configs\\ord1_train.yaml" # 配置文件
output_path = ".\\save" # 存储文件夹
# num_generated, 默认1
num_generated = 1
name = name_with_datetime("ord_train4_tokens9128head4_cost128128apNum2")
seed = 3407
print(f"Generated random seed: {seed}")

cmd = [
    "python", train_path,
    "--name", name,
    "--config_file", yaml_path,
    "--output", output_path,
    "--num_generated", str(num_generated),
    "--num_node", "1",
    "--tensorboard",
    "--seed", str(seed)
]

# 执行命令并检查返回值
# result = subprocess.run(cmd, capture_output=True, text=True)
subprocess.run(cmd)
# print("stdout:\n", result.stdout)
# print("stderr:\n", result.stderr)