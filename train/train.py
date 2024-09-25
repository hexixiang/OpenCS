import re
import yaml
import subprocess
from ruamel.yaml import YAML
from datetime import datetime

def float_representer(dumper, value):
    text = f'{value:.16g}'
    if "." not in text:
        text = re.sub(r'(\d+)e', r'\1.0e', text)
    return dumper.represent_scalar('tag:yaml.org,2002:float', text)

yaml = YAML()
yaml.representer.add_representer(float, float_representer)

now = datetime.now().strftime("%m%d")
# Important: run in Docker! path: /app
# Define the parameters to be modified
bash_file = "bash/llama2-7b-sft.sh"
model_name = "llama2-7b"
datasets_configs = [
    {"dataset": "alpaca_tips_wo_tag_cluster_1k_sharegpt", "config_file": "examples/train_full/llama2-7b_full_sft_ds3.yaml", "learning_rate": 1e-5, "num_train_epochs": 15, "per_device_train_batch_size": 2}
]

cuda_visible_devices = "0,1,2,3,4,5,6,7"

for data_config in datasets_configs:
    dataset = data_config["dataset"]
    config_file = data_config["config_file"]
    learning_rate = data_config["learning_rate"]
    num_train_epochs = data_config["num_train_epochs"]
    per_device_train_batch_size = data_config["per_device_train_batch_size"]
    output_dir = f"/model/output/{model_name}-{dataset}-e{num_train_epochs}lr{learning_rate}"
    run_name = f"{model_name}-{now}-{dataset}-e{num_train_epochs}lr{learning_rate}"

    # Read the YAML file
    with open(config_file, 'r') as file:
        config = yaml.load(file)

    # Modify parameters
    config['dataset'] = dataset
    config['num_train_epochs'] = num_train_epochs
    config['output_dir'] = output_dir
    config['run_name'] = run_name
    config['per_device_train_batch_size'] = per_device_train_batch_size
    config['learning_rate'] = learning_rate

    # Write back to the YAML file
    with open(config_file, 'w') as file:
        yaml.dump(config, file)

    # Write back to the bash script
    with open(bash_file, 'r') as file:
        bash_content = file.read()

    bash_content = bash_content.replace("FORCE_TORCHRUN=1 ", "")

    bash_content = re.sub(
        r'CUDA_VISIBLE_DEVICES=[^\s]+', f'CUDA_VISIBLE_DEVICES={cuda_visible_devices}', bash_content
    )

    bash_content = re.sub(
        r'llamafactory-cli train .*', f'llamafactory-cli train {config_file}', bash_content
    )

    if len(cuda_visible_devices) == 1:
        bash_content = bash_content.replace("CUDA_VISIBLE_DEVICES", "FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES")

    with open(bash_file, 'w') as file:
        file.write(bash_content)

    # Execute the bash script
    subprocess.run([bash_file])
