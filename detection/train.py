import os
import argparse
from subprocess import Popen
from utils.ensure_dir import ensure_dir
from config import darknet_executable, train_yolo_data_path, yolo_cfg, yolo_obj_data, yolo_train_list, yolo_valid_list, yolo_obj_names, yolo_weights_path, yolo_pretrained_weight
from utils import ensure_file

parser = argparse.ArgumentParser()
parser.add_argument("--run", action='store_true', help="Run the training process immediately.")

args = parser.parse_args()

ensure_file(yolo_train_list)
ensure_file(yolo_valid_list)
ensure_dir(yolo_weights_path)

imgs = sorted([os.path.join(train_yolo_data_path, x) for x in os.listdir(train_yolo_data_path) if x.endswith(".jpg")])
percent_valid = 0.2
num_valid = int(len(imgs) * percent_valid)

train_imgs = imgs[:-num_valid]
valid_imgs = imgs[-num_valid:]

with open(yolo_train_list, 'w', encoding='utf-8') as f:
    f.write("\n".join(train_imgs))

with open(yolo_valid_list, 'w', encoding='utf-8') as f:
    f.write("\n".join(valid_imgs))

ensure_file(yolo_obj_data)

with open(yolo_obj_data, 'w', encoding='utf-8') as f:
    f.write(f"""
classes = 1
train  = {yolo_train_list}
valid  = {yolo_valid_list}
names = {yolo_obj_names}
backup = {yolo_weights_path}
    """)
# darknet.exe detector train data/obj.data yolo-obj.cfg yolov4.conv.137
cwd = os.path.dirname(darknet_executable)
executable = "./" + os.path.basename(darknet_executable)

cmd = [
        executable,
        "detector",
        "train",
        yolo_obj_data,
        yolo_cfg,
        yolo_pretrained_weight,
    ]
if args.run:
    if "QT_QPA_PLATFORM_PLUGIN_PATH" in os.environ:
        os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
    Popen(cmd, cwd=cwd)
else:
    print("[I] Training is configuration done.")
    print("Please run the following commands to start training:")
    print(f"cd {cwd}")
    print(" ".join(cmd))
