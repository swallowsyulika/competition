import os
from subprocess import run, PIPE
from config import darknet_executable, eval_containers_path, yolo_obj_data, yolo_cfg, yolo_eval_weight_path, eval_yolo_json_path
from utils import ensure_file

input_dir = eval_containers_path
ensure_file(eval_yolo_json_path)

items = [os.path.join(input_dir, f)
         for f in os.listdir(input_dir) if f.endswith(".jpg")]

# darknet.exe detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights -ext_output -dont_show -out result.json < data/train.txt
cwd = os.path.dirname(darknet_executable)
print('\n'.join(items))
run([darknet_executable,
     "detector",
     "test",
     yolo_obj_data,
     yolo_cfg,
     yolo_eval_weight_path,
     "-ext_output",
     "-dont_show",
     "-out",
     eval_yolo_json_path
     ], cwd=cwd, input='\n'.join(items), encoding='ascii')
