import torch
from os.path import join, expanduser, dirname

### GENERAL ###

# all files produced by our project will be saved here.
generated_path = expanduser("../generated")
# device used to train ANN
device = "cuda" if torch.cuda.is_available() else "cpu"

### CONVERT ###

## TRAIN ##

# original training dataset
train_path = expanduser("../train/train")


# training images with bonding boxes and labels drawn.
train_basepath = join(generated_path, "train")
train_bb_drawn_path = join(train_basepath, "bb_drawn")

# training images cropped to boxes of one or more characters (to train object detection model)
train_containers_path = join(train_basepath, "containers")
train_yolo_data_path = join(train_basepath, "yolo_train")

# training images cropped to single Chinese character (to train image cleaner)
train_characters_path = join(train_basepath, "characters")

pseudo_eval_path = join(generated_path, "pseudo_eval")
pseudo_eval_img_path = join(pseudo_eval_path, "img")
pseudo_eval_csv_path = join(pseudo_eval_path, "coords.csv")
pseudo_eval_ground_csv_path = join(pseudo_eval_path, "ground_coords.csv")



## EVAL (public & private & pseudo eval) ##
mode = "pseudo_eval"
# mode = "public"
if mode == "public":
    eval_path = expanduser("../public/public")
    eval_img_path = join(eval_path, "img_public")
    eval_csv_path = join(eval_path, "Task2_Public_String_Coordinate.csv")
    eval_basepath = join(generated_path, "public")

elif mode == "pseudo_eval":
    eval_path = pseudo_eval_path
    eval_img_path = pseudo_eval_img_path
    eval_csv_path = pseudo_eval_csv_path
    eval_basepath = eval_path

# eval images with boxes of one or more characters drawn (provided by the dataset)
eval_containers_drawn_path = join(eval_basepath, "containers_drawn")
# eval images cropped to boxes of one or more characters (provided by the dataset)
eval_containers_path = join(eval_basepath, "containers")
# yolo prediction result json
eval_yolo_json_path = join(eval_basepath, "result.json")
# bonding boxes predicted by the object detection model drawn onto "containers"
eval_yolo_bb_drawn_path = join(eval_basepath, "pred_containers")
# eval images cropped to single characters (by object detection model)
eval_yolo_characters_path = join(eval_basepath, "pred_characters")

### DETECTION (YOLO v4) ###
# path to darknet executable
darknet_executable = expanduser("~/darknet/darknet")
# path to Yolo v4 pre-trained weights
yolo_pretrained_weight = expanduser("~/darknet/yolov4.conv.137")
# all yolo related generated files will be saved here.
yolo_basepath = join(generated_path, "yolo")
# list of images used to train YOLO classifier.
yolo_train_list = join(yolo_basepath, "train.txt")
# list of images used to evaluate YOLO classifier.
yolo_valid_list = join(yolo_basepath, "valid.txt")
# train metadata; will be generated automatically by detection.train module.
yolo_obj_data = join(yolo_basepath, "obj.data")
# path where the weights of yolo v4 should be saved.
yolo_weights_path = join(yolo_basepath, "weights")
# path to weights of yolo v4 model when inferencing
yolo_eval_weight_path = join(yolo_weights_path, "character_last.weights")

### CLEANER ###
cleaner_basepath = join(generated_path, "cleaner")
cleaner_weights_path = join(cleaner_basepath, "weights")
cleaner_character_list = "characters_3000_freq.txt"
cleaner_log_path = join(cleaner_basepath, "logs")

train_cleaned_path = join(train_basepath, "cleaned")
eval_cleaned_path = join(eval_basepath, "cleaned_e90")

### RECOGNITION ###

# large-scale font library
recognition_basepath = join(generated_path, "recognition")
fonts_lib = expanduser("~/fonts")
fonts_lib_manual_check_path = join(recognition_basepath, "check")
fonts_available_glyph_cache = join(recognition_basepath, "cache.json")
recognition_dataset_cache = join(recognition_basepath, "cache_e26")
recognition_weights_path = join(recognition_basepath, "weights")
recognition_logs_path = join(recognition_basepath, "logs")
recognition_err_check_path = join(recognition_basepath, "errors")
eval_recognition_csv_path = join(eval_basepath, "result.csv")
eval_recognition_preview_path = join(eval_basepath, "recognition_result")

### DO NOT MODIFY ANYTHING BELOW ###

# project root
project_root = dirname(__file__)
project_fonts_path = join(project_root, "assets", "fonts")
project_textures_path = join(project_root, "assets", "textures")
project_character_lists_path = join(project_root, "assets", "character_lists")
font_lib_blacklist_path = join(project_root, "recognition", "blacklist.txt")
# yolo related
detection_root = join(project_root, "detection")
yolo_cfg = join(detection_root, "character.cfg")
yolo_obj_names = join(detection_root, "obj.names")