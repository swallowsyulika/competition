import sys

from utils import levenshteinDistanceDP
import config

if config.mode != "pseudo_eval":
    print("[E] This module is only available when mode is set to 'pseudo_eval'.")
    sys.exit()

ground_csv = config.pseudo_eval_ground_csv_path
inferenced_csv = config.eval_recognition_csv_path

def get_label(line: str):
    return line.split(",")[-1].strip()

ned = 0.0
n_samples = 0
with open(ground_csv, 'r') as ground_f, open(inferenced_csv, 'r') as inf_f:
    while True:
        n_samples += 1
        ground_line = ground_f.readline()
        inf_line = inf_f.readline()

        if ground_line == "" or inf_line == "":
            print(f"n_samples: {n_samples}.")
            break

        ground_label = get_label(ground_line)
        inferenced_label = get_label(inf_line)

        if inferenced_label == "###":
            continue

        d = levenshteinDistanceDP(ground_label, inferenced_label).item()
        max_len = max(len(ground_label), len(inferenced_label))

        ned += (d / max_len)
    
    ned /= n_samples
    print(f"1NED = {1-ned}.")
    