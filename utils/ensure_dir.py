import os
import shutil
import sys

def clean_dir(path: str):
    for root, dirs, files in os.walk(path):
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
        for f in files:
            os.remove(os.path.join(root, f))

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[I] Automatically created directory: {path}")
    elif len(os.listdir(path)) > 0:
        print(f"{path} is not empty. Do you want to clean it? [Y/n/r(ename)] ", end='')
        ans = input()
        if ans == 'Y':
            clean_dir(path)
        elif ans == 'r':
            print("Please enter new name of the directory:", end='')
            new_dir_name = input()
            new_path = os.path.join(os.path.dirname(path), new_dir_name)
            os.rename(path, new_path)
            print(f"Renamed: {path} => {new_path}")

    
def ensure_file(path: str):
    file_dir = os.path.dirname(path)
    if os.path.isfile(path):
        print(f"[W] File {path} already exists. Do you want to replace it? [Y/n/r(ename)]", end='')
        ans = input()
        if ans == 'n':
            sys.exit()
        elif ans == 'r':
            print("Please enter new name of the file:", end='')
            new_file_name = input()
            new_path = os.path.join(os.path.dirname(path), new_file_name)
            os.rename(path, new_path)
            print(f"Renamed: {path} => {new_path}")
    elif not os.path.exists(file_dir):
        os.makedirs(file_dir)

