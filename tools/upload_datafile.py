import argparse
import os
import os.path
import shutil

from google.colab import files

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, help="task to be trained or evaluated")
    args = parser.parse_args()
    task = args.task

    basepath = os.getcwd()
    uploaded = files.upload()
    upload_path = f"./data/raw/{task}"

    os.makedirs(upload_path, exist_ok=True)
    for filename in uploaded.keys():
        shutil.move(os.path.join(basepath, filename), 
                    os.path.join(upload_path, filename))
