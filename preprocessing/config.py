import os

class Config():
    run_dir_path = os.path.dirname(os.path.abspath(__file__))
    last = run_dir_path.split("/")[-1]
    up_dir = run_dir_path.replace(last,"")
    image_dir_path = up_dir + "aozora_data/files/tmp.txt"

