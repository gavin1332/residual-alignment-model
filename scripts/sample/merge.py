import json
import argparse
from glob import glob
import re

def merge_file(filepath, file_prefix):
    path = filepath+'/'+file_prefix+"*.json"
    print(path)
    files = glob(path)
    print(files)
    data = []
    for file in files:
        with open(file) as f:
            objs = json.load(f)
            data += objs
    merged_path = filepath+'/'+file_prefix+"_merged.json"
    with open(merged_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("\n sample results are saved to -->",merged_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, required=True)
    parser.add_argument('--file_prefix', type=str, required=True)
    args = parser.parse_args()
    merge_file(args.filepath, args.file_prefix)