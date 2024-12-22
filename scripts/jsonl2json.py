import json
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', '-i', type=str, required=True)
parser.add_argument('--output_file', '-o', type=str, required=True)
parser.add_argument('--reverse', '-r', action='store_true', help='json2jsonl')
args = parser.parse_args()

with open(args.output_file, 'w') as fout:
    if args.reverse:
        with open(args.input_file, 'r') as fin:
            data = json.load(fin)
        
        for elem in data:
            print(json.dumps(elem, ensure_ascii=False), file=fout)
    else:
        data = []
        with open(args.input_file, 'r') as fin:
            for line in fin:
                data.append(json.loads(line))
        
        print(json.dumps(data, indent=2, ensure_ascii=False), file=fout)

