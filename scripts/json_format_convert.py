import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description='Convert between JSON and JSON Lines formats.')
    parser.add_argument('--input_file', '-i', type=str, help='Path to the input file.')
    parser.add_argument('--output_file', '-o', type=str, help='Path to the output file.')
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} does not exist.")
        return

    input_ext = os.path.splitext(args.input_file)[1]
    if input_ext not in ['.json', '.jsonl']:
        print(f"Error: Input file {args.input_file} must be a .json or .jsonl file.")
        return

    output_ext = os.path.splitext(args.output_file)[1]
    if input_ext not in ['.json', '.jsonl']:
        print(f"Error: output file {args.output_file} must be a .json or .jsonl file.")
        return

    with open(args.input_file) as fin:
        if input_ext == '.json':
            data = json.load(fin)
        else:
            data = []
            for line in fin:
                data.append(json.loads(line))

    with open(args.output_file, 'w') as fout:
        if output_ext == '.json':
            json.dump(data, fout, ensure_ascii=False, indent=2)
        else:
            for item in data:
                print(json.dumps(item, ensure_ascii=False), file=fout)


if __name__ == "__main__":
    main()
