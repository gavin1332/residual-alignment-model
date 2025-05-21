#!/usr/bin/env python3

import os
import pandas as pd
import argparse


parser = argparse.ArgumentParser(description='Summarize leaderboard.csv data under a specified dataset')
parser.add_argument('-b', '--root', type=str, default='_output', help='The root directory')
parser.add_argument('-d', '--dataset', type=str, help='Name of the dataset to summarize', default='alpaca')
parser.add_argument('-s', '--sort', help='Sort the leaderboard by specified column', default='length_controlled_winrate')
parser.add_argument('-fo', '--filter_out', help='Filter out the generator with multiple specific strings separated by comma')
parser.add_argument('-fi', '--filter_in', help='Filter in the generator with multiple specific strings separated by comma')
args = parser.parse_args()

dataset_dir = os.path.join(args.root, args.dataset)
if not os.path.exists(dataset_dir):
    print(f"Directory for dataset {args.dataset} not found.")
    exit(-1)

all_data = []
for model in os.listdir(dataset_dir):
    model_dir = os.path.join(dataset_dir, model)
    for generator in os.listdir(model_dir):
        generator_dir = os.path.join(model_dir, generator)
        if os.path.isdir(generator_dir):
            leaderboard_path = os.path.join(generator_dir, 'leaderboard.csv')
            if os.path.exists(leaderboard_path):
                try:
                    df = pd.read_csv(leaderboard_path)
                    df.columns = ['generator'] + list(df.columns[1:])
                    all_data.append(df)
                except Exception as e:
                    print(f"Error reading {leaderboard_path}: {e}")
if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)
    if args.sort:
        if args.sort in combined_df:
            combined_df = combined_df.sort_values(by=args.sort, ascending=False)
    if args.filter_in:
        patterns = args.filter_in.split(',')
        for p in patterns:
            combined_df = combined_df[combined_df['generator'].str.contains(p, na=False)]
    if args.filter_out:
        patterns = args.filter_out.split(',')
        for p in patterns:
            combined_df = combined_df[~combined_df['generator'].str.contains(p, na=False)]
    combined_df.set_index('generator', inplace=True)
    pd.set_option('display.max_colwidth', None)
    print(combined_df[['length_controlled_winrate', 'win_rate', 'lc_standard_error', 'n_total', 'avg_length']])
else:
    print(f"No valid leaderboard.csv files found under dataset {args.dataset}.")

