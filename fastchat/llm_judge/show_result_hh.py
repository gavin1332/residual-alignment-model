"""
Usage:
python3 show_result.py --mode [single|pairwise-baseline|pairwise-all]
"""
import argparse
import pandas as pd
import re

pd.set_option('display.max_rows',None)
def display_result_single(args):
    if args.input_file is None:
        # input_file = (f"data/{args.bench_name}/model_judgment/{args.judge_model}_{args.dataset}_single.jsonl")
        input_file = (f"data/{args.bench_name}/model_judgment/{args.judge_model}_{args.dataset}_single.jsonl")
        print(input_file)
    else:
        input_file = args.input_file

    print(f"Input file: {input_file}")
    df_all = pd.read_json(input_file, lines=True)
    df = df_all[["model", "score", "turn"]]
    df = df[df["score"] != -1]

    if args.model_list is not None:
        df = df[df["model"].isin(args.model_list)]
    # df = df.groupby('model', sort=False).tail(160)
    print(len(df["model"]))
    for i in range(len(df["model"])):
        # print(type(df.loc[1] ))
        try:
            if df["model"][i]=="pythia-12b-28-sft-our-ins-2e8-005":
                pass
                # print(df["model"][i])
                # print(i)
        except:
            continue
    """
    pythia-12b-28-sft-our-ins-2e8-005       2.371875
pythia12b_sft_norm-ins-2e8              2.334375
    """
    print("\n########## First turn ##########")
    df_1 = df[df["turn"] == 1].groupby(["model", "turn"]).mean()
    print(df_1.sort_values(by="score", ascending=False))

    if args.bench_name == "mt_bench":
        print("\n########## Second turn ##########")
        df_2 = df[df["turn"] == 2].groupby(["model", "turn"]).mean()
        print(df_2.sort_values(by="score", ascending=False))

        print("\n########## Average ##########")
        df_3 = df[["model", "score"]].groupby(["model"]).mean()
        print(df_3.sort_values(by="score", ascending=False))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-name", type=str, default="mt_bench")
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--judge-model", type=str, default="gpt-4")
    parser.add_argument("--dataset", type=str, default="hh")
    parser.add_argument("--baseline-model", type=str, default="gpt-3.5-turbo")
    parser.add_argument(
        "--model-list",
        type=str,
        nargs="+",
        default=None,
        help="A list of models to be evaluated",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["pairwise-baseline", "pairwise-all", "single"],
        help=(
            "Evaluation mode. "
            "`pairwise-baseline` runs pairwise comparision against a baseline. "
            "`pairwise-all` runs pairwise comparision between all pairs. "
            "`single` runs single answer grading."
        ),
    )
    args = parser.parse_args()

    if args.mode == "single":
        display_result_func = display_result_single
    else:
        if args.mode == "pairwise-all":
            args.baseline_model = None
        display_result_func = display_result_pairwise

    print(f"Mode: {args.mode}")
    display_result_func(args)
