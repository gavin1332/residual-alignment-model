"""
Usage:
python gen_judgment.py --model-list [LIST-OF-MODEL-ID] --parallel [num-concurrent-api-call] --mode [single|pairwise-baseline|pairwise-all]
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import re
import numpy as np
from tqdm import tqdm

from fastchat.llm_judge.common_hh import (
    load_questions,
    load_model_answers,
    load_judge_prompts,
    check_data,
    play_a_match_pair,
    play_a_match_single,
    get_model_list,
    Judge,
    MatchPair,
    MatchSingle,
    NEED_REF_CATS,
)


def make_match(
    questions,
    models,
    model_answers,
    judge,
    baseline_model,
    ref_answers=None,
    multi_turn=False,
):
    matches = []
    for q in questions:
        if multi_turn and len(q["turns"]) != 2:
            continue
        for i in range(len(models)):
            q_id = q["question_id"]
            m_1 = models[i]
            m_2 = baseline_model
            if m_1 == m_2:
                continue
            a_1 = model_answers[m_1][q_id]
            a_2 = model_answers[baseline_model][q_id]
            if ref_answers is not None:
                ref = ref_answers[judge.model_name][q_id]
                match = MatchPair(
                    dict(q),
                    m_1,
                    m_2,
                    a_1,
                    a_2,
                    judge,
                    ref_answer=ref,
                    multi_turn=multi_turn,
                )
            else:
                match = MatchPair(
                    dict(q), m_1, m_2, a_1, a_2, judge, multi_turn=multi_turn
                )
            matches.append(match)
    return matches


def make_match_all_pairs(
    questions,
    models,
    model_answers,
    judge,
    baseline_model=None,
    ref_answers=None,
    multi_turn=False,
):
    matches = []
    for q in questions:
        if multi_turn and len(q["turns"]) != 2:
            continue
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                q_id = q["question_id"]
                m_1 = models[i]
                m_2 = models[j]
                a_1 = model_answers[m_1][q_id]
                a_2 = model_answers[m_2][q_id]
                if ref_answers is not None:
                    ref = ref_answers[judge.model_name][q_id]
                    match = MatchPair(
                        dict(q),
                        m_1,
                        m_2,
                        a_1,
                        a_2,
                        judge,
                        ref_answer=ref,
                        multi_turn=multi_turn,
                    )
                else:
                    match = MatchPair(
                        dict(q), m_1, m_2, a_1, a_2, judge, multi_turn=multi_turn
                    )
                matches.append(match)
    return matches


def make_match_single(
    questions,
    models,
    model_answers,
    judge,
    baseline_model=None,
    ref_answers=None,
    multi_turn=False,
):
    matches = []
    for q in questions:
        if multi_turn and len(q["turns"]) != 2:
            continue
        for i in range(len(models)):
            q_id = q["question_id"]
            m = models[i]
            a = model_answers[m][q_id]
            if ref_answers is not None:
                ref = ref_answers[judge.model_name][q_id]
                matches.append(
                    MatchSingle(
                        dict(q), m, a, judge, ref_answer=ref, multi_turn=multi_turn
                    )
                )
            else:
                matches.append(MatchSingle(dict(q), m, a, judge, multi_turn=multi_turn))
    return matches


def make_judge_pairwise(judge_model, judge_prompts):
    judges = {}
    judges["default"] = Judge(judge_model, judge_prompts["pair-v2"])
    judges["math"] = Judge(judge_model, judge_prompts["pair-math-v1"], ref_based=True)
    judges["default-mt"] = Judge(
        judge_model, judge_prompts["pair-v2-multi-turn"], multi_turn=True
    )
    judges["math-mt"] = Judge(
        judge_model,
        judge_prompts["pair-math-v1-multi-turn"],
        ref_based=True,
        multi_turn=True,
    )
    return judges


def make_judge_single(judge_model, judge_prompts):
    judges = {}
    judges["default"] = Judge(judge_model, judge_prompts["single-v1"])
    judges["math"] = Judge(judge_model, judge_prompts["single-math-v1"], ref_based=True)
    judges["default-mt"] = Judge(
        judge_model, judge_prompts["single-v1-multi-turn"], multi_turn=True
    )
    judges["math-mt"] = Judge(
        judge_model,
        judge_prompts["single-math-v1-multi-turn"],
        ref_based=True,
        multi_turn=True,
    )
    return judges


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--judge-file",
        type=str,
        default="data/judge_prompts.jsonl",
        help="The file of judge prompts.",
    )
    parser.add_argument("--judge-model", type=str, default="gpt-4")
    parser.add_argument("--baseline-model", type=str, default="gpt-3.5-turbo")
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
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="The model id to be evaluated",
    )
    parser.add_argument(
        "--parallel", type=int, default=1, help="The number of concurrent API calls."
    )
    parser.add_argument(
        "--first-n", type=int, help="A debug option. Only run the first `n` judgments."
    )
    args = parser.parse_args()

    question_file = f"/private/home/liudianqing/projects/direct-preference-optimization-main/data/hh-test.json"
    answer_dir = f"data/{args.bench_name}/model_answer"
    ref_answer_dir = f"data/{args.bench_name}/reference_answer"

    # Load questions
    questions=[]
    ref_answers=[]
    with open(question_file) as fr:
        i=0
        for line in fr:
            i+=1
            obj=json.loads(line.strip())
            lst = [_ for _ in re.split("\n\nHuman:|\n\nAssistant:", obj["prompt"]) if _]
            # print( obj["prompt"])
            #
            # print(len(lst))
            # print("==========================================")
            # prompt = obj["prompt"]
            prompt = "\n\nHuman:"+lst[0]+"\n\nAssistant:"
            questions.append(prompt)
            # ref_answers.append(obj["chosen"])
            if len(lst)==1:
                ref_answers.append(obj["chosen"])
            else:
                ref_answers.append(lst[1])

    # Load answers
    model_answers = load_model_answers(answer_dir)
    model_answers = model_answers[args.model_id]
    # ref_answers = load_model_answers(ref_answer_dir)

    # Load judge
    judge_prompts = load_judge_prompts(args.judge_file)
    summ = """Which of the following summaries does a better job of summarizing the mostimportant points in the given forum post, without including unimportant orirrelevant details? A good summary is both precise and concise.\nPost :\n{post}\nSummary A:\n{Summary_A}\nSummary B:\n{Summary_B}\nFIRST provide a one-sentence comparison of the two summaries,explaining whichyou prefer and why.SECOND,on a new line, state only "A" or "B" to indicate yourchoice.Your response should use the format :Comparison:<one-sentence comparison and expanation>Preferred:<"A"or "B">"""
    diag = """For the following query to a chatbot, which response is more helpful?\n Context:{context}\nResponse A:\n{model_res}\nResponse B:\n{gold_answer}\nFIRST provide a one-sentence comparison of the two responses and explainwhich you feel is more helpful.SECOND, on a new line, state only "A" or"B" to indicate which response is more helpful. Your response should usethe format :\nComparison:<one-sentence comparison andexplanation>More helpful:<"A" or "B">"""


    models = args.model_id

    output_file = (f"data/{args.bench_name}/model_judgment/{args.judge_model}_hh_single.jsonl")
    import random
    random.seed((1120))
    if args.mode == "single":
        judges = make_judge_single(args.judge_model, judge_prompts)
        play_a_match_func = play_a_match_single
        for i,qs in enumerate(questions):
            if i + 1 in model_answers:
                if len(re.split("\n",model_answers[i + 1]["answer"]))>10:
                    model_answers[i + 1]["answer"]="\n".join(re.split("\n",model_answers[i + 1]["answer"])[:3])
                elif len(re.split("[\.\?!]",model_answers[i + 1]["answer"]))>20:
                    model_answers[i + 1]["answer"]="\n".join(re.split("[\.\?!]",model_answers[i + 1]["answer"])[:3])

                if random.randint(1, 2) == 1:
                    answer = model_answers[i + 1]["answer"]
                    ref_answer = ref_answers[i]
                    turn = 1
                else:
                    answer = ref_answers[i]
                    ref_answer = model_answers[i + 1]["answer"]
                    turn = 2
                question=qs
                question_id=i+1
                # answer=model_answers[i+1]["answer"]
                # ref_answer=ref_answers[i]
                # turn=1
                play_a_match_func(question, question_id, args.model_id, answer, judges["default"], ref_answer, turn, output_file=output_file)

        # for i, qs in enumerate(questions):
        #     if i + 1 in model_answers:
        #         question = qs
        #         question_id = i + 1
        #         answer =ref_answers[i]
        #         ref_answer = model_answers[i + 1]["answer"]
        #         turn = 2
        #         play_a_match_func(question, question_id, args.model_id, answer, judges["default"], ref_answer, turn, output_file=output_file)

        """
            question, model, answer, judge, ref_answer, multi_turn = (
        match.question,
        match.model,
        match.answer,
        match.judge,
        match.ref_answer,
        match.multi_turn,
    )
        for match in tqdm(matches):
            play_a_match_func(match, output_file=output_file)
        python judgment_hh.py --judge-model Qwen2-72B-Instruct-GPTQ-Int4 --model_id pythia28b-dpo-our_single_turn_hh
        python show_result_hh.py --input-file=data/mt_bench/model_judgment/Qwen2-72B-Instruct-GPTQ-Int4_hh_single.jsonl
        """
        # def run_judge_single(question, answer, judge, ref_answer, multi_turn=False)
        # def play_a_match_single(match: MatchSingle, output_file: str):


