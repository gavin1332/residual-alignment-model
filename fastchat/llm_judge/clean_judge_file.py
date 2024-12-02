# encoding:utf-8
#    Author:  a101269
#    Date  :  2024/9/24
import sys
import json
import argparse

def clean(file,model_id):
    res=[]
    with open(file) as fr:
        i=0
        for line in fr:
            i+=1
            obj=json.loads(line.strip())
            if obj["model"]==model_id:
                continue
            res.append(line)
    with open(file,'w') as fw:
        for line in res:
            fw.write(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id",type=str)
    parser.add_argument("--file", type=str)
    args = parser.parse_args()
    clean(args.file,args.model_id)
