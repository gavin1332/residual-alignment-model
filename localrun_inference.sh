#!/bin/bash

set -u

if [[ $# != 3 && $# != 4 ]]; then
    echo 'Usage: $0 input_file output_file devices [id]'
    echo '    Examples: $0 input.json output.json 0,1,2,3,4,5,6,7 [id]'
    exit
fi

INPUT_FILE=$1
OUTPUT_FILE=$2
DEVICES=$3

ID=id
if [[ $# == 4 ]]; then
    ID=$4
fi

if [ ! -f $INPUT_FILE ]; then
    echo "$INPUT_FILE not exists."
    exit 1
fi

N_GPUS=`echo $DEVICES | awk -F',' '{print NF}'`

TMP_DIR=_tmp
BASENAME=`basename $INPUT_FILE`
CACHE_DIR=$TMP_DIR/${BASENAME}_${N_GPUS}
INPUT_PREFIX=$CACHE_DIR/part-${ID}-

mkdir -p $CACHE_DIR

if [[ $INPUT_FILE == *.json ]]; then
    JSONL_FILE=${TMP_DIR}/${BASENAME}.jsonl
    python scripts/json_format_convert.py -i $INPUT_FILE -o $JSONL_FILE
elif [[ $INPUT_FILE == *.jsonl ]]; then
    JSONL_FILE=$INPUT_FILE
else
    echo "Only .json or .jsonl is supported."
    exit 1
fi

total_lines=`wc -l $JSONL_FILE | awk '{print $1}'`
lines_per_part=$(( (total_lines + N_GPUS - 1) / N_GPUS ))
echo "total_lines=$total_lines N_GPUS=$N_GPUS lines_per_part=$lines_per_part"

ls $INPUT_PREFIX* &> /dev/null
if [ $? -eq 0 ]; then
    read -r -p "Remove all $INPUT_PREFIX*? [Y/n] " input
    case $input in
        [yY][eE][sS]|[yY])
            rm -f $INPUT_PREFIX*
            ;;

        [nN][oO]|[nN])
            exit
            ;;

        *)
            echo "Invalid input \"$input\""
            exit 1
            ;;
    esac
fi

set -e

split -a 3 -d -l $lines_per_part $JSONL_FILE $INPUT_PREFIX

idx=0
IFS=',' read -ra device <<< "$DEVICES"
for dev in "${device[@]}"; do
    INDEX=`printf "%03d" $idx`
	PART_INPUT_FILE=$INPUT_PREFIX$INDEX
    PART_OUTPUT_FILE=$PART_INPUT_FILE.out
	bash run_inference.sh $PART_INPUT_FILE $PART_OUTPUT_FILE $dev &
    idx=$((idx+1))
done

wait

FINAL_JSONL=$CACHE_DIR/${ID}.jsonl
cat $INPUT_PREFIX*.out > $FINAL_JSONL

if [[ $INPUT_FILE == *.json ]]; then
    python scripts/json_format_convert.py -i $FINAL_JSONL -o $OUTPUT_FILE
else
    cp $FINAL_JSONL $OUTPUT_FILE
fi

echo "finish $INPUT_FILE -> $OUTPUT_FILE"
