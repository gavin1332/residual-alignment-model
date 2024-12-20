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

ID=$(date +%s)
if [[ $# == 4 ]]; then
    ID=$4
fi

if [ ! -f $INPUT_FILE ]; then
    echo "$INPUT_FILE not exists."
    exit 1
fi

ls $INPUT_FILE.* &> /dev/null
if [ $? -eq 0 ]; then
    echo "$INPUT_FILE.* exists"
    exit 1
fi

total_lines=`wc -l $INPUT_FILE | awk '{print $1}'`
num_gpus=`echo $DEVICES | awk -F',' '{print NF}'`
lines_per_part=$(( (total_lines + num_gpus - 1) / num_gpus ))
echo "total_lines=$total_lines num_gpus=$num_gpus lines_per_part=$lines_per_part"

BASENAME=`basename $INPUT_FILE`
DIRNAME=`dirname $INPUT_FILE`
WORKDIR=$DIRNAME/_${BASENAME}_$num_gpus
INPUT_PREFIX=$WORKDIR/part-${ID}-

mkdir -p $WORKDIR

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

split -a 3 -d -l $lines_per_part $INPUT_FILE $INPUT_PREFIX

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
cat $INPUT_PREFIX*.out > $OUTPUT_FILE

echo finished
