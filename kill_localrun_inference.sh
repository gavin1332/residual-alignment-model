#!/bin/bash

ps -ef | grep inference.py | grep -v grep

read -r -p "Kill all processes shown above? [Y/n] " input
case $input in
    [yY][eE][sS]|[yY]) echo "Got it" ;;
    [nN][oO]|[nN]) exit ;;
    *) echo "Invalid input \"$input\""; exit 1 ;;
esac

ps -ef | grep inference.py | grep -v grep | awk '{print $2}' | xargs -i -n 1 kill -9 {}
