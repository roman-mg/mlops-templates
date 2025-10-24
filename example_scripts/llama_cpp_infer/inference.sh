#!/bin/bash

HF_MODEL="Qwen/Qwen3-0.6B-GGUF:Q8_0"
REASONING_FORMAT="deepseek"
MAX_NEW_LENGTH=99
SPLIT_MODE="row"
TEMP=0.6
TOP_K=20
TOP_P=0.95
MIN_P=0
CONTEXT_SIZE=40960
MAX_TOKENS=32768
HOST=0.0.0.0
PORT=8081

./llama.cpp/build/bin/llama-server -hf $HF_MODEL --jinja --reasoning-format $REASONING_FORMAT -ngl $MAX_NEW_LENGTH -fa -sm $SPLIT_MODE --temp $TEMP --top-k $TOP_K --top-p $TOP_P --min-p $MIN_P -c $CONTEXT_SIZE -n $MAX_TOKENS --no-context-shift --host $HOST --port $PORT
