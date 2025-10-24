#!/bin/bash

set -e

sudo apt update && sudo apt install -y build-essential cmake git curl wget libcurl4-openssl-dev

git clone https://github.com/ggml-org/llama.cpp.git && cd llama.cpp
cmake -B build && cmake --build build --config Release -t llama-server
