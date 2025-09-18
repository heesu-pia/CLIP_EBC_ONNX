#!/bin/bash

LOG_DIR="output_IAR/logs_eval"
mkdir -p "$LOG_DIR"

ts="$(date +%Y%m%d_%H%M%S)"
log_file="$LOG_DIR/eval_${ts}.log"
python eval.py 2>&1 | tee -a "$log_file"