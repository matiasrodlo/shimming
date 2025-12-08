#!/bin/bash
# Run script for Savart Optimizer

cd "$(dirname "$0")"
python3 shim_coil_biot_savart.py "$@"

