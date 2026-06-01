#!/bin/bash
set -euxo pipefail

# Upgrade pip
pip install --upgrade pip

pip install "numpy<2.0.0"

pip install tensorflow==2.16.1


pip install elephas==7.2.0
