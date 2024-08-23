#!/usr/bin bash
conda activate py310
python -m unittest discover -s tests -p "*.py"
