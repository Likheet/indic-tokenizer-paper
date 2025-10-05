from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

ANALYSIS_OUT = Path("analysis/out")
FIGURES_DIR = Path("figures")
TABLES_DIR = Path("tables")
FIGURES_DIR.mkdir(exist_ok=True, parents=True)
TABLES_DIR.mkdir(exist_ok=True, parents=True)

EFFICIENCY_FILES = {
    "English": "tokenizer-output/english-fast.csv",
    "Hindi": "tokenizer-output/hindi-fast.csv",
    "Kannada": "tokenizer-output/kannada-fast.csv",
    "Tamil": "tokenizer-output/tamil-fast.csv",
    "Hinglish": "tokenizer-output/hinglish-fast.csv",
}

TOKENIZER_LABELS = {
    "openai/tiktoken/o200k_base": "o200k_base",
    "openai/tiktoken/cl100k_base": "cl100k_base",
    "ai4bharat/IndicBERTv2-MLM-only": "IndicBERTv2",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "Llama 3.1",
}

TOKENIZER_IDS = [
    "openai/tiktoken/o200k_base",
    "openai/tiktoken/cl100k_base",
    "ai4bharat/IndicBERTv2-MLM-only",
]


