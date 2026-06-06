import os
import sys

try:
    import transformers
    import torch
    print("TRANSFORMERS_AVAILABLE")
except ImportError:
    print("TRANSFORMERS_MISSING")
