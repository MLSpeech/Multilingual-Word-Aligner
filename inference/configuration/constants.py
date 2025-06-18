
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(os.path.abspath(__file__)).parent.parent.parent
sys.path.append(PROJECT_DIR)
PROJECT_NAME = str(PROJECT_DIR.name)

INFERENCE_PART_DIR = Path(os.path.abspath(__file__)).parent.parent
sys.path.append(INFERENCE_PART_DIR)

CONFIG_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.yaml")

OUTPUT_DIR = os.path.join(INFERENCE_PART_DIR, 'results')