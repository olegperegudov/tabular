
import os
from pathlib import Path

# project_dir
PROJECT_DIR = Path.cwd().parent
# project_dir/input
INPUT_DIR = os.path.join(PROJECT_DIR, 'input')
# project_dir/src
TRAINING_FILE = os.path.join(INPUT_DIR, 'mnist_train_folds.csv')
# project_dir/models
MODEL_OUTPUT = os.path.join(PROJECT_DIR, 'models')
