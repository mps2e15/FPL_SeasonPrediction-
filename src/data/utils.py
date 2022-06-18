
import os

def build_output_dirs(dirs):
    "Function to build output directories for saved data"
    for path in dirs:
        if not os.path.exists(path):
            os.makedirs(path)