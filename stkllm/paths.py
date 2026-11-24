import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

                     
LLAMA_MODEL_PATH = "/home/buaa/yyx/One-Fits-All-main/llama3"

def abs_under_root(path_value):
    if os.path.isabs(path_value):
        return os.path.normpath(path_value)
    return os.path.normpath(os.path.join(ROOT, path_value))
