import os
import sys
from datetime import datetime

from stkllm.paths import ROOT

class _Tee(object):
    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, message):
        self.stream.write(message)
        self.log_file.write(message)

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def isatty(self):
        return getattr(self.stream, "isatty", lambda: False)()

def open_log(name):
    folder = os.path.join(ROOT, "logs")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "%s_%s.txt" % (name, datetime.now().strftime("%Y%m%d_%H%M%S")))
    handle = open(path, "w", encoding="utf-8", buffering=1)
    stdout, stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(stdout, handle)
    sys.stderr = _Tee(stderr, handle)

    def restore():
        sys.stdout = stdout
        sys.stderr = stderr
        handle.close()

    return path, restore
