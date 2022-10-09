import sys
import os

try:
    run_config = sys.argv[1]
except IndexError:
    run_config = None


os.system(f"python3 {run_config}")




