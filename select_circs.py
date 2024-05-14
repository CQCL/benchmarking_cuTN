import sys
import random
from pathlib import Path
import shutil

n_files = int(sys.argv[1])
if n_files > 480:
  raise ValueError("There are not that many files!")

all_dir = Path("tket_circuits")
new_dir = Path("selected_circs")

new_dir.mkdir(exist_ok=True)

selected_circ_files = random.sample(list(all_dir.iterdir()), n_files)

for file_path in selected_circ_files:
  shutil.copy(file_path, new_dir)

