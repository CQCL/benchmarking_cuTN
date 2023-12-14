import os

chi = 300

for data_name in [f"quimb_chi_{chi}", f"itensors_chi_{chi}", f"pytket-cutn_chi_{chi}"]:
  with open(f"{data_name}.dat", "w") as combined_file:
    for k, file in enumerate(os.listdir("tmp")):
      filename = os.fsdecode(file)
      if filename.startswith(data_name):
        with open("tmp/"+filename, "r") as f:
            lines = f.readlines()
            for l in lines:
                combined_file.write(l)
