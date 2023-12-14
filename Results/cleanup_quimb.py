
file = "quimb_chi_300.dat"

with open(file, "r") as fin:
  with open("clean_"+file, "w") as fout:
    with open("timeout_"+file, "w") as ffail:
      lines = fin.readlines()

      for l in lines:
        if len(l.split()) > 1:
          fout.write(l)
        elif len(l.split()) > 0:
          ffail.write(l)
