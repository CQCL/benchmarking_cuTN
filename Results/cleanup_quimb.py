
file = "quimb_chi_300.dat"

with open(file, "r") as fin:
  with open("succ_"+file, "w") as fout:
    with open("failed_"+file, "w") as ffail:
      lines = fin.readlines()

      for l in lines:
        if len(l.split()) > 1 and l.split()[1] != "nan":
          fout.write(l)
        else:
          ffail.write(l)
