
file = "pytket-cutn_chi_300.dat"

with open(file, "r") as fin:
  with open("clean_"+file, "w") as fout:
      lines = fin.readlines()

      for l in lines:
        if len(l.split()) > 1 and l.split()[1] != "nan":
          fout.write(l)
        else:
          print("Warning! a circuit did fail!")
