def preProc(fName):
	fO = open(fName, "r")
	f = fO.read()
	f = f.split("Epoch: ")
	f = f[1:20]
	for i in range(len(f)): # need to change to be unsed for more than 1 sim per epoch
		print("f[i]: ")
		f[i] = f[i].split("score: ")
		f[i] = f[i][1]
		f[i] = f[i].split("\n")
		f[i] = f[i][0]
		print(f[i])

preProc("regular_j.axh")
