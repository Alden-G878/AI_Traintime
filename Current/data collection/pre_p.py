def preProc(fName):
	fO = open(fName, "r")
	f = fO.read()
	f = f.split("Epoch: ")
	f = f[:-1]
	for i in range(len(f)): # need to change to be unsed for more than 1 sim per epoch
		f[i] = f[i].split("score: ")
		f[i] = f[i][1:]
		for j in range(len(f[i])):
			f[i][j] = f[i][j].split("\n")
			f[i][j] = f[i][j][0]
		print("Epoch: " + str(i))
		for j in range(len(f[i])):
			print(f[i][j])

preProc("mod_j_r2.axh")
