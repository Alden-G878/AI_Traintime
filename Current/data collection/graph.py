import matplotlib.pyplot as plt

def gFFile(f):
	g_data = []
	f = f.read()
	f = f.split("Epoch: ")
	f = f[2:]
	for i in range(len(f)):
		f[i] = f[i].split("\n")
		f[i] = f[i][1:-1]
		tot = 0
		num = 0
		avg = 0
		for j in range(len(f[i])):
			tot = tot + int(f[i][j])
			num = num + 1
		avg = tot/num
		g_data = g_data + [avg]	
	plt.ylim([100,210])
	plt.plot(range(len(g_data)), g_data)

plt.figure(0)
gFFile(open("regular_j_p" + ".axh"))
plt.figure(1)
gFFile(open("mod_j_r1_p" + ".axh"))
plt.figure(2)
gFFile(open("mod_j_r2_p" + ".axh"))
plt.show()