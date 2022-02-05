import matplotlib.pyplot as plt

g1_data = []

f = open(input("> ") + ".axh")
f = f.read()
f = f.split("Epoch: ")
f = f[2:]
for i in range(len(f)):
	f[i] = f[i].split("\n")
	f[i] = f[i][1:-1]
	tot = 0
	num = 0
	avg = 0
	print(len(f[i]))
	for j in range(len(f[i])):
		tot = tot + int(f[i][j])
		num = num + 1
	avg = tot/num
	g1_data = g1_data + [avg]
print(g1_data)
plt.plot(range(len(g1_data)), g1_data)
