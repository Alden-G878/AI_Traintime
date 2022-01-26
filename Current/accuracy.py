import math
import io

class sales():
	def __init__(self, name, num):
		self.totalPredictions = 0
		self.totalActual = 0
		self.numJobs = 0
		self.name = name
		self.id = num
		self.filename = str(self.name) + "_" + str(self.id) + ".axh"
		try:
			self.file = open(self.filename, "x")
		except:
			self.file = open(self.filename, "r+")
			print("The file you are trying to create already exists.")
			print("I suggest running 'refreshFromLoadedFile' to comfirm that")
			print("this data is not important before overwriting")
	def addJob(self, a, b):
		self.totalPredictions = self.totalPredictions + a
		self.totalActual = self.totalActual + b
		self.numJobs = self.numJobs + 1
	def perError(self):
		print(((self.totalPredictions-self.totalActual)/self.totalPredictions)*100)
	def save(self):
		self.file.writelines([str(self.name) + " \n", str(self.id) + " \n", str(self.totalPredictions) + " \n",str(self.totalActual) + " \n",str(self.numJobs) + " \n "])
	def loadF(self, nName, nID):
		self.filename = nName + "_" + nID
		self.file = open(self.fileName, 'r')
	def refreshFromLoadedFile(self):
		self.tempFS = self.file.readLines()
		self.name = self.tempFS[0]
		self.id = self.tempFS[1]
		self.totalPredictions = self.tempFS[2]
		self.totalActual = self.tempFS[3]
		self.numJobs = self.tempFS[4]

try:
	sp1 = sales("John", 1175)
except:
	print("Another person with this same name and ID already exists.")
	print("Please make sure you have the right name and ID.")
	print()
	print("If you are trying to load a archive that already exists, ")
	print("use 'refreshFromLoadedFile' instead.")

sp1.addJob(3,5)
print(sp1.totalPredictions)
print(sp1.totalActual)
sp1.perError()