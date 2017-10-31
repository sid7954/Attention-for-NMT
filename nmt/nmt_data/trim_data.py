f=open("test.tgt","r")
g=open("small_test.tgt","w+")

for line in f:
	words = line.split(" ")
	if (len(words) <= 8 ):
		g.write(line)
	else:
		for i in range(8):
			g.write(words[i]+" ")
		g.write("\n")
	