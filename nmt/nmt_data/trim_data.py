import codecs

f=codecs.open("big_dev.tgt","r","utf-8")
g=codecs.open("dev.tgt","w+","utf-8")

for line in f:
	words = line.split(" ")
	if (len(words) <= 8 ):
		g.write(line)
	else:
		for i in range(8):
			g.write(words[i]+" ")
		g.write("\n")
	