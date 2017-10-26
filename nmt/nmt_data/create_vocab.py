g=open("vocab.tgt","w")
strs=set()

f=open("train.tgt","r")
for line in f:
	words = line.split(" ")
	for word in words:
		if word not in strs:
			strs.add(word)
			g.write(word+"\n")
f.close()

f=open("dev.tgt","r")
for line in f:
	words = line.split(" ")
	for word in words:
		if word not in strs:
			strs.add(word)
			g.write(word+"\n")
f.close()

f=open("test.tgt","r")
for line in f:
	words = line.split(" ")
	for word in words:
		if word not in strs:
			strs.add(word)
			g.write(word+"\n")
f.close()

g.close()