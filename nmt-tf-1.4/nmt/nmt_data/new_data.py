g=open("train.tgt","r")
strs=set()

for line in g:
	words = line.split(" ")
	for word in words:
		if word not in strs:
			strs.add(word)

newset=set()

f=open("test.tgt","r")
for line in f:
	words = line.split(" ")
	for word in words:
		if word not in strs:
			newset.add(word)
f.close()

f=open("dev.tgt","r")
for line in f:
	words = line.split(" ")
	for word in words:
		if word not in strs:
			newset.add(word)
f.close()

counter=0;
g.close()
g=open("temp.tgt","w+")

for word in newset:
	if(counter<6):
		g.write(word+" ")
		counter=counter+1
	if(counter==6):
		g.write("\n")
		counter=0

g.close()