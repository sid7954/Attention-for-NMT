f=open('dev.src',"r")
g=open('dev.tgt',"w+")
for line in f:
	words=line.split(" ")
	count=0
	for word in words:
		if(word=="\n"):
			g.write(word)
			break
		if(count==0):
			g.write(word)
		else:
			g.write(word+" ")
		count=abs(1-count)
