f=open('vocab.tgt',"r")

sum=0
num=0
for line in f:
	sum=sum+ len(line)
	num=num+1

print float(sum)/float(num)