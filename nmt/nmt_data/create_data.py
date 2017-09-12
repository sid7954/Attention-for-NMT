import string
import sys
import random
import numpy as np

orig_stdout=sys.stdout

def map_char(T):
	mapping={}
	chosen=np.random.choice(T,num,replace=False)
	for i in range(num):
		mapping[S[i]]=chosen[i]
	return mapping

n=30
num=30
S=np.chararray(num)
B=np.chararray(num)
T=np.chararray(num)

f1=open("vocab.src","w+")
f2=open("vocab.tgt","w+")
f3=open("train.src","w+")
f4=open("train.tgt","w+")
f5=open("dev.src","w+")
f6=open("dev.tgt","w+")
f7=open("test.src","w+")
f8=open("test.tgt","w+")

f1.write("<unk>\n<s>\n</s>\n")
f2.write("<unk>\n<s>\n</s>\n")

for i in range(35,65):
	S[i-35]=(chr(i))
	f1.write(chr(i)+"\n")
for i in range(65,95):
	B[i-65]=(chr(i))
	f1.write(chr(i)+"\n")
for i in range(95,125):
	T[i-95]=(chr(i))
	f2.write(chr(i)+"\n")
mapping=map_char(T)
input_string=[' ' for i in range(0,n)]
pos=np.arange(n)

for s in range(10000):
	random.seed(s)
	np.random.seed(s)
	k=random.randint(1,n/3)
	selected_char=np.random.choice(S,k,replace=False)
	selected_pos=np.random.choice(pos,k,replace=False)
	for i in range(k):
		input_string[selected_pos[i]]=selected_char[i]
	for i in range(n):
		np.random.seed(i)
		if (input_string[i]==' '):
			input_string[i]=np.random.choice(B,1,replace=False)[0]

	output_string=[' ' for i in range(0,k)]

	for i in range(k):
		output_string[i]=mapping[selected_char[i]]
	temp=''.join(output_string)
	temp2=sorted(temp)
	if(s<5000):
		for j in range(len(input_string)):
			f3.write(str(input_string[j])+" ")
		f3.write("\n")
		for j in range(len(temp2)):
			f4.write(str(temp2[j])+" ")
		f4.write("\n")
	elif (s<8000):
		for j in range(len(input_string)):
			f5.write(str(input_string[j])+" ")
		f5.write("\n")
		for j in range(len(temp2)):
			f6.write(str(temp2[j])+" ")
		f6.write("\n")
	else:
		for j in range(len(input_string)):
			f7.write(str(input_string[j])+" ")
		f7.write("\n")
		for j in range(len(temp2)):
			f8.write(str(temp2[j])+" ")
		f8.write("\n")