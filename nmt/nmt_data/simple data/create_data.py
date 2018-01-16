import string
import sys
import random
import numpy as np

orig_stdout=sys.stdout

n=15
num=10
S=np.zeros(num)
B=np.zeros(num)

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

for i in range(11,21):
	S[i-11]=i
	f1.write(str(i)+"\n")
	f2.write(str(i)+"\n")
for i in range(21,31):
	B[i-21]=i
	f1.write(str(i)+"\n")
input_string=np.zeros(n)
pos=np.arange(n)

for s in range(1000):
	input_string=np.zeros(n)
	k=random.randint(1,n/3)
	selected_char=np.random.choice(S,k,replace=False)
	selected_pos=np.random.choice(pos,k,replace=False)
	# print "selected chars are ",selected_char
	# print "selected pos are ",selected_pos
	for i in range(k):
		input_string[selected_pos[i]]=selected_char[i]
	for i in range(n):
		#np.random.seed(i)
		if (int(input_string[i])==0):
			input_string[i]=np.random.choice(B,1,replace=False)[0]

	output_string=np.zeros(k)
	for i in range(k):
		output_string[i]=selected_char[i]
	temp2=sorted(output_string)
	# print "input ",input_string
	# print "output ",temp2 
	if(s<900):
		for j in range(len(input_string)):
			f3.write(str(int(input_string[j]))+" ")
		f3.write("\n")
		for j in range(len(temp2)):
			f4.write(str(int(temp2[j]))+" ")
		f4.write("\n")
	elif(s<950):
		for j in range(len(input_string)):
			f5.write(str(int(input_string[j]))+" ")
		f5.write("\n")
		for j in range(len(temp2)):
			f6.write(str(int(temp2[j]))+" ")
		f6.write("\n")
	else:
		for j in range(len(input_string)):
			f7.write(str(int(input_string[j]))+" ")
		f7.write("\n")
		for j in range(len(temp2)):
			f8.write(str(int(temp2[j]))+" ")
		f8.write("\n")