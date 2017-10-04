import string
import sys
import random
import numpy as np
import codecs

orig_stdout=sys.stdout

f1=codecs.open("tst2012.en","r","utf-8")
f2=codecs.open("train.src","w+","utf-8")
f3=codecs.open("train.tgt","w+","utf-8")

while True:
    char=f1.read(1)
    if not char: 
        break
    else:
    	f3.write(char)
    	if not (char==" "):
    		f2.write(char+" ")

f1.close()
f2.close()
f3.close()
f1=codecs.open("tst2013.en","r","utf-8")
f2=codecs.open("test.src","w+","utf-8")
f3=codecs.open("test.tgt","w+","utf-8")

while True:
    char=f1.read(1)
    if not char: 
        break
    else:
    	f3.write(char)
    	if not (char==" "):
    		f2.write(char+" ")

f1.close()
f2.close()
f3.close()
f1=codecs.open("dev2013.en","r","utf-8")
f2=codecs.open("dev.src","w+","utf-8")
f3=codecs.open("dev.tgt","w+","utf-8")

while True:
    char=f1.read(1)
    if not char: 
        break
    else:
    	f3.write(char)
        if not (char==" "):
            f2.write(char)
            f2.write(" ")