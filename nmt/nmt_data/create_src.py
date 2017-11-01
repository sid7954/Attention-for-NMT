import codecs

f=codecs.open("train.tgt","r","utf-8")
g=codecs.open("train.src","w+","utf-8")

while True:
    char=f.read(1)
    if not char: 
        break
    else:
    	if not (char==" "):
    		if(char=="\n"):
    			g.write(char);
    		else:
    			g.write(char+" ")

f.close()
g.close()