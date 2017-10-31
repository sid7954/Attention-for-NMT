f=open("small_train.tgt","r")
g=open("small_train.src","w+")

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