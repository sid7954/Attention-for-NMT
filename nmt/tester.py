import tensorflow as tf
sess = tf.InteractiveSession()

encoder_parts=tf.constant([ [
					[2,2] , [3,3] , [4,4] 
				]
				,
				[
					[6,6] , [7,7] , [8,8] 
				]  
				, 
				[
					[10,10] , [11,11] , [12,12]      
				] ])

eshape=encoder_parts.get_shape().as_list()
print(eshape[0],eshape[1],eshape[2])
new_encoder_parts=encoder_parts
l=2
for i in range(eshape[1]-l,eshape[1]-1):
	print("i ",i)
	temp_add=tf.slice(encoder_parts,[0,0,0],[eshape[0],i+1,eshape[2]])
	for j in range(1,eshape[1]-i):
		print("j ",j)
		temp_e=tf.slice(encoder_parts,[0,j,0],[eshape[0],i+1,eshape[2]])
		print(temp_e)
		temp_add=tf.add(temp_add,temp_e)
	if(i==eshape[1]-l):
		new_encoder_parts=temp_add
	else:	
		new_encoder_parts=tf.concat([temp_add, new_encoder_parts],1)
	print("new_encoder_parts",new_encoder_parts)
if not(l==1):
	new_encoder_parts=tf.concat([encoder_parts,new_encoder_parts],1)

b= sess.run([new_encoder_parts])
print(b)