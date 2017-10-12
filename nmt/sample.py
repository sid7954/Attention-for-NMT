import tensorflow as tf
import numpy as np
sess = tf.InteractiveSession()

encoder_parts=tf.constant([ [
					[2,2] , [3,3] , [4,4] , [5,5] 
				]
				,
				[
					[6,6] , [7,7] , [8,8] , [9,9]
				]  
				, 
				[
					[10,10] , [11,11] , [12,12] , [13,13]     
				] ],dtype=tf.float32)

eshape=encoder_parts.get_shape().as_list()
print(eshape[0],eshape[1],eshape[2])
new_encoder_parts=encoder_parts
l=4
enc=tf.reshape(encoder_parts,[eshape[0],eshape[1],eshape[2],1])

# k=tf.constant([[ [[1]] ],[ [[1]] ]]  ,dtype=tf.float32)
# ans=tf.reshape(tf.nn.conv2d(enc, k, strides=[1,1,1,1], padding='VALID'),[eshape[0],eshape[1]-1,eshape[2]])


for i in range(1,l):
	temp=tf.placeholder(tf.float32, shape=(i+1,1,1,1))
	k = tf.fill(tf.shape(temp), 1.0)

	# ans=tf.reshape(tf.nn.conv2d(enc, k, strides=[1,1,1,1], padding='VALID'),[eshape[0],eshape[1]-i,eshape[2]])	
	ans=tf.squeeze(tf.nn.conv2d(enc, k, strides=[1,1,1,1], padding='VALID'),3)	
	new_encoder_parts=tf.concat([new_encoder_parts,ans],1)

b= sess.run([new_encoder_parts])

a= sess.run([encoder_parts])
print(a)
print(b)
