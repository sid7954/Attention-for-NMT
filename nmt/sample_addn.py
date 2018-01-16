import tensorflow as tf
sess = tf.InteractiveSession()

encoder_parts=tf.constant([ [
					[2,2,3] , [3,3,4]  
				]
				,
				[
					[6,6,7] , [7,7,8]  
				]  
				, 
				[
					[10,10,11] , [11,11,12]       
				] ])

biass= tf.constant([1 ,2, 3])

new_encoder_parts=encoder_parts + biass

b= sess.run([new_encoder_parts])
print(b)