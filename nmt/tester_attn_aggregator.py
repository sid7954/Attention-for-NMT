import tensorflow as tf
sess = tf.InteractiveSession()

inputs=tf.constant([ [
					[2,2] , [3,3] , [4,4] , [6,6] , [7,7] , [8,8] 
				]
				,
				[
					[6,6] , [7,7] , [8,8] , [10,10] , [11,11] , [12,12]
				]  ])
#Batch , Encoder parts
ishape=inputs.get_shape().as_list()
print(ishape[0],ishape[1],ishape[2])

class AttentionAggregator():
	def __init__(self,
				lstm_size=1,
				**kwargs):
		self.lstm_size=lstm_size
		self.cell=tf.contrib.rnn.BasicLSTMCell(lstm_size)
		self.kernel=self.add_variable('kernel',
                                    shape=[lstm_size, 1],
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    dtype=self.dtype,
                                    trainable=True)

	def call(self,inputs,state):
		inputs_to_lstm = tf.reshape(inputs,[-1,1])
		ishape=inputs.get_shape().as_list()
		output, state = self.cell(inputs_to_lstm, self.state)
		output=output*self.kernel
		output=tf.reshape(output,[ishape[0],-1])
		return output, state
