

def wavenet_prior(net, is_training):
	from ops import causal_conv, gated_cnn, residual_block
	import json
	with open('wavenet.json') as file:
		wavenet_parameters = json.load(file)

	net = causal_conv(net, **wavenet_parameters['preprocess'])
	skip_outputs = []
	num_layers = len(wavenet_parameters['dilation_rates'])
	print('num_layers:', num_layers)
	print('net_0:', net.shape)
	for i in range(num_layers):
		layer_id = 'layer_%d' % (i + 1)
		conv_args = wavenet_parameters['residual_stack']
		conv_args['dilation_rate'] = wavenet_parameters['dilation_rates'][i]
		net = tf.keras.layers.BatchNormalization()(net)
		skip, net = residual_block(net, **conv_args)
		print('net_%d:'%(i+1), net.shape, ' dr: %d' % conv_args['dilation_rate'])
		skip_outputs.append(skip)
		net = tf.keras.layers.Dropout(0.5)(net, training=is_training)
	net = sum(skip_outputs)
	net = tf.keras.activations.relu(net)
	net = causal_conv(net, **wavenet_parameters['postprocess1'])
	print('net:', net.shape)
	net = causal_conv(net, **wavenet_parameters['postprocess2'])
	print('net:', net.shape)

	shape = net.shape
	net = tf.reshape(net, [-1, shape[1] * shape[2]])

	return net

