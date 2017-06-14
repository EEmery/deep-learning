import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range


############################# Importa e carrega dados #############################

path = './../assignments/notMNIST.pickle'								# Caminho para arquivo ".pickle" com os dados
pickle_file = open(path, 'rb')											# Abre o arquivo para leitura-binaria (read binary (rb))
save = pickle.load(pickle_file)											# Carrega os dados em save

# Acessa os dados de treino (50%)
train_dataset = save['train_dataset']
train_labels = save['train_labels']

# Acessa os dados de teste (25%)
test_dataset = save['test_dataset']
test_labels = save['test_labels']

# Acessa os dados de validacao (25%)
valid_dataset = save['valid_dataset']
valid_labels = save['valid_labels']

# Limpa memoria (ajuda a aliviar a memoria RAM)
del save


###################### Variaveis e parametros da arquitetura ######################

# Relacionado aos dados de entrada
image_size = 28															# 28x28 pixels
num_labels = 10															# Caracteres de A a J
num_channels = 1														# Escala de cinza (Grayscale, sem RGB)

# Relacionados a arquitetura da rede
num_steps = 2000
batch_size = 20
filter_size = 5															# Tamanho do filtro (5x5)
out_channels = 16														# Numero de channels da imagem da saida da convolucao
num_hidden = 64															# Numero de neuronios na camada escondida


def reformat(dataset, labels):
	dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)		# Reformata para (num_images, image_size, image_size, num_channels)
	labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)							# Reformata para (num_images, num_labels)
	return dataset, labels

# Reformata os dados para se tornarem adequados para a alimentacao da rede
train_dataset, train_labels = reformat(train_dataset, train_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)


def accuracy(predictions, labels):
	"""
	Compara as predicoes com as respostas (labels) e transforma resultado em porcentagem
	"""
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])


############################## Define a Rede Neural ###############################

graph = tf.Graph()
with graph.as_default():
	
	# Placeholders dos dados de entrada
	dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
	labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
	
	# Placeholder dos dados de teste e validacao
	#tf_test_dataset = tf.constant(test_dataset)
	#tf_valid_dataset = tf.constant(valid_dataset)

	# Camada 1
	weights1 = tf.Variable(tf.truncated_normal([filter_size, filter_size, num_channels, out_channels], stddev=0.1))
	net1 = tf.nn.relu(tf.nn.conv2d(dataset, weights1, [1, 2, 2, 1], padding='SAME'))
	layer1 = tf.nn.relu(net1)
	
	# Camada 2
	weights2 = tf.Variable(tf.truncated_normal([filter_size, filter_size, out_channels, out_channels], stddev=0.1))
	net2 = tf.nn.relu(tf.nn.conv2d(layer1, weights2, [1, 2, 2, 1], padding='SAME'))
	layer2 = tf.nn.relu(net2)

	# Reformata rede
	shape = layer2.get_shape().as_list()
	reshaped_layer2 = tf.reshape(layer2, [shape[0], shape[1] * shape[2] * shape[3]])

	# Camada 3
	weights3 = tf.Variable(tf.truncated_normal([image_size // 4 * image_size // 4 * out_channels, num_hidden], stddev=0.1))
	net3 = tf.matmul(reshaped_layer2, weights3)
	layer3 = tf.nn.relu(net3)
	
	# Camada 4 (camada de saida)
	weights4 = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
	output = tf.matmul(layer3, weights4)

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=labels))
	optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

	train_prediction = tf.nn.softmax(output)


###################### Rotina de treinamento da Rede Neural #######################

with tf.Session(graph=graph) as sess:

	# Inicializa um grafo
	sess.run(tf.global_variables_initializer())

	for step in range(num_steps):

		# O offset vai "caminhando" em cada ciclo, ja que os dados ja estao embaralhados
		offset = (step * batch_size) % (train_labels.shape[0] - batch_size)

		# Separa um pedaco dos dados de treinamento (batch)
		batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
		batch_labels = train_labels[offset:(offset + batch_size), :]

		# Roda a rede neural
		_, l, predictions = sess.run([optimizer, loss, train_prediction], feed_dict={dataset: batch_data, labels: batch_labels})

		# Imprime resultados
		if (step+1)%50 == 0:
			print('Loss no ciclo %d: %f' % (step+1, l))
			print('Taxa de acerto no batch: %.1f%%\n' % accuracy(predictions, batch_labels))