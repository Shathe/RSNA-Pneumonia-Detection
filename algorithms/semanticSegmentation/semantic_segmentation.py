import os
import csv
import random
import pydicom
import MnasNet
import numpy as np
import pandas as pd
from skimage import measure
from skimage.transform import resize
import math
import PneumoniaGenerator
import tensorflow as tf
from tensorflow import keras
import cv2
import random
from matplotlib import pyplot as plt
np.random.seed(7)
random.seed(7)

'''
The code below loads the Pneumonia table and transforms it into a dictionary.

Pneumonia table:
	Table contains [filename : pneumonia location] pairs per row.

	If a filename contains multiple pneumonia, the table contains multiple rows with the same filename but different pneumonia locations.
	If a filename contains no pneumonia it contains a single row with an empty pneumonia location.

The dictionary uses the filename as key and a list of pneumonia locations in that filename as value.
If a filename is not present in the dictionary it means that it contains no pneumonia.

Example: 'e7b3fd9c-d51f-42dd-9f90-25160990faa8': [[211, 260, 150, 524], [564, 274, 223, 419]]
'''

def get_pneumonia_locations(file_table_pneumonia):
	# empty dictionary
	pneumonia_locations = {}
	# load table
	with open(os.path.join(file_table_pneumonia), mode='r') as infile:
		# open reader
		reader = csv.reader(infile)
		# skip header
		next(reader, None)
		# loop through rows
		for rows in reader:
			# retrieve information
			filename = rows[0]
			location = rows[1:5]
			pneumonia = rows[5]
			# if row contains pneumonia add label to dictionary
			# which contains a list of pneumonia locations per filename
			if pneumonia == '1':
				# convert string to float to int
				location = [int(float(i)) for i in location]
				# save pneumonia location in dictionary
				if filename in pneumonia_locations:
					pneumonia_locations[filename].append(location)
				else:
					pneumonia_locations[filename] = [location]

	return pneumonia_locations 

# cosine learning rate annealing
def cosine_annealing(x):
	lr = 0.001
	epochs = 200
	now_lr = (lr - 0.000007) * math.pow(1 - x / 1. / epochs, 0.9) + 0.000007
	print ('lr: ' + str(now_lr))
	return  now_lr

	# define iou or jaccard loss function
def iou_loss(y_true, y_pred):
	y_true = tf.reshape(y_true, [-1])
	y_pred = tf.reshape(y_pred, [-1])
	intersection = tf.reduce_sum(y_true * y_pred)
	score = (intersection + 1.) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection + 1.)
	return 1 - score

# combine bce loss and iou loss
def iou_bce_loss(y_true, y_pred):
	return 0.4 * keras.losses.binary_crossentropy(y_true, y_pred) + 0.6 * iou_loss(y_true, y_pred)

# mean iou as a metric
def mean_iou(y_true, y_pred):
	y_pred = tf.round(y_pred)
	intersect = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
	union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
	smooth = tf.ones(tf.shape(intersect))
	return tf.reduce_mean((intersect + smooth) / (union - intersect + smooth))

if __name__ == "__main__":

	# load pneumonia info
	pneumonia_locations = get_pneumonia_locations('../../data/stage_1_train_labels.csv')
	print (str(len(pneumonia_locations)) + ' patietns with pneumonia')

	# load and shuffle image filenames
	folder = '../../data/stage_1_train_images'
	filenames = os.listdir(folder)
	random.shuffle(filenames)
	# split into train and validation filenames
	n_elems = len(filenames)
	percentage_validation = 0.05
	n_valid_samples = int(n_elems * percentage_validation)
	n_train_samples = len(filenames) - n_valid_samples
	train_filenames = filenames[n_valid_samples:]
	valid_filenames = filenames[:n_valid_samples]
	print('n train samples', n_train_samples)
	print('n valid samples', n_valid_samples)

	IMAGE_SIZE = 256
	BATCH_SIZE = 8



	# create network and compiler
	model = MnasNet.FCMnasNet(n_classes=1, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1), alpha = 2)
	model.compile(optimizer='adam',
				  loss=iou_bce_loss,
				  metrics=['accuracy', mean_iou])


	learning_rate = keras.callbacks.LearningRateScheduler(cosine_annealing)

	# create train and validation generators
	train_gen = PneumoniaGenerator.PneumoniaGenerator(folder, train_filenames, pneumonia_locations, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, shuffle=True, augment=True, predict=False)
	valid_gen = PneumoniaGenerator.PneumoniaGenerator(folder, valid_filenames, pneumonia_locations, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, shuffle=False, predict=False)

	print(model.summary())	

	weights_filename = './models/weights.hdf5'

	cp_callback = tf.keras.callbacks.ModelCheckpoint(weights_filename, save_weights_only=True, verbose=1)


	try:
		model.load_weights(weights_filename)
		print ('model loaded')

	except Exception:
		print ('model not loaded')

	history = model.fit_generator(train_gen, validation_data=valid_gen, callbacks=[learning_rate, cp_callback], epochs=200, shuffle=True)
	'''
	plt.figure(figsize=(12,4))
	plt.subplot(131)
	plt.plot(history.epoch, history.history["loss"], label="Train loss")
	plt.plot(history.epoch, history.history["val_loss"], label="Valid loss")
	plt.legend()
	plt.subplot(132)
	plt.plot(history.epoch, history.history["acc"], label="Train accuracy")
	plt.plot(history.epoch, history.history["val_acc"], label="Valid accuracy")
	plt.legend()
	plt.subplot(133)
	plt.plot(history.epoch, history.history["mean_iou"], label="Train iou")
	plt.plot(history.epoch, history.history["val_mean_iou"], label="Valid iou")
	plt.legend()
	plt.show()
	'''

	# load and shuffle filenames
	folder = '../../data/stage_1_test_images'
	test_filenames = os.listdir(folder)
	print('n test samples:', len(test_filenames))
	BATCH_SIZE = 10
	# create test generator with predict flag set to True
	test_gen = PneumoniaGenerator.PneumoniaGenerator(folder, test_filenames, None, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, shuffle=False, predict=True)

	# create submission dictionary
	submission_dict = {}
	# loop through testset
	for imgs, filenames in test_gen:
		# predict batch of images
		preds = model.predict(imgs)
		# loop through batch
		for pred, filename in zip(preds, filenames):
			# resize predicted mask

			pred = cv2.resize(pred, (1024, 1024), interpolation=cv2.INTER_NEAREST)
			# threshold predicted mask
			comp = pred[:, :] > 0.5
			# apply connected components
			comp = measure.label(comp, neighbors=8)
			# apply bounding boxes
			predictionString = ''
			for region in measure.regionprops(comp):
				# retrieve x, y, height and width
				y, x, y2, x2 = region.bbox
				height = y2 - y
				width = x2 - x
				# proxy for confidence score
				conf = np.mean(pred[y:y+height, x:x+width])
				# add to predictionString
				predictionString += str(conf) + ' ' + str(x) + ' ' + str(y) + ' ' + str(width) + ' ' + str(height) + ' '
			# add filename and predictionString to dictionary
			filename = filename.split('.')[0]
			submission_dict[filename] = predictionString
		# stop if we've got them all
		print(len(submission_dict)*100./len(test_filenames) )
		if len(submission_dict)*100./len(test_filenames) >= 100.:
			break
			
	print("Done predicting...")
			
	# save dictionary as csv file
	sub = pd.DataFrame.from_dict(submission_dict, orient='index')
	sub.index.names = ['patientId']
	sub.columns = ['PredictionString']
	sub.to_csv('submission.csv')


