
import numpy as np 
import cv2
import sys  
sys.path.append('./models')
import keras
from keras.models import Model
from models import AttentionResNetCifar10
from feature_visualize import get_row_col,visualize_feature_map


def rotate(image, angle, center=None, scale=1.0):
	(h, w) = image.shape[:2] 
	if center is None: 
		center = (w // 2, h // 2) 
	M = cv2.getRotationMatrix2D(center, angle, scale) 
	rotated = cv2.warpAffine(image, M, (w, h)) 
	return rotated 


def predict(image_path,TTA=True):
	# build a model
	model = AttentionResNetCifar10(n_classes=10)
	model.load_weights('logs/weights/residual_attention_71_0.89.h5')
	#model = Model(inputs=model.input, outputs=model.get_layer('residual_attention_stage1').output)
	labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

	image = cv2.imread(image_path)

	if(TTA):
		h_flip = cv2.flip(image, 1)		# 水平翻转
		v_flip = cv2.flip(image, 0)		# 垂直翻转
		rotated45 = rotate(image, 45)	#旋转
		rotated90 = rotate(image, 90)
		rotated180 = rotate(image, 180)
		rotated270 = rotate(image, 270)
		image_list = []
		image_list.append(h_flip)
		image_list.append(v_flip)
		image_list.append(rotated45)
		image_list.append(rotated90)
		image_list.append(rotated180)
		image_list.append(rotated270)

		pred_list = []
		for i in range(len(image_list)):
			test = cv2.resize(image_list[i], (32, 32))	#将测试图片转化为model需要的大小
			test = np.array(test, np.float32) / 255	#归一化送入，float/255
			test = test.reshape(1,32,32,3)	#model需要的是1张input_size*input_size得3通道rgb图片，所以转化为(1,input_size,input_size,3)

			pred = model.predict(test)
			pred_list.append(pred)

		TTA_pred = np.zeros(shape=(1,10))
		for i in range(len(pred_list)):
			TTA_pred = TTA_pred+pred_list[i]

		print('TTA_pred:',TTA_pred,'pred_shape:',TTA_pred.shape)
		max_score = np.where(TTA_pred==np.max(TTA_pred))
		label = labels[int(max_score[1])]
		print(label)

	else:
		test = cv2.resize(image, (32, 32))	#将测试图片转化为model需要的大小
		test = np.array(test, np.float32) / 255	#归一化送入，float/255
		test = test.reshape(1,32,32,3)	#model需要的是1张input_size*input_size得3通道rgb图片，所以转化为(1,input_size,input_size,3)

		pred = model.predict(test)
		print('prediction:',pred,'pred_shape:',pred.shape)

		max_score = np.where(pred==np.max(pred))
		print(max_score)
		label = labels[int(max_score[1])]
		print(label)


def visuable(image_path,name):
	# build a model
	model = AttentionResNetCifar10(n_classes=10)
	model.load_weights('logs/weights/residual_attention_71_0.89.h5')
	model = Model(inputs=model.input, outputs=model.get_layer('residual_attention_stage1').output)
	labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

	test = cv2.imread(image_path)
	test = cv2.resize(test, (32, 32))	#将测试图片转化为model需要的大小
	test = np.array(test, np.float32) / 255	#归一化送入，float/255
	test = test.reshape(1,32,32,3)	#model需要的是1张input_size*input_size得3通道rgb图片，所以转化为(1,input_size,input_size,3)
	block_pool_features = model.predict(test)
	print(block_pool_features.shape)

	feature = block_pool_features.reshape(block_pool_features.shape[1:])
	visualize_feature_map(feature,name)

if __name__ == '__main__':
	import tensorflow as tf
	from keras.backend.tensorflow_backend import set_session
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.5
	set_session(tf.Session(config=config))

	import os
	import random
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'

	file_names = next(os.walk('test_images'))[2]
	file_names = random.choice(file_names)
	filepath = os.path.join('test_images',file_names)
	print(filepath)
	predict(image_path=filepath,TTA=True)
	visuable(image_path=filepath,name=file_names.split('.')[0])



