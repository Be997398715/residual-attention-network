# In[1]:
import pickle
import numpy as np
import time
import sys  
sys.path.append('./models')
import matplotlib.pyplot as plt

import keras
from keras import backend as K
#from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from models import AttentionResNetCifar10

def main():
    # In[2]:
    # 开始下载数据集
    t0 = time.time()  

    DOWNLOAD = True
    # CIFAR10 图片数据集
    if(DOWNLOAD):
        (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()  # 32×32

    else:
        # load data
        with open('cifar-10-batches-py\\data_batch_1','rb') as f:
            dict1 = pickle.load(f,encoding='bytes')
        x = dict1[b'data']
        x = x.reshape(len(x), 3, 32, 32).astype('float32')
        y = np.asarray(dict1[b'labels'])
        X_test = x[0:int(0.2 * x.shape[0]), :, :, :]
        Y_test = y[0:int(0.2 * y.shape[0])]
        X_train = x[int(0.2 * x.shape[0]):x.shape[0], :, :, :]
        Y_train = y[int(0.2 * y.shape[0]):y.shape[0]]

    X_train = X_train.astype('float32')  # uint8-->float32
    X_test = X_test.astype('float32')
    X_train /= 255  # 归一化到0~1区间
    X_test /= 255
    print('训练样例:', X_train.shape, Y_train.shape,
          ', 测试样例:', X_test.shape, Y_test.shape)

    nb_classes = 10  # label为0~9共10个类别
    # Convert class vectors to binary class matrices
    Y_train = to_categorical(Y_train, nb_classes)
    Y_test = to_categorical(Y_test, nb_classes)
    print("取数据耗时: %.2f seconds ..." % (time.time() - t0))

    # In[3]:
    # define generators for training and validation data
    train_datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    val_datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True)

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    train_datagen.fit(X_train)
    val_datagen.fit(X_test)


    # In[4]:
    # build a model
    model = AttentionResNetCifar10(n_classes=nb_classes)


    # In[ ]:
    # 模型可视化模块
    #SVG(model_to_dot(model).create(prog='dot', format='svg'))

    # In[5]:
    # build callbacks
    # prepare usefull callbacks
    callbacks = [
        ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=7, min_lr=1e-9, epsilon=0.01, verbose=1),
        EarlyStopping(monitor='val_acc', min_delta=0, patience=14, verbose=1),
        ModelCheckpoint(monitor='val_acc',
                         filepath='logs/weights/residual_attention_{epoch:02d}_{val_acc:.2f}.h5',
                         save_best_only=True,
                         save_weights_only=True,
                         mode='auto',
                         verbose=1,
                         period=5)
                ]

    # In[6]:
    # define metrics, optimizer, loss
    def recall(y_true, y_pred):
        '''计算多标签分类中的recall'''
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        '''计算多标签分类中的precision'''
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def f1_score(y_true, y_pred):
        precision_ = precision(y_true, y_pred)
        recall_ = recall(y_true, y_pred)
        f1 = 2*((precision_*recall_)/(precision_+recall_+K.epsilon()))
        return f1

    model.compile(keras.optimizers.Adam(lr=1e-06), loss='categorical_crossentropy', metrics=['accuracy',precision,recall,f1_score])


    # In[7]:

    #model.load_weights('logs/weights/residual_attention_66_0.89.h5')
    batch_size = 8
    
    history = model.fit_generator(train_datagen.flow(X_train, Y_train, batch_size=batch_size),
                        steps_per_epoch=len(X_train)//batch_size, epochs=100,
                        validation_data=val_datagen.flow(X_test, Y_test, batch_size=batch_size), 
                        validation_steps=len(X_test)//batch_size,
                        callbacks=callbacks, initial_epoch=66, shuffle=True, verbose=2)

    # In[8]:
    loss,accuracy,precision,recall,f1_score= model.evaluate_generator(val_datagen.flow(X_test, Y_test), steps=len(X_test)/batch_size, use_multiprocessing=False, verbose=2)
    print('test_loss:',loss,'test_acc:',accuracy,'precision:',precision,'recall:',recall,'f1_score',f1_score)

    # 显示训练的总曲线
    def plot_acc_loss(h, nb_epoch):
        acc, loss, val_acc, val_loss = h.history['acc'], h.history['loss'], h.history['val_acc'], h.history['val_loss']
        plt.figure(figsize=(15, 5))
        plt.subplot(121)
        plt.plot(range(nb_epoch), acc, label='Train')
        plt.plot(range(nb_epoch), val_acc, label='Test')
        plt.title('Accuracy over ' + str(nb_epoch) + ' Epochs', size=15)
        plt.legend()
        plt.grid(True)
        plt.subplot(122)
        plt.plot(range(nb_epoch), loss, label='Train')
        plt.plot(range(nb_epoch), val_loss, label='Test')
        plt.title('Loss over ' + str(nb_epoch) + ' Epochs', size=15)
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.savefig('logs/train_test_history.png')
    plot_acc_loss(history, nb_epoch=100)

if __name__ == '__main__':
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    #config.gpu_options.visible_device_list = "0"
    set_session(tf.Session(config=config))

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()