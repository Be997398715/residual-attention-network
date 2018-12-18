# coding: utf-8
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from pylab import *



def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row,col
 
def visualize_feature_map(img_batch,name):
    feature_map = img_batch
    print(feature_map.shape)
 
    feature_map_combination=[]
    plt.figure()
 
    num_pic = feature_map.shape[2]
    row,col = get_row_col(num_pic)
 
    for i in range(0,num_pic):
        feature_map_split=feature_map[:,:,i]
        feature_map_combination.append(feature_map_split)
        plt.subplot(row,col,i+1)
        plt.imshow(feature_map_split)
        axis('off')
 
    plt.savefig('logs/feature_map/'+name+'_feature_map.jpg')
    plt.show()
 
    # 各个特征图按1：1 叠加
    feature_map_sum = sum(ele for ele in feature_map_combination)
    plt.imshow(feature_map_sum)
    plt.savefig('logs/feature_map/'+name+'_feature_map_sum.jpg')
 
 
if __name__ == "__main__":
    base_model = VGG19(weights='imagenet', include_top=False)
    # model = Model(inputs=base_model.input, outputs=base_model.get_layer('block1_pool').output)
    # model = Model(inputs=base_model.input, outputs=base_model.get_layer('block2_pool').output)
    # model = Model(inputs=base_model.input, outputs=base_model.get_layer('block3_pool').output)
    # model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)
 
    img_path = input()
    img = image.load_img(img_path)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    block_pool_features = model.predict(x)
    print(block_pool_features.shape)
 
    feature = block_pool_features.reshape(block_pool_features.shape[1:])
 
    visualize_feature_map(feature)