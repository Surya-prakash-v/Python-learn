#V7777

import keras
from keras.models import Model
from keras.layers import Flatten, Input, AveragePooling2D, merge, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization,GlobalAveragePooling2D
from keras.layers import Concatenate
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Lambda,SeparableConv2D
from keras.layers.merge import concatenate
import tensorflow as tf


def space_to_depth_x2(x):
    return tf.space_to_depth(x, block_size=2)
  
def resNetBlock(input_image):
 
  # Conv Layer 2
  layer2 = SeparableConv2D(1024, 3, strides=(1,1), padding='same', use_bias=False)(input_image)
  layer2 = BatchNormalization()(layer2)
  layer2 = LeakyReLU(alpha=0.1)(layer2)

  # Conv Layer 3
  layer3 = SeparableConv2D(256,3, strides=(1,1), padding='same', use_bias=False)(layer2)
  layer3 = BatchNormalization()(layer3)
  
  return layer3  

input_shape = Input(shape=(None,None,3)) 
# Layer 1
layer_1 = SeparableConv2D(64, 3, strides=(1,1), padding='same', use_bias=False)(input_shape)
layer_1 = BatchNormalization()(layer_1)
layer_1 = LeakyReLU(alpha=0.1)(layer_1)
  
# Layer 1
layer_2 = SeparableConv2D(128, 3, strides=(1,1), padding='same', use_bias=False)(layer_1)
layer_2 = BatchNormalization()(layer_2)
layer_2 = LeakyReLU(alpha=0.1)(layer_2)
  
# Layer 2
layer_3 = SeparableConv2D(256, 3, strides=(1,1), padding='same', use_bias=False)(layer_2)
layer_3 = BatchNormalization()(layer_3)
layer_3 = LeakyReLU(alpha=0.1)(layer_3)
 
# Layer 2
layer_4 = SeparableConv2D(512, 3, strides=(1,1), padding='same', use_bias=False)(layer_3)
layer_4 = BatchNormalization()(layer_4)
layer_4 = LeakyReLU(alpha=0.1)(layer_4)
  
resNetBlock1 = resNetBlock(layer_4)
concact1 = concatenate([resNetBlock1, layer_3])
layer_l = LeakyReLU(alpha=0.1)(concact1)
maxpool1 = MaxPooling2D(pool_size=(2, 2))(layer_l)

resNetBlock2 = resNetBlock(maxpool1)
adjusted_resNetBlock1 = Lambda(space_to_depth_x2)(resNetBlock1)
concact2 = concatenate([resNetBlock2, adjusted_resNetBlock1])
layer_l2 = LeakyReLU(alpha=0.1)(concact2)
maxpool2 = MaxPooling2D(pool_size=(2, 2))(layer_l2)

resNetBlock3 = resNetBlock(maxpool2)
adjusted_resNetBlock2 = Lambda(space_to_depth_x2)(resNetBlock2)
concact3 = concatenate([resNetBlock3, adjusted_resNetBlock2])
layer_l3 = LeakyReLU(alpha=0.1)(concact3)

#maxpool3 = MaxPooling2D(pool_size=(2, 2))(layer_l3)

#resNetBlock4 = resNetBlock(maxpool3)
#layer_ant4 = Conv2D(256, (1,1), strides=(1,1), padding='same', use_bias=False)(resNetBlock4)
#adjusted_resNetBlock3 = Lambda(space_to_depth_x2)(resNetBlock3)
#concact4 = concatenate([layer_ant4, adjusted_resNetBlock3])
#layer_l4 = LeakyReLU(alpha=0.1)(concact4)

layer_ant = SeparableConv2D(200, 3, strides=(1,1), padding='same', use_bias=False)(layer_l3)
global_avg = GlobalAveragePooling2D()(layer_ant)

output = Activation('softmax')(global_avg)

model = Model(inputs=input_shape, outputs=output)

model.summary()
