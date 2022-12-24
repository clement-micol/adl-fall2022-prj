import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.image import rot90, random_flip_left_right, random_brightness, random_hue, random_saturation, random_contrast
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.applications.inception_v3 import InceptionV3

class DataAugLayer(Layer):

    def __init__(self, delta_bright=64/255, delta_sat=0.25, delta_hue=0.04, delta_contrast=0.75):
        super(DataAugLayer,self).__init__()
        self.bright_max = delta_bright
        self.sat_max = delta_sat
        self.hue_max = delta_hue
        self.contrast_max = delta_contrast
    
    def call(self,inputs):
        augmented_inputs = random_brightness(inputs,self.bright_max)
        augmented_inputs = random_saturation(augmented_inputs,0,self.sat_max)
        augmented_inputs = random_hue(augmented_inputs,self.hue_max)
        augmented_inputs = random_contrast(augmented_inputs,0,self.contrast_max)
        random_rot = np.random.randint(0,4)
        augmented_inputs = rot90(augmented_inputs,k=random_rot)
        augmented_inputs = random_flip_left_right(augmented_inputs)
        return 2*(augmented_inputs/255-1/2)

def get_InceptionHierarchicalMagnification_model(
    level_of_zooms,
    trainable = True,
    hidden_dim = 256,
    dropout_rate = 0.3,
    ):
    level_of_zooms = sorted(level_of_zooms,key=lambda x: int(x[-1]),reverse=True)   
    inception_tower = {
    zoom : InceptionV3(weights="imagenet", include_top = False, input_shape=(299,299,3),pooling="max") for zoom in level_of_zooms
    }
    for zoom in level_of_zooms:
        inception_tower[zoom]._name = "inception_v3_"+zoom
        inception_tower[zoom].trainable = trainable
    patches_input = {zoom : Input(shape=(299,299,3)) for zoom in level_of_zooms}
    extracted_feature = None
    for zoom in level_of_zooms:
        rotated_patch_input = DataAugLayer()(patches_input[zoom])
        if extracted_feature == None :
            concat_features = inception_tower[zoom](rotated_patch_input)
        else :
            concat_features = tf.concat([extracted_feature,inception_tower[zoom](rotated_patch_input)],axis=1)
        extracted_feature = Dense(hidden_dim,activation="relu")(concat_features)
        extracted_feature = Dropout(dropout_rate)(extracted_feature)
    encoded_features = Dense(2*hidden_dim,activation="relu")(extracted_feature)
    encoded_features = Dropout(dropout_rate)(encoded_features)
    output = Dense(1,activation="sigmoid")(encoded_features)
    model = Model(inputs=patches_input,outputs=[output])
    return model