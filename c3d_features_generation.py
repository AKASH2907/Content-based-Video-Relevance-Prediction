import skvideo.io
import keras.backend as K
from keras.models import Sequential, Model
from keras.utils.data_utils import get_file
from keras.layers.core import Dense, Dropout, Flatten
from sports1M_utils import preprocess_input, decode_predictions
from keras.layers.convolutional import Conv3D, MaxPooling3D, ZeroPadding3D
import numpy as np
from os.path import isfile, join
from os import rename, listdir, rename, makedirs
from shutil import copyfile, move

WEIGHTS_PATH = 'https://github.com/adamcasson/c3d/releases/download/v0.1/sports1M_weights_tf.h5'

def C3D(weights='sports1M'):
    if weights not in {'sports1M', None}:
        raise ValueError('weights should be either be sports1M or None')
    
    if K.image_data_format() == 'channels_last':
        shape = (16,112,112,3)
    else:
        shape = (3,16,112,112)
        
    model = Sequential()
    model.add(Conv3D(64, 3, activation='relu', padding='same', name='conv1', input_shape=shape))
    model.add(MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), padding='same', name='pool1'))
    
    model.add(Conv3D(128, 3, activation='relu', padding='same', name='conv2'))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool2'))
    
    model.add(Conv3D(256, 3, activation='relu', padding='same', name='conv3a'))
    model.add(Conv3D(256, 3, activation='relu', padding='same', name='conv3b'))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool3'))
    
    model.add(Conv3D(512, 3, activation='relu', padding='same', name='conv4a'))
    model.add(Conv3D(512, 3, activation='relu', padding='same', name='conv4b'))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool4'))
    
    model.add(Conv3D(512, 3, activation='relu', padding='same', name='conv5a'))
    model.add(Conv3D(512, 3, activation='relu', padding='same', name='conv5b'))
    model.add(ZeroPadding3D(padding=(0,1,1)))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool5'))
    
    model.add(Flatten())
    
    model.add(Dense(4096, activation='relu', name='fc6'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(0.5))
    model.add(Dense(487, activation='softmax', name='fc8'))

    if weights == 'sports1M':
        weights_path = get_file('sports1M_weights_tf.h5',
                                WEIGHTS_PATH,
                                cache_subdir='models',
                                md5_hash='b7a93b2f9156ccbebe3ca24b41fc5402')
        
        model.load_weights(weights_path)
    
    return model
    
if __name__ == '__main__':
    model = C3D(weights='sports1M')
    # for _ in range(6):
    #     model.layers.pop()
    # print(model.summary())
    # model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    model = Model(inputs=model.input, outputs=model.get_layer('pool5').output)
    # print(model.summary())
    video_path = './trailers/'
    save_path = './feature/'
    files = listdir(video_path)
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    # print(files)

    save_files = listdir(save_path)
    save_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    for i, j in zip(files, save_files):
        vid_path = join(video_path, i)
        # f_path = join()

        vid = skvideo.io.vread(vid_path)
        # # vid = vid[40:56]
        # print(len(vid))
        vid = preprocess_input(vid)
        
        preds = model.predict(vid)
        # mean_f = np.mean(preds, axis=3)
        # print(preds)
        preds_arr = np.asarray(preds)
        # print(preds_arr.shape)
        preds_arr = np.reshape(preds_arr, (4, 4, 512))
        # print(preds_arr.shape)

        print(np.mean(np.mean(preds_arr, axis=0), axis=0))
        mean_1d =np.mean(np.mean(preds_arr, axis=0), axis=0)

        features_path = join(save_path, j)

        np.save(features_path + '/' + str(j) + '-c3d-pool5.npy', mean_1d)
        
        # print(mean_f)
        # print(len(mean_f))
        # print(decode_predictions(preds))