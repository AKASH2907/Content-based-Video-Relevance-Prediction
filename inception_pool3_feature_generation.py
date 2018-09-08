from __future__ import print_function

import argparse
import os
import sys

from moviepy.editor import VideoFileClip
import numpy as np
import scipy.misc
from tqdm import tqdm
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
from keras.applications import InceptionV3
from os.path import isfile, join
from os import rename, listdir, rename, makedirs
# batch_size = 32

def crop_center(im):
    h, w = im.shape[0], im.shape[1]

    if h < w:
        return im[0:h,int((w-h)/2):int((w-h)/2)+h,:]
    else:
        return im[int((h-w)/2):int((h-w)/2)+w,0:w,:]


def extract_features(input_dir, model_type='inceptionv3', batch_size=32):

    # Load desired ImageNet model
    model = None
    input_shape = (224, 224)

    model = InceptionV3(include_top=True, weights='imagenet')
    shape = (299, 299)
    model = Model(model.inputs, output=model.layers[-2].output)

    tr = listdir(input_dir)
    tr.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    for i in tr[65:70]:

        file= join(input_dir, i)
        video_filenames = listdir(file)
        print(video_filenames)

        save_path = './feature'
        features_path = join(save_path, i)
        # Go through each video and extract features
        for video_filename in tqdm(video_filenames):

            # Open video clip for reading
            clip = VideoFileClip( os.path.join(file, video_filename) ) 
            print(clip)


            # Sample frames at 1fps
            fps = int( np.round(clip.fps) )
            frames = [scipy.misc.imresize(crop_center(x.astype(np.float32)), shape)
                      for idx, x in enumerate(clip.iter_frames()) if idx % fps == fps//2]


            n_frames = len(frames)

            frames_arr = np.empty((n_frames,)+shape+(3,), dtype=np.float32)
            for idx, frame in enumerate(frames):
                frames_arr[idx,:,:,:] = frame

            frames_arr = preprocess_input(frames_arr)

            features = model.predict(frames_arr, batch_size=batch_size)
            # print(features)
            print(features.shape)

            np.save(features_path + '/' + i + '-inception-pool3.npy', features)
            print(features_path + '/' + i + '-inception-pool3.npy')


        #     break
        # break


if __name__ == '__main__':

    inputs = './trailers_id'
    extract_features(inputs)




