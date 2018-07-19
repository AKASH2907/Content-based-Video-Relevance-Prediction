from keras.models import *
from keras.layers import *
from os.path import join
import pandas as pd
import numpy as np
import csv
from keras.optimizers import Adam

datapath = 'shows/track_1_shows/feature/'


# Read csv files
def read_csv_file(f):
    read_file = pd.read_csv(f, header=None)
    read_file.fillna(-1, inplace=True)
    return read_file


# Loading Dataset
def load_data(datapath, show_id):
    shows_c3d = join(datapath, str(show_id) + '/' + str(show_id) + '-c3d-pool5.npy')
    c3d_data = np.load(shows_c3d)
    return c3d_data


# List of all the C3D features present in TV Shows
def make_list():
    train_X_c3d_features = []

    for i in range(7536):
        c3d_feature = load_data(datapath, i)
        c3d_feature = np.asarray(c3d_feature)
        train_X_c3d_features.append(c3d_feature)

    print("Features loaded..........")
    return train_X_c3d_features


# Model using c3d features only
def model_c3d():
    visible1 = Input(shape=(512,))
    dense1 = Dense(256, activation='relu')(visible1)
    output = Dense(7536)(dense1)
    model = Model(inputs=visible1, outputs=output)
    # model.summary()

    # Learning rate - 10e-4 (20 epochs), 10e-5(Next 10 epochs)
    adam = Adam(lr=0.00001)
    model.compile(optimizer=adam, loss='cosine_proximity', metrics=['accuracy'])
    return model


# Ground Truth matrix
def ground_truth():
    # Read relevance file given
    relevance_file = pd.read_csv('relevance_train_shows.csv', header=None)
    relevance_file.fillna(-1, inplace=True)

    # Initialize ground truth matrix of shape= (3000, 7536)
    gd = np.zeros([relevance_file.shape[0], 7536])
    print("Ground Truth made ...............")

    for j in range(relevance_file.shape[0]):

        lists = []
        for i in range(relevance_file.shape[1]):
            if relevance_file[i][j] != -1:
                lists.append(relevance_file[i][j])
        for item, target in enumerate(lists[1:], start=1):
            target = int(target)
            gd[j][target] = 100
    print(gd.shape)
    return gd


# Ground Truth matrix and c3d features list
gd = ground_truth()
train_c3d = make_list()


# Training model
def train_model():
    train_x = read_csv_file('relevance_train_shows.csv')[0]
    model = model_c3d()

    # Loading pre-trained weights
    # model.load_weights('./c3d_256d/1layer_50_30f.h5')
    print("Training Model...............")

    # Number of epochs
    for epoch in range(10):
        for i, j in enumerate(train_x):
            train_c3d[j] = np.reshape(train_c3d[j], (1, 512))
            # Batch wise training
            val = model.train_on_batch([train_c3d[j]], gd[i].reshape(1, 7536))

            # Print number of epochs, iteration number, loss
            print(epoch, i, val)
            # Save model after 50 iterations
            if i % 50 == 0 and i % 100 != 0:
                model.save_weights('c3d_50_30f.h5')
            # Save model after 100 iterations
            if i % 100 == 0:
                model.save_weights('c3d_100_30f.h5')

        # Save model after 1 epoch
        model.save_weights('./c3d_epoch_30f.h5')

    return model


# Final prediction on Validation and testing
def predict():
    final = []
    # Loading model
    predict_model = model_c3d()
    # Loading trained weights
    predict_model.load_weights('c3d_epoch_30f.h5')

    # Read validation file
    val_x = read_csv_file('shows/track_1_shows/split/val.csv')[0]
    for i in val_x:
        y_predict = predict_model.predict([train_c3d[i].reshape(1, 512)])
        sort = np.argsort(y_predict)
        print(i, sort)
        final.append(np.flip(sort[0], axis=0))

    # Final matrix of values of prediction
    final = np.asarray(final)
    with open('validation.csv', 'w+') as f:
        csvWriter = csv.writer(f, delimiter=',')
        csvWriter.writerows(final)


if __name__ == '__main__':
    train_model()
    # predict()

