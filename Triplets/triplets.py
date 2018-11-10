from keras.models import Model, Sequential
from keras.layers import TimeDistributed, Conv3D, MaxPooling3D, ZeroPadding3D, Flatten, Dense, Dropout, Reshape, Conv2D, Input, merge, ConvLSTM2D,  Bidirectional, LSTM, GRU, Activation
from keras.optimizers import Adam, RMSprop
from keras import backend as K
import pandas as pd
import numpy as np
from os.path import isfile, join
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

def read_csv_file(f):
    df = pd.read_csv(f,header=None)
    df.fillna(-1, inplace=True)
    rows, columns = df.shape
    # print(rows, columns)
    # df[column] [row]
    return df




def contrastive_loss(y, d):
    """ Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    margin = 2
    return 0.5*K.mean((1-y) * K.square(d) + (y) * K.square(K.maximum(margin - d, 0)))

def euclidean_distance(inputs):
    assert len(inputs) == 2, \
        'Euclidean distance needs 2 inputs, %d given' % len(inputs)
    u, v = inputs
    # print (K.sqrt((K.square(u - v))))
    return K.sqrt(K.sum(K.square(u - v),axis=1, keepdims=True))

# def model1():
	# model = Sequential()
	# model.add(TimeDistributed(Dense(512), input_shape=(2048,)))
	# model.add(TimeDistributed(Dense(256)))
	# model.add(Bidirectional(GRU(128,return_sequences=True)))
	# model.add(Bidirectional(GRU(128,return_sequences=True)))
inp1 = Input(shape=(None,2048))
dense11 = TimeDistributed(Dense(1024, activation='relu'))(inp1)
dense12 = TimeDistributed(Dense(512, activation='relu'))(dense11) 
# dense13 = TimeDistributed(Dense(256, activation='relu'))(dense12) 
gru11= Bidirectional(GRU(64,return_sequences=True), merge_mode='concat')(dense12) 
gru12= Bidirectional(GRU(64,return_sequences=False), merge_mode='concat')(gru11) 
# model = Model(input=inp, output=gru2)
# return model

# def model2():
inp2 = Input(shape=(512,))
dense21 = Dense(256, activation = 'relu')(inp2)
out2 = Dense(128, activation = 'relu')(dense21)
# out2 = Dense(128)(dense22)
# model = Model(input=inp, output=out)
# model = Sequential()
# model.add(Dense(256, input_shape=(512,)))
# model.add(Dense(256))
# model.add(Dense(128))
# return model


m1 = merge([gru12, out2], mode = 'concat')

dense = Dense(256, activation="sigmoid")(m1)

final_model = Model(input=[inp1,inp2], output=dense)

# final_model.summary()

input_11 = Input(shape=(None,2048))
input_21 = Input(shape=(None,2048))
input_12 = Input(shape=(512,))
input_22 = Input(shape=(512,))

# input1 = [input_11,input_12]
# input2 = [input_21,input_22]

input_final = [input_11,input_12, input_21,input_22]
left = final_model(input_final[:2])
right = final_model(input_final[2:])
both = merge([left,right], mode = euclidean_distance, output_shape=lambda x: x[0])
prediction = Activation('sigmoid')(both)
# print (prediction)


siamese_net = Model(input=input_final,output=prediction)
# siamese_net = Model(input=[input1,input2],output=both)
rms = RMSprop()
adam = Adam(lr=0.0001)
# siamese_net.summary()
siamese_net.compile(loss=contrastive_loss, optimizer=adam)
siamese_net.summary()

###############################Input Loading###########################




def load_data(datapath, show_id):
    shows_c3d = join(datapath, str(show_id) + '/' + str(show_id) + '-c3d-pool5.npy')
    shows_inception = join(datapath, str(show_id) + '/' + str(show_id) + '-inception-pool3.npy')
    c3d_data = np.load(shows_c3d)
    i_data = np.load(shows_inception)
    return c3d_data, i_data

print ("Loading Data...............")

# datapath = "../shows/track_1_shows/feature/"

train_X = read_csv_file('/track_1_shows/split/train.csv')[0]


train_X_c3d_features = []
train_X_inception_features = []
for i in range(7536):
    # print (i)
    c3d_feature, i_feature = load_data(datapath, i)
    c3d_feature, i_feature = np.asarray(c3d_feature), np.asarray(i_feature)
    train_X_c3d_features.append(c3d_feature)
    train_X_inception_features.append(i_feature)

print ("Loading Ground Truth...............")

relevance_file = pd.read_csv('./relevance_train.csv')
relevance_file.fillna(-1, inplace=True)

ground_truth_train = {}

for index, row in relevance_file.iterrows():    
    k = 0
    for j in row:
        if(k==0):
            ground_truth_train[int(j)] = []
            k+=1
            continue
        if(j==-1.0):break
        ground_truth_train[int(row["Ind"])].append(int(j))

print ("Loading Data Complete...............")
    

def make_pairs(train_X):
    pairs = []
    truth = []
    for i in train_X:
        # k = 0
        for j in ground_truth_train[i]:
            truth += [0.0]
            pair = [train_X_inception_features[i], train_X_c3d_features[i], train_X_inception_features[j], train_X_c3d_features[j]]
            pairs+=[pair]
            for k in range(j+1,7536):
                if(k not in ground_truth_train[i] and k!=i):
                    truth+=[1.0]
                    pair = [train_X_inception_features[i], train_X_c3d_features[i], train_X_inception_features[k], train_X_c3d_features[k]]
                    pairs+=[pair]

                    break
    return np.array(pairs), np.array(truth)


def make_pairs_old(train_X):
    pairs = []
    truth = []
    for i in train_X:
        k = 0
        for j in range(7536):
            if(i==j): continue
            
            if(j not in ground_truth_train[i]):
                if(k>30):
                	continue
                truth+=[1.0]
                k+=1
            else:
                # val = (ground_truth_train[i].index(j)/(len(ground_truth_train[i])*1.0))
                # truth += [val]
                truth += [0.0]

            pair = [train_X_inception_features[i],train_X_c3d_features[i],train_X_inception_features[j],train_X_c3d_features[j]]
            pairs+=[pair]
            # if(j not in ground_truth_train[i]):
            #     truth+=[0]
            # else: 
            #     val = 1/(ground_truth_train[i].index(j) + 1)
            #     truth += [val]
            
            
    return np.array(pairs), np.array(truth)

def make_pairs_pred(train_X):
    pairs = []
    truth = []
    for i in train_X:
        k = 0
        for j in range(7536):
            # if(i==j): continue
            
            # if(j not in ground_truth_train[i]):
            #     if(k>50):
            #     	continue
            #     truth+=[1.0]
            # 	k+=1
            # else:
            #     val = (ground_truth_train[i].index(j)/(len(ground_truth_train[i])*1.0))
            #     truth += [val]

            pair = [train_X_inception_features[i],train_X_c3d_features[i],train_X_inception_features[j],train_X_c3d_features[j]]
            pairs+=[pair]
            # if(j not in ground_truth_train[i]):
            #     truth+=[0]
            # else: 
            #     val = 1/(ground_truth_train[i].index(j) + 1)
            #     truth += [val]
            
        break        
    return np.array(pairs)


iterations = 1000
epoch = 20
print ("Make Pairs....")
from random import shuffle
tr_pairs, tr_true = make_pairs(train_X)
# pred_pairs = make_pairs_pred(train_X)

print ("Starting fitting..")
# siamese_net.summary()
print (len(tr_pairs))

siamese_net.load_weights('./3_Weights/Full/shows10_adam_100.h5')
for j in range(10,15):
    count = 0
    loss_this_iteration = 0
    for i in range(len(tr_pairs)):
        if(i==92884 or i==91862 or i==90402 or i==87918 or i==87798 or i==84362 or i==84360 or i==82440 or i==81674 or i==80456 or i==79210 or i==77868 or i==77842 or i==77544 or i==77542 or i==76546 or i==76544 or i==74678 or i==74590 or i==71350 or i== 71856 or i==71192 or i==69782 or i==70050 or i==65958 or i==68092 or i==59742 or i==64190 or i==60156 or i==57482 or i==59586 or i==57190 or i==57280 or i==56390 or i==57164 or i==55606 or i==55332  or i== 55604 or i ==52388 or i==51068 or i== 49388 or i==27540 or i== 34734 or i==27794 or i==36730 or i==41350 or i==41376 or i==41388 or i==45068 or i==45070 or i==45114  or i==46152): continue

        if(i== 17152 or i==26892 or i == 24238 or i==27118 or i== 27476 or i== 17153 or i==19762 or i== 19764 or i==0 or i==68 or i==822 or i==3968 or i==3972 or i==6258 or i==6774 or i==6778 or i==8840 or i==10962 or i==12322 or i==13994 or i ==13998 or i==16046 or i==16048 or i==16678 or i==16711 or i==16710):continue
        # if(i==0 or i==68 or i==822 or i==3968 or i==3972 or i==6258 or i==6774 or i==6778 or i==8840 or i==10962 or i==12322 or i==13994 or i ==13998 or i==16046):continue
        # if(i==16048):break
        # if(i==44 or i==81 or i==989 or i==5279 or i==5284 or i==8679 or i==9467 or i==9472 or i==12549 or i==15596 or i==17469): continue
        tr_pairs[i][0] = np.reshape(tr_pairs[i][0], (1,-1,2048))
        tr_pairs[i][1] = np.reshape(tr_pairs[i][1], (1,512))
        tr_pairs[i][2] = np.reshape(tr_pairs[i][2], (1,-1,2048))
        tr_pairs[i][3] = np.reshape(tr_pairs[i][3], (1,512))
        count+=1
        val = siamese_net.train_on_batch([tr_pairs[i][0],tr_pairs[i][1], tr_pairs[i][2], tr_pairs[i][3]], np.asarray([tr_true[i]]))
        if(i%5000==0):
        	print (i, val, tr_true[i], j)
        if(np.isnan(val)): break
        loss_this_iteration +=val
        if(i%50==0 and i%100!=0):
            siamese_net.save_weights('./3_Weights/Full/shows15_adam_50.h5')
        elif(i%100==0):
            siamese_net.save_weights('./3_Weights/Full/shows15_adam_100.h5')
    siamese_net.save_weights('./3_Weights/Full/shows15_adam_100.h5')
    print(loss_this_iteration/(count))
    print(loss_this_iteration)



# Fixed
# 0.47284602365474293 - 6
# 0.4675804394176138 - 10
# 0.46243622323789246 - 19


# 256 dense layer
# 0.45917047495904134 - 20


# 0.5034885689081862
# 0.4872786486314609
# 0.4273606743554351
# 0.4167293273781442
# 0.4120519517173832 - 18
# 0.4100513023791192 - 20

# siamese_net.fit([tr_pairs[:,0],tr_pairs[:,1],tr_pairs[:,2],tr_pairs[:,3]], tr_true, batch_size=1, nb_epoch=epoch)
