import librosa
from scipy import signal
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras import backend as K

path='audio/emergency.wav'
emergency,sample_rate = librosa.load(path, sr = 16000)
path='audio/non emergency.wav'
non_emergency,sample_rate = librosa.load(path, sr = 16000)

duration1 = librosa.get_duration(y = emergency, sr = 16000)
duration2 = librosa.get_duration(y = non_emergency, sr = 16000)

print(duration1, duration2)

def prepare_data(samples, num_of_samples = 32000, num_of_common = 16000):
    data = []
    for offset in range(0, len(samples), num_of_common):
        chunk = samples[offset: offset + num_of_samples]
        if len(chunk) == 32000:
            data.append(chunk)
    return data

emergency = prepare_data(emergency)
non_emergency = prepare_data(non_emergency)

print(len(emergency), len(non_emergency))

plt.figure(figsize=(14,4))
plt.plot(np.linspace(0, 2, num=32000),emergency[103])
plt.title('Emergency')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.figure(figsize=(14,4))
plt.plot(np.linspace(0, 2, num=32000),non_emergency[102])
plt.title('Non Emergency')
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.show()

audio = np.concatenate([emergency,non_emergency])
labels1 = np.zeros(len(emergency))
labels2 = np.ones(len(non_emergency))
labels = np.concatenate([labels1,labels2])
print(audio.shape)

x_tr, x_val, y_tr, y_val = train_test_split(np.array(audio),np.array(labels), stratify=labels,test_size = 0.1, random_state=777,shuffle=True)

x_tr_features  = x_tr.reshape(len(x_tr),-1,1)
x_val_features = x_val.reshape(len(x_val),-1,1)
print("Reshaped Array Size",x_tr_features.shape)

def cnn(x_tr):
  K.clear_session()
  inputs = Input(shape=(x_tr.shape[1],x_tr.shape[2]))
  #First Conv1D layer
  conv = Conv1D(8, 13, padding='same', activation='relu')(inputs)
  conv = Dropout(0.3)(conv)
  conv = MaxPooling1D(2)(conv)
  #Second Conv1D layer
  conv = Conv1D(16, 11, padding='same', activation='relu')(conv)
  conv = Dropout(0.3)(conv)
  conv = MaxPooling1D(2)(conv)
  #MaxPooling 1D
  conv = GlobalMaxPool1D()(conv)
  #Dense Layer 
  conv = Dense(16, activation='relu')(conv)
  outputs = Dense(1,activation='sigmoid')(conv)
  model = Model(inputs, outputs)
  model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
  model_checkpoint = ModelCheckpoint('best_model.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
  return model, model_checkpoint
model, model_checkpoint = cnn(x_tr_features)

model.summary()

history=model.fit(x_tr_features, y_tr ,epochs=10,       callbacks=[model_checkpoint], 
batch_size=32, validation_data=(x_val_features,y_val))

model.load_weights('best_model.hdf5')

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.xlabel('Time')

plt.ylabel('epoch')

plt.legend(['train','validation'],loc = 'upper left')

plt.show()

_, acc = model.evaluate(x_val_features,y_val)
print("Validation Accuracy:",acc)

ind=35
test_audio = x_val[ind]
ipd.Audio(test_audio,rate=16000)
feature = x_val_features[ind]
prob = model.predict(feature.reshape(1,-1,1))
if (prob[0][0] < 0.5 ):
 pred='emergency'
else:
 pred='non emergency'
print("Prediction:", pred)