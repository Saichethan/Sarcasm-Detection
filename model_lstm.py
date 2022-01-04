import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, GlobalMaxPooling1D, MaxPooling1D, Dropout, Embedding, LeakyReLU, Flatten, GlobalAveragePooling1D, Dropout, TimeDistributed, GlobalMaxPool1D
from keras.layers import Bidirectional, GRU, Masking, TimeDistributed, Lambda, Activation, dot, multiply, concatenate, LSTM, ReLU
from keras.layers.merge import concatenate
from keras.models import load_model
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K
import os
import csv
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report,  confusion_matrix
import pandas as pd
from keras_self_attention import SeqSelfAttention


from load import load_data


# Specify which GPU(s) to use
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Or 2, 3, etc. other than 0

# On CPU/GPU placement
config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True
tf.compat.v1.Session(config=config)


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


print("Loading data.....")
#albert, elmo, xlnet, y  = load_data()
albert, bert, elmo, ft, w2v, xlnet, y   = load_data()
print("Data Loading Completed")

samples = len(y)

albert = albert.reshape(samples,1,1024)
bert = bert.reshape(samples,1,1024)
elmo = elmo.reshape(samples,1,1024)
ft = ft.reshape(samples,1,300)
w2v = w2v.reshape(samples,1,300)
xlnet = xlnet.reshape(samples,1,1024)


############# change embeddings here ####################
X_train, X_temp, y_train, y_temp = train_test_split(xlnet, y, test_size=0.2)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)
########################################################

embeddings = Input(shape = (1, 1024), name="xlnet")
#embeddings = Input(shape = (1, 300), name="w2v")

gru_units = 300


p = Bidirectional(GRU(gru_units, return_sequences=True, dropout=0.25, recurrent_dropout=0.25))(embeddings)
p = GlobalMaxPool1D()(p)
p = Dense(100, activation="relu")(p)
p = Dropout(0.1)(p)


#merge = concatenate([x, y])

output = Dense(1, activation='sigmoid')(p) #(merge)

#model = Model(inputs=[embeddings1, embeddings2], outputs=output) 
model = Model(inputs=embeddings, outputs=output) 
model.summary()

#plot_model(model, to_file="model.png")

model.compile(optimizer="adam", loss="binary_crossentropy",  metrics=['accuracy']) #Adadelta

saved_model = "models/xlnet.h5"
checkpoint = ModelCheckpoint(saved_model, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=25)
callbacks_list = [checkpoint, early]


print("Fitting model...")
history = model.fit(X_train, y_train, 
	                    batch_size=500,
	                    epochs=25, callbacks=callbacks_list, validation_data=(X_val, y_val)) 



# load model
weights = load_model('models/xlnet.h5').get_weights() #, custom_objects={'loss': consensus(f1, f2, f3) }
model.set_weights(weights)
#model.summary()

probability = model.predict([X_test]) #
print("no problem")
print(probability.shape)
probability = probability.round()

probability = probability.tolist()


print("macro: ", f1_score(y_test, probability, average='macro'))

print("micro: ", f1_score(y_test, probability, average='micro'))

target_names = ['class 0', 'class 1']
print(classification_report(y_test, probability, target_names=target_names))

ff = open("results/xlnet.txt","w+")
ff.write(str(f1_score(y_test, probability, average='macro')))
ff.write("\n")
ff.write(str(f1_score(y_test, probability, average='micro')))
ff.write("\n")
ff.write(str(classification_report(y_test, probability, target_names=target_names)))
ff.close()


#plotting
import matplotlib.pyplot as plt
import seaborn as sn


matrix = confusion_matrix(y_test, probability)
sn.heatmap(matrix/np.sum(matrix), annot=True, 
            fmt='.2%')#, cmap='Blues')
plt.savefig("plots/xlnet_cm.pdf")


training_acc = history.history['acc']
test_acc = history.history['val_acc']

training_loss = history.history['loss']
test_loss = history.history['val_loss']

epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
f1 = plt.figure()
plt.plot(epoch_count, training_acc, 'r--')
plt.plot(epoch_count, test_acc, 'b-')
plt.legend(['Training accuracy', 'Validation accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
f1.savefig("plots/xlnet_accuracy.pdf", bbox_inches='tight')


f2 = plt.figure()
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training loss', 'Validation loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
f2.savefig("plots/xlnet_loss.pdf", bbox_inches='tight')



