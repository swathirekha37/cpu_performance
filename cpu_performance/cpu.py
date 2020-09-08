import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

dataset_path = keras.utils.get_file("machine.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/cpu-performance/machine.data")
dataset_path

column_names = ['vendor_name','Model_name','MYCT','MMIN','MMAX','CACH','CHMIN','CHMAX','PRP','ERP']


dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=",", skipinitialspace=True)

dataset.tail()
dataset.isna().sum()
dataset = dataset.dropna()

'''dataset['vendor_name'] = dataset['vendor_name'].map({1: 'adviser',2: 'amdahl',3:'apollo',4:'basf',5:'bti',6: 'burroughs',7:'c.r.d',8:'cambex', 9:'cdc',10:'dec', 
       11:'dg', 12:'formation', 13:'four-phase', 14:'gould', 15:'honeywell', 16:'hp', 17:'ibm', 18: 'ipl', 19:'magnuson', 
       20:'microdata', 21:'nas', 22:'ncr',23: 'nixdorf', 24:'perkin-elmer', 25:'prime', 26:'siemens', 27:'sperry', 
       28:'sratus', 29:'wang'})
dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')
dataset.tail()'''

dataset.pop('vendor_name')
dataset.pop('Model_name')

train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

sns.pairplot(train_dataset[['MYCT','MMIN']], diag_kind="kde")

train_stats = train_dataset.describe()
train_stats.pop('CHMIN')
train_stats = train_stats.transpose()
train_stats

train_labels = train_dataset.pop('CHMIN')
test_labels = test_dataset.pop('CHMIN')

'''train_labels = train_dataset.pop('Model_name')
test_labels = test_dataset.pop('Model_name')
train_labels = train_dataset.pop('vendor_name')
test_labels = test_dataset.pop('vendor_name')
'''

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

def build_model():
  model = keras.Sequential([
    layers.Dense(20, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(20, activation='relu'),
    layers.Dense(1)
  ])
  optimizer = tf.keras.optimizers.RMSprop(0.00001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = build_model()

model.summary()
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
example_result

EPOCHS = 1000
dataset['PRP'].shape

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[tfdocs.modeling.EpochDots()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)

plotter.plot({'Basic': history}, metric = "mae")
plt.ylim([0, 10])
plt.ylabel('MAE [MPG]')

plotter.plot({'Basic': history}, metric = "mse")
plt.ylim([0, 20])
plt.ylabel('MSE [MPG^2]')

model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

early_history = model.fit(normed_train_data, train_labels, 
                    epochs=EPOCHS, validation_split = 0.2, verbose=0, 
                    callbacks=[early_stop, tfdocs.modeling.EpochDots()])


plotter.plot({'Early Stopping': early_history}, metric = "mae")
plt.ylim([0, 10])
plt.ylabel('MAE [MPG]')

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

test_predictions = model.predict(normed_test_data).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")



































