import tensorflow as tf
from tensorflow import keras
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import pathlib
import shutil
import tempfile
logdir = pathlib.Path(tempfile.mkdtemp())/"tensorboard_logs"
shutil.rmtree(logdir, ignore_errors=True)
gz= tf.keras.utils.get_file('HIGGZ.csv.gz', 'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz')
FEATURES=28
ds= tf.data.experimental.CsvDataset(gz,[float(),]*(FEATURES+1), compression_type="GZIP")
def pack_row(*row):
    label= row[0]
    features= tf.stack(row[1:],1)
    return features, label
packed_ds= ds.batch(10000).map(pack_row).unbatch()
for features, label in packed_ds.batch(1000).take(1):
    print(features[0])
    plt.hist(features.numpy().flatten(), bins=101)
N_VALIDATION= int(1e3)
N_TRAIN= int(1e4)
BUFFER_SIZE= int(1e4)
BATCH_SIZE=500
STEP_PER_EPOCH= N_TRAIN//BATCH_SIZE
validate_ds= packed_ds.take(N_VALIDATION).cache()
train_ds= packed_ds.skip(N_VALIDATION).take(N_TRAIN).cache()
lr_schedule= tf.keras.optimizers.schedules.InverseTimeDecay(
    0.001,
    decay_steps=STEP_PER_EPOCH*1000,
    decay_rate=1,
    staircase=False
)
def get_optimizer():
    return tf.keras.optimizers.Adam(lr_schedule)
step=np.linspace(0,100000)
lr=lr_schedule(step)
plt.figure(figsize=(8,6))
plt.plot(step/STEP_PER_EPOCH, lr)
plt.ylim([0,max(plt.ylim())])
plt.xlabel('Epoch')
_ = plt.ylabel('Learning Rate')
def get_callbacks(name):
    return [
        tfdocs.modeling.EpochDots(),
        tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=200),
        tf.keras.callbacks.TensorBoard(logdir/name),
    ]
def compile_and_fit(model, name, optimizer=None, max_epochs=10000):
    if optimizer is None:
        optimizer= get_optimizer()
    model.compile(optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.losses.BinaryCrossentropy(from_logits=True, name='binary_crossentropy'),'accuracy'
    ])
    model.summary()
    history=model.fit(
        train_ds,
        step_per_epoch=STEP_PER_EPOCH,
        validation_data=validate_ds,
        callbacks=get_callbacks(name),
        verbose=0,
    )
    return history
# Define a simple sequential model
def create_model():
  model = tf.keras.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10)
  ])

  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

  return model

# Create a basic model instance
model = create_model()

# Display the model's architecture
model.summary()
