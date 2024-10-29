import os

import numpy as np
import tensorflow as tf
import glob
import kagglehub
from model import initialize_model, BUCKET_COUNT
from tensorflow.keras import backend as K


# Get decoder input by appending an empty start vector and removing the stop vector
def get_decoder_input(expected_output):
    empty_array = np.zeros((1, 129), dtype=np.int32)
    return np.vstack((empty_array[:-1], expected_output))


def prepare_training_data(x, y):
    # Load spectrogram and give it an extra dimension
    encoder_input = tf.numpy_function(np.load, [x], tf.float32)
    encoder_input = tf.expand_dims(encoder_input, axis=-1)
    # Load note map and offset it for the decoder
    expected_output = tf.numpy_function(np.load, [y], tf.int32)
    decoder_input = tf.numpy_function(get_decoder_input, [expected_output], tf.int32)
    return (encoder_input, decoder_input), expected_output


DATASET_PATH = "processed_musicnet"
EPOCHS = 10

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth for the GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
K.clear_session()

# Download dataset
path = kagglehub.dataset_download("imsparsh/musicnet-dataset")

# Get and pair together paths of sound files and labels
train_paths = (sorted(glob.glob(f"{DATASET_PATH}/train_data/*.npy")),
               sorted(glob.glob(f"{DATASET_PATH}/train_labels/*.npy")))

# Create dataset from list of paths
train_dataset = tf.data.Dataset.from_tensor_slices(train_paths)
train_dataset = train_dataset.map(prepare_training_data)

encoder, decoder, model = initialize_model()
# TODO: Handle validation and training
model.fit(train_dataset.batch(1), epochs=EPOCHS)
