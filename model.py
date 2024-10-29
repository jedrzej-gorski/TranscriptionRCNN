from keras.models import Model
from keras.layers import Input, LSTM, Dense, Conv2D, AveragePooling2D, Flatten, Reshape, LayerNormalization
from keras.utils import plot_model

# Amount of frequency buckets in spectrogram
BUCKET_COUNT = 1025
# Amount of possible notes + stop token
OUTPUT_DIM = 129
# Dimensionality of the encoded representation
LATENT_DIM = 128
CONV_FILTER_COUNT = 4

# TODO: Implement inference

def initialize_model():
    # The architecture is divided into an encoder and decoder
    # Encoder takes a spectrogram with a variable range of time and 1025 frequency buckets
    # The input is four-dimensional so that it is compatible with the Conv2D layer
    encoder_inputs = Input(shape=(None, BUCKET_COUNT, 1))

    # Apply convolution for more precise rhythm recognition
    # TODO: Reconsider if a kernel_size of (1, 2) wouldn't be better to avoid overfitting to instrument tone
    conv_layer = Conv2D(filters=CONV_FILTER_COUNT, kernel_size=(2, 2), padding="same", activation="relu")(
        encoder_inputs)

    # Lower the dimensionality by pooling
    pooling_layer = AveragePooling2D(pool_size=(2, 2))(conv_layer)
    # Reshape by concatenating values from each filter
    reshape_layer = Reshape((-1, CONV_FILTER_COUNT * (BUCKET_COUNT // 2)))(pooling_layer)
    normalized_layer = LayerNormalization()(reshape_layer)
    # Discard encoder_output. Save cell and hidden state
    _, state_h, state_c = LSTM(LATENT_DIM, return_state=True)(normalized_layer)

    encoder_states = [state_h, state_c]

    encoder = Model(encoder_inputs, encoder_states)

    # Define decoder
    # Input is a variable length sequence of arrays of 129 values. In training this is the correct note map shifted once
    decoder_inputs = Input(shape=(None, OUTPUT_DIM))

    # Decoder consists of a LSTM layer that is initialized with the encoder's state
    # Decoder is progressively fed the correct note map during training (teacher forcing)
    decoder_lstm, decoder_state_h, decoder_state_c = LSTM(LATENT_DIM, return_sequences=True, return_state=True)(
        decoder_inputs, initial_state=encoder_states)
    decoder_states = [decoder_state_h, decoder_state_c]
    # Use sigmoid function for output rather than softmax as classes are largely independent of each other
    decoder_outputs = Dense(OUTPUT_DIM, activation='sigmoid')(decoder_lstm)

    decoder = Model([decoder_inputs] + encoder_states, [decoder_outputs] + decoder_states)

    # Combine both to form model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics="binary_crossentropy")

    plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)
    return encoder, decoder, model
