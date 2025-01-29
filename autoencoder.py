import mlflow
import mlflow.keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Lambda
import keras.backend as K
import numpy as np

# Function for Random Masking
def random_mask(x, mask_fraction=0.2):
    """
    Randomly mask a fraction of the input data by setting it to zero.
    
    :param x: Input tensor
    :param mask_fraction: Fraction of the input to mask (default 20%).
    :return: Masked input tensor
    """
    mask = K.cast(K.greater(K.random_uniform(K.shape(x)), mask_fraction), K.floatx())
    return x * mask  # Element-wise multiplication to mask

# Function to create the encoder
def create_encoder(input_shape, layer_sizes, mask_fraction=0.2):
    encoder = Sequential()
    encoder.add(Lambda(random_mask, output_shape=input_shape, arguments={'mask_fraction': mask_fraction}))  
    for size in layer_sizes:
        encoder.add(Dense(size, activation='relu'))
        encoder.add(Dropout(0.3))  # Dropout for regularization
    return encoder

# Function to create the decoder
def create_decoder(latent_size, layer_sizes, input_shape):
    decoder = Sequential()
    decoder.add(Dense(layer_sizes[0], activation='relu', input_shape=(latent_size,)))
    for size in layer_sizes[1:]:
        decoder.add(Dense(size, activation='relu'))
        decoder.add(Dropout(0.3))  # Dropout for regularization
    decoder.add(Dense(input_shape[0], activation='sigmoid'))
    return decoder

# Function to build the full autoencoder model
def build_autoencoder(input_shape, encoder_layer_sizes, decoder_layer_sizes, mask_fraction=0.2):
    encoder = create_encoder(input_shape, encoder_layer_sizes, mask_fraction)
    latent_size = encoder_layer_sizes[-1]  # Bottleneck size
    decoder = create_decoder(latent_size, decoder_layer_sizes, input_shape)
    input_img = Input(shape=input_shape)
    encoded_repr = encoder(input_img)
    decoded_output = decoder(encoded_repr)
    autoencoder = Model(input_img, decoded_output)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    return autoencoder, encoder, decoder

# Example usage
input_shape = (10,)
encoder_layer_sizes = [5, 5, 3]
decoder_layer_sizes = [5, 5, 10]

# Build the model
autoencoder, encoder, decoder = build_autoencoder(input_shape, encoder_layer_sizes, decoder_layer_sizes)

# Start an MLFlow experiment
mlflow.start_run()

# Log model hyperparameters (architecture details)
mlflow.log_param("input_shape", input_shape)
mlflow.log_param("encoder_layer_sizes", encoder_layer_sizes)
mlflow.log_param("decoder_layer_sizes", decoder_layer_sizes)

# Log the model architecture and parameters
autoencoder.summary()

# Generate synthetic data (replace with your actual dataset)
x_train = np.random.rand(1000, 10)
x_test = np.random.rand(200, 10)

# Log training data size
mlflow.log_param("train_data_size", len(x_train))
mlflow.log_param("test_data_size", len(x_test))

# Train the model
history = autoencoder.fit(
    x_train, 
    x_train,  # Autoencoder is unsupervised (input and target are the same)
    epochs=50, 
    batch_size=32, 
    validation_data=(x_test, x_test)
)

# Log metrics (e.g., final loss)
mlflow.log_metric("final_loss", history.history['loss'][-1])

# Log the trained Keras model
mlflow.keras.log_model(autoencoder, "autoencoder_model")

# End the MLFlow run
mlflow.end_run()

# Optionally, print a link to the experiment if you're using MLFlow UI
print("Run logged. You can view the results in the MLFlow UI.")
