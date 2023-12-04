import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, Reshape, Dense, Add, Conv2DTranspose, Activation
from tensorflow.keras.models import Model
import pdb
class AutoEncoder:

    def __init__(self, input_shape, latent_dim):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.num_channel = 3
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    # Define a residual block
    def enc_residual_block(self, x, filters, kernel_size=3, strides=2):
        conv = Conv2D(filters, kernel_size, strides=strides, padding='same', activation='relu')(x)
        skip = Conv2D(filters, 1, strides=strides, padding='same')(x)
        add = Add()([conv, skip])
        output = Activation('relu')(add)
        return output

    def dec_residual_block(self, x, filters, kernel_size=3, strides=2):

        # pdb.set_trace()
        convT = Conv2DTranspose(filters, kernel_size, strides=strides, padding='same')(x)
        skip = Conv2DTranspose(filters, 1, strides=strides, padding='same')(x)
        add = Add()([convT, skip])
        output = Activation('relu')(add)
        return output
    
# Define the autoencoder with residual blocks

    def build_encoder(self):
        # Encoder
        enc_input = Input(shape=self.input_shape)

        conv1 = self.enc_residual_block(enc_input, filters=32)
        conv2 = self.enc_residual_block(conv1, filters=64)

        flatten = Flatten()(conv2)
        encoded = Dense(self.latent_dim, activation='relu')(flatten)

        return Model(enc_input, encoded)

    def build_decoder(self):
        # Decoder
        dec_input = Input(self.latent_dim)
        

        dense = Dense(self.input_shape[0]//4 * self.input_shape[1]//4 * 64, 'relu') (dec_input)
        reshape = Reshape((self.input_shape[0]//4, self.input_shape[1]//4, 64))(dense)

        # pdb.set_trace()
        conv1 = self.dec_residual_block(reshape, filters=64)
        conv2 = self.dec_residual_block(conv1, filters=32)
        # pdb.set_trace()
        decoded = Conv2D(self.num_channel, (3, 3), activation='sigmoid', padding='same')(conv2)
        
        # pdb.set_trace()
    
        return Model(dec_input, decoded, name="Decoder")

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
