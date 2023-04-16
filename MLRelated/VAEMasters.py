"""
VAE implementation for embedding of SDSS optical spectra for future classification
via Keras
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.losses import mse
from keras import backend as K
from progress.bar import FillingSquaresBar
import numpy as np
import csv
import psutil
from sklearn.model_selection import train_test_split


def sampling(args):
   """
   Reparameterization trick

   :param args: mean and log of variance of Q(z|X)
   :return: sampled latent vector
   """

   z_mean, z_log_var = args
   batch = K.shape(z_mean)[0]
   dim = K.int_shape(z_mean)[1]
   epsilon = K.random_normal(shape=(batch, dim))
   return z_mean + K.exp(0.5 * z_log_var) * epsilon


# setting network parameters
intermediate_dim = 512
batch_size = 128
latent_dim = 10
epochs = 100
print("#################################################################")
print("=================================================================")
print("#################################################################")
print("epochs: ", epochs)
print("intermediate dimensions: ", intermediate_dim)
print("latent dimensions: ", latent_dim)
print("batch size: ", batch_size)
print(psutil.virtual_memory())

# read in data matrix
FinalArray = []
bar = FillingSquaresBar('Reading data in', max=7)
for i in range(1, 8):
   try:
       filename = "D:\\DATA\\CARBON\\output_"+str(i)+".csv"
       with open(filename) as csv_file:
           csv_reader = csv.reader(csv_file)
           line_count = 0
           for row in csv_reader:
               if line_count == 0:
                   array_initial = np.array(row)
                   array_initial = array_initial.astype(float, copy=False)
                   line_count = line_count + 1
                   continue
               loop = row
               loop = np.array(loop)
               loop = loop.astype(float, copy=False)
               array_initial = np.vstack((array_initial, loop))
               line_count = line_count + 1
       subFinalArray = array_initial
       if i == 1:
           FinalArray = subFinalArray
       else:
           FinalArray = np.vstack((FinalArray, subFinalArray))
       bar.next()
   except FileNotFoundError as e:
       print("File not found")
       continue
bar.finish()
x_all = FinalArray
FinalArray = []

# split data into training and validation sets
x_train, x_test = train_test_split(x_all, test_size=.1, random_state=42)

# building encoder layers
original_dim = 3775
input_shape = (original_dim, )
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# defining reparametrization layer
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# building decoder layers
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x)

# creating encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()

# creating decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# creating VAE model by linking encoder outputs to decoder inputs
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

if __name__ == '__main__':
   models = (encoder, decoder)

   # defining loss functions
   reconstruction_loss = mse(inputs, outputs)
   kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
   kl_loss = K.sum(kl_loss, axis=-1)
   kl_loss *= -0.5

   # adding loss function to VAE and compiling the network
   vae_loss = K.mean(reconstruction_loss + kl_loss)
   vae.add_loss(vae_loss)
   vae.compile(optimizer='adam')
   vae.summary()

   # training network
   vae.fit(x_train,
           epochs=epochs,
           batch_size=batch_size,
           shuffle=True,
           validation_data=(x_test, None))

   # trained models can be saved for later testing
   # or to resume training
   encoder.save('encoder.h5')
   decoder.save('decoder.h5')
   vae.save('vae.h5')

