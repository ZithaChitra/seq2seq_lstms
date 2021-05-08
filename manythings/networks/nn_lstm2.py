import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model

class Seq2Seq(keras.Model):
	def __init__(
		self,
		num_encoder_tokens: int=64,
		num_decoder_tokens: int=64, 
		latent_dim: int=300, **kwargs
	):
		super(Seq2Seq, self).__init__(**kwargs)
		self.latent_dim = latent_dim
		self.encoder_input = layers.Input(shape=(None, num_encoder_tokens))
		self.encoder_lstm = layers.LSTM(latent_dim, return_state=True)

		self.decoder_input = layers.Input(shape=(None, num_decoder_tokens))
		self.decoder_lstm = layers.LSTM(latent_dim, return_sequences=True, return_state=True)
		self.decoder_dense = layers.Dense(latent_dim, activation="softmax")

	def call(self, inputs, training=None):
		encoder_inputs = self.encoder_input
		_, state_h, state_c = self.encoder_lstm(encoder_input)
		encoder_states = [state_h, state_c]

		decoder_inputs = self.decoder_input
		decoder_outputs, _, _ = self.decoder_lstm(decoder_input, initial_state=encoder_states)
		outputs = self.decoder_dense(decoder_outputs)
		if not training:
			model_1 = keras.Model(encoder_inputs, outputs)
			return model_1(inputs)

		else:
			encoder_model = keras.Model(encoder_inputs, encoder_states)

			decoder_state_input_h = layers.Input(shape=(self.latent_dim,))
			decoder_state_input_c = layers.Input(shape=(self.latent_dim,))
			decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

			decoder_outputs, state_h, state_c = self.decoder_lstm(
                        decoder_inputs, initial_state=decoder_states_inputs)
			decoder_states = [state_h, state_c]
			decoder_outputs = self.decoder_dense(decoder_outputs)

			decoder_model = Model(
                          [decoder_inputs] + decoder_states_inputs,
                          [decoder_outputs] + decoder_states)

			
			



encoder_input = tf.ones((1, 64))
decoder_input = tf.ones((1, 64))
data = [encoder_input, decoder_input]

model = Seq2Seq()
result = model(data)
model.summary()




