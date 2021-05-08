from tensorflow import keras
from tensorflow.keras import Model


def nn_lstm1(
	num_encoder_tokens: int,
	num_decoder_tokens: int,
	latent_dim: int=300,
	training=True
)->Model:

	encoder_inputs = keras.Input(shape=(None, num_encoder_tokens))
	encoder = keras.layers.LSTM(latent_dim, return_state=True)
	decoder_inputs = keras.Input(shape=(None, num_decoder_tokens))
	decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
	decoder_dense = keras.layers.Dense(num_decoder_tokens, activation="softmax")

	if training:
		# Define an input sequence and process it.
		encoder_outputs, state_h, state_c = encoder(encoder_inputs)
		# We discard `encoder_outputs` and only keep the states.
		encoder_states = [state_h, state_c]

		# Set up the decoder, using `encoder_states` as initial state.
		# We set up our decoder to return full output sequences,
		# and to return internal states as well. We don't use the
		# return states in the training model, but we will use them in inference.
		decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
		decoder_outputs = decoder_dense(decoder_outputs)

		# Define the model that will turn
		# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
		model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

		return model

	else:
		encoder_model = Model(encoder_inputs, encoder_states)

		decoder_state_input_h = keras.Input(shape=(latent_dim,))
		decoder_state_input_c = keras.Input(shape=(latent_dim,))
		decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

		decoder_outputs, state_h, state_c = decoder_lstm(
  								decoder_inputs, initial_state=decoder_states_inputs)
		decoder_states = [state_h, state_c]
		decoder_outputs = decoder_dense(decoder_outputs)

		decoder_model = Model(
  						[decoder_inputs] + decoder_states_inputs,
  						[decoder_outputs] + decoder_states)