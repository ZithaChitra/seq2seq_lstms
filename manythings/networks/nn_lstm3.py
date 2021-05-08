import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model


def nn_lstm3(
	num_encoder_tokens: int=64,
	num_decoder_tokens: int=64,
	latent_dim: int=300,
	training=True
):
	encoder_input = layers.Input(shape=(None, num_encoder_tokens), name="encoder_input")
	encoder_lstm = layers.LSTM(latent_dim, return_state=True, name="encoder_lstm")
	_, state_h, state_c = encoder_lstm(encoder_input)
	encoder_state = [state_h, state_c]

	decoder_input = layers.Input(shape=(None, num_decoder_tokens), name="decoder_input")
	decoder_lstm = layers.LSTM(latent_dim, return_sequences=True, return_state=True, name="decoder_lstm")
	decoder_lstm_output, _, _ = decoder_lstm(decoder_input, initial_state=encoder_state)

	decoder_dense = layers.Dense(num_decoder_tokens, activation="softmax", name="decoder_dense")
	dense_output = decoder_dense(decoder_lstm_output)

	if training:
		model = keras.Model([encoder_input, decoder_input], dense_output)
		return model

	else:
		encoder_model = keras.Model(encoder_input, encoder_state)


		decoder_state_input_h = layers.Input(shape=(latent_dim,), name="decoder_state_input_h")
		decoder_state_input_c = layers.Input(shape=(latent_dim,), name="decoder_state_input_c")
		decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

		decoder_outputs, state_h, state_c = decoder_lstm(
							decoder_input, initial_state=decoder_states_inputs)
		decoder_states = [state_h, state_c]
		decoder_output = decoder_dense(decoder_outputs)

		decoder_model = keras.Model([decoder_input] + decoder_states_inputs, [decoder_output] + decoder_states)

		return [encoder_model, decoder_model]




# model = nn_lstm3()
encoder_model, decoder_model = nn_lstm3(training=False)
encoder_model.summary()
plot_model(encoder_model, show_shapes=True)
decoder_model.summary()
plot_model(decoder_model, show_shapes=True)

	