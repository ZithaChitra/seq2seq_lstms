from tensorflow import keras
from tensorflow.keras import layers


class Seq2Seq():
	def __init__(
		self,num_encoder_tokens: int=64,
		num_decoder_tokens: int=64,
		latent_dim: int=300,
		model_weights: str=None
	):
		self.latent_dim = latent_dim
		self.num_decoder_tokens = num_decoder_tokens
		self.num_encoder_tokens = num_encoder_tokens
		self.encoder_input = layers.Input(shape=(None, self.num_encoder_tokens), name="encoder_input")
		self.encoder_lstm = layers.LSTM(self.latent_dim, return_state=True, name="encoder_lstm")
		_, state_h, state_c = self.encoder_lstm(self.encoder_input)
		self.encoder_state = [state_h, state_c]

		self.decoder_input = layers.Input(shape=(None, self.num_decoder_tokens), name="decoder_input")
		self.decoder_lstm = layers.LSTM(self.latent_dim, return_sequences=True, return_state=True, name="decoder_lstm")
		decoder_lstm_output, _, _ = self.decoder_lstm(self.decoder_input, initial_state=self.encoder_state)

		self.decoder_dense = layers.Dense(num_decoder_tokens, activation="softmax", name="decoder_dense")
		dense_output = self.decoder_dense(decoder_lstm_output)

		self.training_model = keras.Model([self.encoder_input, self.decoder_input], dense_output)
		if model_weights:
			self.training_model.load_weights(model_weights)

		self.encoder_model, self.decoder_model = self.create_inference_models()

		


	def create_inference_models(self):
		encoder_model = keras.Model(self.encoder_input, self.encoder_state, name="encoder_model")

		self.decoder_state_input_h = layers.Input(shape=(self.latent_dim,), name="decoder_state_input_h")
		self.decoder_state_input_c = layers.Input(shape=(self.latent_dim,), name="decoder_state_input_c")
		decoder_states_inputs = [self.decoder_state_input_h, self.decoder_state_input_c]
			
		decoder_outputs, state_h, state_c = self.decoder_lstm(
						self.decoder_input, initial_state=decoder_states_inputs)
		self.decoder_states = [state_h, state_c]
		decoder_output = self.decoder_dense(decoder_outputs)

		decoder_model = keras.Model(
								[self.decoder_input] + decoder_states_inputs,
								[decoder_output] + self.decoder_states,
								name="decoder_model")

		return [encoder_model, decoder_model]






# model.create_inference_models()
# enc_model = model.encoder_model
# enc_model.summary()
# plot_model(enc_model, show_shapes=True)




