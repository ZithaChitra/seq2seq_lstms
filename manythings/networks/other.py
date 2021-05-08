from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from manythings.data.dta_ManyThings import ManyThings

class Seq2Seq():
	def __init__(
		self,num_encoder_tokens: int=64,
		num_decoder_tokens: int=64,
		latent_dim: int=300,
		training=True
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



data = ManyThings()
data.load_or_generate()
data.preprocess()
encoder_input_data = data.encoder_input_data
decoder_input_data = data.decoder_input_data
decoder_target_data = data.decoder_target_data

batch_size = 64  # Batch size for training.
epochs = 100

model = Seq2Seq()
model.training_model.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)
model.training_model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
)


# model.create_inference_models()
# enc_model = model.encoder_model
# enc_model.summary()
# plot_model(enc_model, show_shapes=True)




