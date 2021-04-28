from tensorflow import keras
from tensorflow.keras.layers import Input, LSTM, Dense
import numpy as np

# Define an input sequence and process it.
num_encoder_tokens = None
num_decoder_tokens = None
latent_dim = 300
encoder_input_data = None
decoder_input_data = None
decoder_target_data = None
batch_size = 100
epochs = 50

# define an input sequence and process it
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# we discard "encoder_outputs" and only keep the states
encoder_states = [state_h, state_c]


# Set up the decoder, using ""
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# we setup our decoder to return full output sequences,
# and to internal return states as well. we don't use the
# return states in the training model, but we'll use them in infrnce
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

# define the model that will turn 
# "encoder_input_data" & "decoder_input_data" into
# "decoder_target data"
model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer="rmsprop", loss="categorical_crossentropy")
model.fit(
	[encoder_input_data, decoder_input_data], decoder_target_data,
	batch_size=batch_size,
	epochs=epochs,
	validation_split=0.2
)


"""
After one hour or so on a macbook cpu, we are ready for inference. 
"""
# Here's our inference setup

encoder_model = keras.Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
	decoder_inputs, initial_state=decoder_states_inputs
)
decoder_states = [state_h, state_c]
decoder_model = keras.Model(
	[decoder_inputs] + decoder_states_inputs,
	[decoder_outputs] + decoder_states
)


target_token_index = []
reverse_target_char_index = []
max_decoder_seq_length = 20

# we use it to implement the inference loop described above
def decode_sequence(input_seq):
	# encode the input as state vectors.
	states_value = encoder_model.predict(input_seq)

	# Generate empty target sequence of length 1
	target_seq = np.zeros((1, 1, num_decoder_tokens))
	# populate the first charector of target_seq with start char 
	target_seq[0, 0, target_token_index["\t"]] = 1

	# sampling loop for a batch of sequences
	# (to simplify, here we assume a batch of size 1)
	stop_condition = False
	decoded_sentence = ""
	while not stop_condition:
		output_tokens, h, c = decoder_model.predict(
			[target_seq] + states_value)

		# Sample a token
		sampled_token_index = np.argmax(output_tokens[0, -1, :])
		sampled_char = reverse_target_char_index[sampled_token_index]
		decoded_sentence += sampled_char

		# Exit condition: either hit max length
		# or find stop charector.
		if (sampled_char == "\n" or
			len(decoded_sentence) > max_decoder_seq_length):
			stop_condition = True

		
		# update the target sequence (of length 1)
		target_seq = np.zeros((1, 1, num_decoder_tokens))
		target_seq[0, 0, sampled_token_index] = 1

		# update states
		states_values = [h, c]
	
	return decoded_sentence













