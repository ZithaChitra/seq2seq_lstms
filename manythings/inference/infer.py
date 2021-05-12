import numpy as np
from manythings.data.dta_ManyThings import ManyThings
from manythings.networks.nn_lstm import Seq2Seq
import pathlib
import os


def decode_and_translate(
	input_seq, encoder_model, decoder_model,
	num_decoder_tokens, target_char_index,
	reverse_target_char_index, max_decoder_seq_length	
):
	encoder_model = encoder_model
	decoder_model = decoder_model

	# encode the input as state vectors
	states_values = encoder_model.predict(input_seq)

	# Generate empty target sequence of length 1
	target_seq = np.zeros((1, 1, num_decoder_tokens))
	
	# populate the first charector of the target sequence with the start charector
	target_seq[0, 0, target_char_index["\t"]] = 1.0

	# Sampling loop for a batch of sequences
	# (To simplify, here we assume a batch size of one)
	stop_condition = False
	decoded_sentence = ""
	while not stop_condition:
		output_tokens, state_h, state_c = decoder_model.predict([target_seq] + states_values)

		# Sample a token
		sampled_token_index = np.argmax(output_tokens[0, -1, :])
		sampled_char = reverse_target_char_index[sampled_token_index]
		decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
		if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
		    stop_condition = True

        # Update the target sequence (of length 1).
		target_seq = np.zeros((1, 1, num_decoder_tokens))
		target_seq[0, 0, sampled_token_index] = 1.0

        # Update states
		states_value = [state_h, state_c]

	return decoded_sentence


def translation_example(weights_name: str="nn_lstm_weights.hdf5"):

	weights = pathlib.Path.cwd() / "saved_models" / "weights"
	weights_path = os.path.join(weights, weights_name)

	data = ManyThings()
	data.load_or_generate()
	data.preprocess()
	encoder_input_data = data.encoder_input_data
	num_encoder_tokens = data.num_encoder_tokens
	num_decoder_tokens = data.num_decoder_tokens
	input_texts = data.input_texts


	model = Seq2Seq(
		num_encoder_tokens=num_encoder_tokens, num_decoder_tokens=num_decoder_tokens,
		model_weights=weights_path)


	encoder_model, decoder_model = model.create_inference_models()
	target_token_index = data.target_token_index
	reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())
	max_decoder_seq_length = data.max_decoder_seq_length


	for seq_index in range(3):
    	# Take one sequence (part of the training set)
    	# for trying out decoding.
		input_seq = encoder_input_data[seq_index : seq_index + 1]
		decoded_sentence = decode_and_translate(
								input_seq,
								encoder_model,
								decoder_model, num_decoder_tokens, target_token_index,
								reverse_target_char_index, max_decoder_seq_length)
		print("-")
		print("Input sentence:", input_texts[seq_index])
		print("Decoded sentence:", decoded_sentence)
	return


if __name__ == "__main__":
	translation_example()




