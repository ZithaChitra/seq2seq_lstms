from tensorflow import keras
import pathlib
from manythings.data.dta_ManyThings import ManyThings
from manythings.networks.nn_lstm import Seq2Seq
import os
import wandb
from wandb.keras import WandbCallback
import click


@click.command()
@click.option("--latent_dim", default=250, help="latent dim of lstm layer")
def train_model(
	latent_dim,
	project_name: str="manythings",
	use_wandb: bool=True
):
	# A simple example of how to combine the data
	# and the lstm neural net to create a model

	path = pathlib.Path.cwd() / "saved_models" / "weights"
	# model_path = os.path.join(path, "lstm.h5")
	model_weights_path = os.path.join(path, "nn_lstm_weights.hdf5")

	# The ManyThings class creates an easy to use interface
	# to the dataset 
	data = ManyThings()
	data.load_or_generate()
	data.preprocess()
	encoder_input_data = data.encoder_input_data
	decoder_input_data = data.decoder_input_data
	decoder_target_data = data.decoder_target_data
	num_encoder_tokens = data.num_encoder_tokens
	num_decoder_tokens = data.num_decoder_tokens
	input_texts = data.input_texts
	

	batch_size = 100  # Batch size for training.
	epochs = 2

	model = Seq2Seq(
		num_encoder_tokens=num_encoder_tokens, num_decoder_tokens=num_decoder_tokens,
		latent_dim=latent_dim
	)
	model.training_model.compile(
		optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
	)

	checkpoint_callback = keras.callbacks.ModelCheckpoint(
		model_weights_path, monitor="val_loss", save_best_only=True
	)

	if use_wandb:
		with wandb.init(project=project_name) as run:
			model.training_model.fit(
				[encoder_input_data, decoder_input_data],
				decoder_target_data,
				batch_size=batch_size,
				epochs=epochs,
				validation_split=0.2,
				callbacks=[checkpoint_callback, WandbCallback()],
			)
			model.training_model.save(model_weights_path)
			model_artifact = wandb.Artifact("Seq2Seq", type="model")
			model_artifact.add_file(model_weights_path)
			run.log_artifact(model_artifact)

	else:
		model.training_model.fit(
		[encoder_input_data, decoder_input_data],
		decoder_target_data,
		batch_size=batch_size,
		epochs=epochs,
		validation_split=0.2,
		callbacks=[checkpoint_callback, WandbCallback()])
		model.training_model.save(model_weights_path)

	
if __name__ == "__main__":
	train_model()