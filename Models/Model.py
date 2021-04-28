""" 
A model is a combination of the neural net and the
data used to train it.
"""

from pathlib import Path
from typing import Callable, Dict
from tensorflow import keras as KerasModel
from util_yaml import yaml_dump, yaml_loader
from tensorflow.keras.optimizers import SGD


DIRNAME = Path(__file__).parents[1].resolve() / "weights"


class Model():
	def __init__(
		self,
		dataset_cls: type,
		network_fn: Callable[..., KerasModel],
		net_args: Dict,
		net_io_shapes: Dict,
		dataset_args: Dict = None,
	):
		self.name = f"{self.__class__.__name__}_{dataset_cls.__name__}_{network_fn.__name__}"

		if dataset_args == None:
			dataset_args = {}
		self.data = dataset_cls(**dataset_args)
		self.data.load_or_generate()
		self.data.preprocess()


		
		self.net_args = net_args
		self.network = network_fn(net_args, net_io_shapes)

	
	@property
	def weights_filename(self)->str:
		DIRNAME.mkdir(parent=True, exist_ok=True)
		return str(DIRNAME / f"{self.name}_weights.h5")

	
	def fit(
		self,
		callbacks
	):
		config = self.net_args

		sgd = SGD(lr=config["learn_rate"], decay=config["decay"], momentum=config["momentum"],
          nesterov=True)

		# if callbacks in None:
		# 	callbacks = []

		self.network.compile(
			loss="categorical_crossentropy",
			optimizer=sgd,
			metrics=["accuracy"]
		)

		self.network.fit(
			[self.data.encoder_input_data, self.data.decoder_input_data],
			self.data.decoder_target_data,
			validation_split=0.2,
			epochs = config["epochs"],
			callbacks=callbacks
		)























