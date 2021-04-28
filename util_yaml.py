import yaml

# data = yaml.load(file_descriptor)
# yaml.dump(data)

def yaml_loader(filepath):
    """ Loads data from a yaml file """
    with open(filepath, "r") as file_descriptor:
        data = yaml.load(file_descriptor, Loader=yaml.FullLoader)
        return data


def yaml_dump(filepath, data):
    """ Writes data to a yaml file """
    with open(filepath, "w") as file_descriptor:
        yaml.dump(data, file_descriptor)


if __name__ == "__main__":
	
	data = {
		"project_name": "cnn_mnist",
		"dataset": {
			"name": "Mnist",
			"dataset_args": {
				"img_width": None,
				"img_height": None
			}
		},
		"network": {
			"name": "nn_conv1",
			"net_args": {
				"decay": 1.0e-06,
				"dropout": 0.2,
				"epochs": 20,
				"hidden-layer-size": 128,
				"layer-1-size": 16,
				"layer-2-size": 32,
				"learn-rate": 0.01,
				"momentum": 0.9
			},
			"io_shapes": {
				"input_shape": None,
				"output_shape": None
			}
		}
	}

	yaml_dump("experiment.yaml", data)
	print("Done!")

