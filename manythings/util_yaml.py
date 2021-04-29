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
        "project_name": "ManyThings",
		"model": "Model",
        "dataset": {
            "name": "ManyThings",
            "dataset_args": None
        },
        "network": {
            "name": "nn_lstm1",
            "net_args": {
                "decay": 1.0e-06,
                "dropout": 0.2,
                "epochs": 20,
                "latent_dim": 300,
                "learn-rate": 0.01,
                "momentum": 0.9
            },
            "io_shapes": {
                "input_shape": None,
                "output_shape": None
            }
        }
    }

    yaml_dump("manythings/experiment.yaml", data)
    print("Done!")
