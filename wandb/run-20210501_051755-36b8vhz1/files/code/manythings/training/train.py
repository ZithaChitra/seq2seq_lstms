from typing import Dict
import importlib
import numpy as np
from manythings.util_yaml import yaml_loader
import click
import wandb
from wandb.keras import WandbCallback
import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec

# api_key = os.environ['WANDB_API_KEY']

@click.command()
@click.argument("experiment-config",
                type=click.Path(exists=True),
                default="manythings/experiment.yaml")
@click.option("--latent_dim", default=300)
@click.option("--decay", default=1.0e-06)
@click.option("--dropout", default=0.2)
@click.option("--epochs", default=3)
@click.option("--learn_rate", default=0.01)
@click.option("--momentum", default=0.9)
def main(experiment_config, latent_dim: int, decay: float, dropout: float,
         epochs: int, learn_rate: float, momentum: float):
    """ Update values in experiment configuration file  """
    exp_config = yaml_loader("manythings/experiment.yaml")
    proj_name = exp_config.get("project_name")
    net_name = exp_config.get("network")["name"]

    net_args = exp_config.get("network")["net_args"]
    net_args["hidden_layer_size"] = latent_dim
    net_args["decay"] = decay
    net_args["dropout"] = dropout
    net_args["epochs"] = epochs
    net_args["learn_rate"] = learn_rate
    net_args["momentum"] = momentum

    dataset_cls = exp_config.get("dataset")["name"]
    dataset_args = exp_config.get("dataset")["dataset_args"]

    model = exp_config.get("model")

    wandb.login()


    train(proj_name, model, dataset_cls, net_name, net_args, dataset_args)


def train(
    proj_name: str,
    Model: str,
    dataset_cls: str,
    net_fn: str,
    net_args: Dict,
    dataset_args: Dict,
):
    """ Train Function """

    dataset_module = importlib.import_module(
        f"manythings.data.dta_{dataset_cls}")
    dataset_cls_ = getattr(dataset_module, dataset_cls)

    network_module = importlib.import_module(f"manythings.networks.{net_fn}")
    network_fn_ = getattr(network_module, net_fn)

    model_module = importlib.import_module(f"manythings.models.{Model}")
    model_cls_ = getattr(model_module, Model)

    config = {
        "model": Model,
        "dataset_cls": dataset_cls,
        "net_fn": net_fn,
        "net_args": net_args,
        "dataset_args": dataset_args
    }

    # input_schema = Schema([
    #     TensorSpec(np.dtype(np.uint8), (-1, 28, 28, 1)),
    # ])
    # output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 10))])
    # signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    with wandb.init(project=proj_name, config=config):
        """"""
        config = wandb.config
        model = model_cls_(dataset_cls_, network_fn_, net_args, dataset_args)

        callbacks = [WandbCallback(
			# log_weights=True,
			#  log_gradients=True
			 )]

        model.fit(callbacks=callbacks)
        # mlflow.keras.save_model(model.network,
        #                         "saved_models/seq2seq",
        #                         signature=signature)


if __name__ == "__main__":
    main()
