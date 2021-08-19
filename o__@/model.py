import importlib
from osgeo import gdal
import click


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    pass


@cli.command()
@click.argument("model", required=True)
@click.option("--gpus", default="0")
def train(model, gpus):
    mod = importlib.import_module(model)
    mod.run_train(gpus)


@cli.command()
@click.argument("model", required=True)
@click.option("--gpus", default="0")
def test(model, gpus):
    mod = importlib.import_module(model)
    mod.run_test(gpus)


if __name__ == "__main__":
    cli()
