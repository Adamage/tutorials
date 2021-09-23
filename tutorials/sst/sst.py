from pathlib import Path

import click

from src.exporters import execute_multiple_exporters, execute_single_exporter
from src.format_converter import set_output_extension_and_type
from src.output_types import OutputTypes, supported_types


@click.group()
def cli():
    pass


@cli.command()
@click.option('--source', '-s', required=True, type=Path,
              help='Absolute or relative path to python file to be converted')
@click.option('--output', '-o', required=True, type=Path,
              help='Absolute or relative path to output file. Output file type is taken from extension. '
                   'Output filepath can be provided without extenstion, then type is taken from --type.')
@click.option('--type', '-f', type=click.Choice(supported_types() + [None]), default=None,
              help='Desired output file type. Parameter is ignored when --output contains specified file extension')
@click.option('--execute/--no-execute', default=True, help='Flag whether the notebook is to be executed or not')
def convert(source: Path, output: Path, type: OutputTypes, execute: bool) -> None:
    """
    Command used to generate all outputs with one flow.
    """
    output, type = set_output_extension_and_type(output, type)
    assert output != source, f'Your source file and the expected output file name are the same: {source}, ' \
                             f'specify different outfile name using --output flag.'
    execute_single_exporter(execute, output, source, type)


@cli.command()
@click.option('--config', '-c', required=True, type=Path,
              help='Absolute or relative path to YAML file with list of all tutorials to execute')
@click.option('--input-dir', '-o', required=True, type=Path,
              help='Absolute or relative path to directory with all tutorials, relative to which, the config YML has '
                   'been created')
@click.option('--output-dir', '-o', required=True, type=Path,
              help='Absolute or relative path to output directory for all tutorials')
@click.option('--execute/--no-execute', default=True, help='Flag whether the notebook is to be executed or not')
def batch_convert(config: Path, input_dir: Path, output_dir: Path, execute: bool) -> None:
    """
    Command used to generate all outputs with one flow.
    """
    execute_multiple_exporters(
        config_path=config,
        input_directory=input_dir,
        output_directory=output_dir,
        execute=execute
    )


if __name__ == '__main__':
    cli()  # pragma: no cover
