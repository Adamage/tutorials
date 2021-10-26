# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from pathlib import Path

import click

from src.execute import execute_conversion, execute_multiple_conversions, \
    convert_to_outputs, prepare_conversion_config
from src.output_types import OutputTypes, supported_types
from src.utils.file import set_output_extension_and_type
from src.constants import README_FILE_NAME


@click.group()
def cli():
    pass


@cli.command()
@click.option('--source', '-s', required=True, type=Path,
              help='Absolute or relative path to python file to be converted')
@click.option('--output', '-o', required=True, type=Path,
              help='Absolute or relative path to output file. Output file type is taken from extension. '
                   'Output filepath can be provided without extenstion, then type is taken from --type.')
@click.option('--type', '-t', type=click.Choice(supported_types()), default=None,
              help='Desired output file type. Parameter is ignored when --output contains specified file extension')
@click.option('--execute/--no-execute', default=False, help='Flag whether the file is to be executed or not')
def convert(source: Path, output: Path, type: OutputTypes, execute: bool) -> None:
    """
    Transforms source python file into specified format (jupyter notebook, markdown, python code file without
    documentation)
    """
    output, type = set_output_extension_and_type(output, type)

    assert source.suffix == '.py', 'Only python file can be single source file'
    assert output != source, f'Your source file and the expected output file name are the same: {source}, ' \
                             f'specify different outfile name using --output flag.'
    execute_conversion(source=source, output=output, output_type=type, execute=execute)


@cli.command()
@click.option('--source', '-s', required=True, type=Path,
              help='Absolute or relative path to python file to be converted')
@click.option('--output-dir', '-o', required=False, type=Path,
              help='Absolute or relative path to output directory.')
@click.option('--markdown-name', '-m', required=False, type=Path, default=README_FILE_NAME,
              help='Custom name for output Markdown file.')
@click.option('--include-code-output/--exclude-code-output', required=False, type=bool, default=True,
              help='Flag deciding if the code-only output should be exported.')
def convert2all(source: Path, output_dir: Path, markdown_name: str, include_code_output: bool) -> None:
    """
    Transforms source python file automatically into three specified formats with specified configuration:
    jupyter notebook, executed markdown file and python code script.
    """
    configuration = prepare_conversion_config(source=source,
                                              output_dir=output_dir,
                                              execute=True,
                                              include_code_output=include_code_output,
                                              markdown_name=markdown_name)

    convert_to_outputs(source=source, configuration=configuration)


@cli.command()
@click.option('--config', '-c', required=True, type=Path,
              help='Absolute or relative path to YAML file with list of all files to execute')
@click.option('--source-dir', '-s', required=True, type=Path,
              help='Absolute or relative path to directory with all files, relative to which, the config YML has '
                   'been created')
@click.option('--output-dir', '-o', required=True, type=Path,
              help='Absolute or relative path to output directory for all files')
@click.option('--execute/--no-execute', default=True, help='Flag whether the notebook is to be executed or not')
@click.option('--include-code-output/--exclude-code-output', required=False, type=bool, default=True,
              help='Flag deciding if the code-only output should be exported.')
def batch_convert(config: Path, source_dir: Path, output_dir: Path, execute: bool, include_code_output: bool) -> None:
    """
    Transforms python files specified in config into all possible formats: jupyter notebook, markdown and python code.
    Each source will get it's own output directory under the 'output_dir'.
    """
    execute_multiple_conversions(
        source_directory=source_dir,
        output_directory=output_dir,
        config_path=config,
        execute=execute,
        include_code_output=include_code_output
    )


if __name__ == '__main__':
    cli()  # pragma: no cover
