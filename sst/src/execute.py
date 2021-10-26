# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from pathlib import Path
from typing import List

from nbconvert.exporters.exporter import ResourcesDict
from nbconvert.writers import FilesWriter
from tqdm import tqdm

from src.batch_execution import batch_config
from src.constants import NBCONVERT_RESOURCE_OUTPUT_EXT_KEY, IMAGES_DIR, \
    NBCONVERT_RESOURCE_OUTPUT_DIR_KEY, README_FILE_NAME
from src.exporter.factory import exporter_factory
from src.format_converter import py_to_ipynb
from src.output_types import OutputTypes
from src.utils.file import output_path_code, \
    output_path_markdown, output_path_jupyter
from src.utils.logger import get_logger

logger = get_logger()


def execute_conversion(source: Path, output: Path, output_type: OutputTypes, execute: bool):
    py_text = source.read_text()
    notebook = py_to_ipynb(py_text)

    exporter = exporter_factory(type=output_type, execute_enabled=execute)

    warning_output_location_differs_with_source(output, source)

    output_content, resources = exporter.from_notebook_node(
        notebook,
        resources={
            NBCONVERT_RESOURCE_OUTPUT_DIR_KEY: output.parent / f"{source.stem}_{IMAGES_DIR}"
        }
    )

    save_conversion_results(output, output_content, resources)


def warning_output_location_differs_with_source(output: Path, source: Path):
    if source.parent != output.parent:
        logger.warning(
            f"Outputs will be generated in a location different than the source"
            f" script file. Static links in Markdown and Jupyter will stop"
            f" working. Source: [ {source} ], output: [ {output} ]"
        )


def save_conversion_results(output: Path, output_content: str, resources: ResourcesDict):
    output.parent.mkdir(parents=True, exist_ok=True)

    if resources and NBCONVERT_RESOURCE_OUTPUT_EXT_KEY in resources:
        del resources[NBCONVERT_RESOURCE_OUTPUT_EXT_KEY]

    writer = FilesWriter(build_directory=str(output.parent))
    writer.write(output=output_content, resources=resources, notebook_name=str(output.name))


def execute_multiple_conversions(
        source_directory: Path,
        output_directory: Path,
        config_path: Path,
        execute: bool) -> None:

    for tc in tqdm(batch_config(config_path), desc="SST All Configs"):
        source = source_directory / tc.source
        markdown_name = tc.markdown_name

        tc_output = output_directory / tc.name
        tc_output.mkdir(parents=True, exist_ok=True)

        conversion_config = prepare_conversion_config(
            source=source,
            output_dir=tc_output,
            include_code_output=tc.include_code_only,
            markdown_name=markdown_name if markdown_name else README_FILE_NAME,
            execute=execute
        )
        convert_to_outputs(source=source, configuration=conversion_config)


def convert_to_outputs(source: Path, configuration: List) -> None:
    for outfile, output_type, execution in tqdm(configuration, desc="SST Config", leave=False):
        execute_conversion(
            source=source,
            output=outfile,
            output_type=output_type,
            execute=execution
        )


def prepare_conversion_config(
        source: Path,
        output_dir: Path,
        execute: bool,
        include_code_output: bool,
        markdown_name: str
) -> List:

    assert source.suffix == '.py', 'Only python file can be single source file'

    if output_dir is None:
        output_dir = source.parent

    output_dir.mkdir(parents=True, exist_ok=True)

    output_filename = output_dir / source.stem

    markdown_filename = output_dir / markdown_name

    configuration = [
        [output_path_markdown(markdown_filename), OutputTypes.MARKDOWN_TYPE, execute],
        [output_path_jupyter(output_filename), OutputTypes.JUPYTER_TYPE, False]
    ]

    if include_code_output:
        configuration.append(
            [output_path_code(output_filename), OutputTypes.CODE_TYPE, False]
        )

    return configuration
