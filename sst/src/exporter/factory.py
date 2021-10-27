# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from nbconvert import Exporter, NotebookExporter, MarkdownExporter
from nbconvert.preprocessors import TagRemovePreprocessor, ExtractOutputPreprocessor
from traitlets.config import Config

from src.exporter.code_exporter import CodeExporter
from src.exporter.execute_preprocessor_with_progress_bar import ExecutePreprocessorWithProgressBar
from src.exporter.preprocessors import \
    configure_tag_remove_preprocessor_to_hide_output, \
    configure_copyright_regex_removal_preprocessor, \
    RegexWithFlagsRemovePreprocessor, \
    configure_tag_remove_preprocessor_to_remove_cell
from src.output_types import OutputTypes


def markdown_exporter_with_preprocessors(execute_enabled: bool) -> Exporter:
    exporter = MarkdownExporter()

    config = Config()
    for apply_configuration in [
        configure_tag_remove_preprocessor_to_remove_cell,
        configure_tag_remove_preprocessor_to_hide_output,
        configure_copyright_regex_removal_preprocessor,
    ]:
        config = apply_configuration(config)

    # TagRemovePreprocessor must be before and after the execution because it has to remove the cells before the
    # jupyter notebook is executed and after to remove the generated outputs
    preprocessors = [TagRemovePreprocessor, ExecutePreprocessorWithProgressBar] if execute_enabled else []
    preprocessors += [TagRemovePreprocessor, RegexWithFlagsRemovePreprocessor, ExtractOutputPreprocessor]

    for preprocessor in preprocessors:
        exporter.register_preprocessor(preprocessor(config=config), enabled=True)

    return exporter


def notebook_exporter_with_preprocessors(execute_enabled: bool) -> Exporter:
    exporter = NotebookExporter()

    config = Config()
    for apply_configuration in [
        configure_tag_remove_preprocessor_to_remove_cell,
        configure_tag_remove_preprocessor_to_hide_output,
    ]:
        config = apply_configuration(config)

    preprocessors = [TagRemovePreprocessor]
    if execute_enabled:
        # TagRemovePreprocessor must be before and after the execution because it has to remove the cells before the
        # jupyter notebook is executed and after to remove the generated outputs
        preprocessors += [ExecutePreprocessorWithProgressBar, TagRemovePreprocessor]

    for preprocessor in preprocessors:
        exporter.register_preprocessor(preprocessor(config=config), enabled=True)

    return exporter


def code_exporter(execute_enabled: bool) -> Exporter:
    return CodeExporter()


TYPE2EXPORTER = {
    OutputTypes.JUPYTER_TYPE: notebook_exporter_with_preprocessors,
    OutputTypes.MARKDOWN_TYPE: markdown_exporter_with_preprocessors,
    OutputTypes.CODE_TYPE: code_exporter
}


def exporter_factory(type: OutputTypes, execute_enabled: bool) -> Exporter:
    exporter_factory = TYPE2EXPORTER.get(type)
    exporter = exporter_factory(execute_enabled=execute_enabled)
    return exporter
