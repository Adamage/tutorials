# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
from nbconvert import Exporter, MarkdownExporter, NotebookExporter
from nbconvert.preprocessors import ExecutePreprocessor, TagRemovePreprocessor, ExtractOutputPreprocessor
from traitlets.config import Config

from src.exporter.preprocessors import configure_tag_removal_preprocessor, configure_extract_outputs_preprocessor
from src.exporter.pure_python import PurePythonExporter
from src.output_types import OutputTypes


def markdown_exporter_with_preprocessors(execute_enabled: bool) -> Exporter:
    config = Config()
    config = configure_tag_removal_preprocessor(config)
    config = configure_extract_outputs_preprocessor(config)

    exporter = MarkdownExporter()
    exporter.register_preprocessor(ExecutePreprocessor(), enabled=execute_enabled)
    exporter.register_preprocessor(TagRemovePreprocessor(config=config), enabled=True)
    exporter.register_preprocessor(ExtractOutputPreprocessor(config=config), enabled=True)

    return exporter


def notebook_exporter_with_preprocessors(execute_enabled: bool) -> Exporter:
    exporter = NotebookExporter()
    exporter.register_preprocessor(ExecutePreprocessor(), enabled=execute_enabled)

    return exporter


def pure_python_exporter(execute_enabled: bool) -> Exporter:
    return PurePythonExporter()


TYPE2EXPORTER = {
    OutputTypes.JUPYTER_TYPE: notebook_exporter_with_preprocessors,
    OutputTypes.MARKDOWN_TYPE: markdown_exporter_with_preprocessors,
    OutputTypes.PUREPYTHON_TYPE: pure_python_exporter
}


def exporter_factory(type: OutputTypes, execute_enabled: bool) -> Exporter:
    exporter_factory = TYPE2EXPORTER.get(type)
    exporter = exporter_factory(execute_enabled=execute_enabled)
    return exporter
