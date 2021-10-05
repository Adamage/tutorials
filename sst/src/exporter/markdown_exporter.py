# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import os

from nbconvert import MarkdownExporter
from nbformat import NotebookNode

from src.constants import COPYRIGHT_TAG


class MarkdownExporterWrapper(MarkdownExporter):
    def __init__(self, **kw):
        super().__init__(**kw)

    def from_notebook_node(self, notebook: NotebookNode, **kwargs):
        self.update_first_copyright_node(notebook)
        return super().from_notebook_node(notebook, **kwargs)

    @classmethod
    def update_first_copyright_node(cls, notebook) -> None:
        """
        Search method that finds the first copyright Markdown node, and then remove each line which contain copyright tag
        """
        cell_id_to_remove = None
        for cell_id, cell in enumerate(notebook.cells):
            if COPYRIGHT_TAG in cell.source.lower():
                lines = filter(MarkdownExporterWrapper.is_copyright_in_line, cell.source.splitlines())
                new_source = os.linesep.join(lines).strip(os.linesep)

                if new_source:
                    cell.source = new_source
                else:
                    cell_id_to_remove = cell_id

                break

        if cell_id_to_remove is not None:
            del notebook.cells[cell_id_to_remove]

    @staticmethod
    def is_copyright_in_line(line):
        return COPYRIGHT_TAG not in line.lower()
