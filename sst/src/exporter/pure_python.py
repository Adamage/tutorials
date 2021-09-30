# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import os
from collections import deque

from nbconvert import Exporter
from nbconvert.exporters.exporter import ResourcesDict
from nbformat import NotebookNode


def shebang():
    return f"#!/usr/bin/env python3"


class PurePythonExporter(Exporter):
    file_extension = '.py'

    def __init__(self, **kw):
        super().__init__(**kw)

    def from_notebook_node(self, notebook: NotebookNode, **kwargs):
        code_cells = deque(cell.source + os.linesep
                           for cell in notebook.get('cells', []) if cell.get('cell_type') == 'code')

        code_cells.appendleft(shebang())

        py_code = os.linesep.join(code_cells)
        return py_code, ResourcesDict()
