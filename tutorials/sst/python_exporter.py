import os

from nbconvert import Exporter
from nbformat import NotebookNode


class PythonExporter(Exporter):
    file_extension = '.py'

    def __init__(self, **kw):
        super().__init__(**kw)

    def from_notebook_node(self, notebook: NotebookNode, **kwargs):
        code_cells = [cell.source for cell in notebook.get('cells', []) if cell.get('cell_type') == 'code']
        py_code = os.linesep.join(code_cells + [os.linesep])
        return py_code, []
