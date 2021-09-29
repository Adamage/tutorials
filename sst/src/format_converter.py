# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import os
from enum import Enum
from typing import List

from nbformat import NotebookNode
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

from src.constants import CELL_SEPARATOR, REMOVE_OUTPUT_TAG


def code_preprocessor(input_source: str) -> str:
    return input_source.strip(os.linesep)


def markdown_preprocessor(input_source: str) -> str:
    return input_source


class CellType(Enum):
    CODE = 1
    MARKDOWN = 2


TYPE2FUNC = {
    CellType.CODE: new_code_cell,
    CellType.MARKDOWN: new_markdown_cell,
}

TYPE2PREPROCESSOR = {
    CellType.CODE: code_preprocessor,
    CellType.MARKDOWN: markdown_preprocessor,
}


def py_to_ipynb(py_file_text: str) -> NotebookNode:
    """
    The python file content is parsed line by line. The type of fragment (markdown, codecell) is determined together
    with tags. Based on that data, an object which represents Notebook is created.

    Args:
        py_file_text: The content of a python file stored in a string variable

    Returns:
        NotebookNode file that represents Notebook object
    """
    cells = []

    current_cell_type = CellType.CODE
    cell_lines = []

    for line in py_file_text.splitlines():
        if not line.startswith(CELL_SEPARATOR):
            cell_lines.append(line)
            continue

        if cell_lines:
            new_cell = create_cell_from_lines(cell_lines=cell_lines, cell_type=current_cell_type)
            cells.append(new_cell)
            cell_lines = []

        if current_cell_type == CellType.CODE:
            current_cell_type = CellType.MARKDOWN
        elif current_cell_type == CellType.MARKDOWN:
            current_cell_type = CellType.CODE

    if cell_lines:
        new_cell = create_cell_from_lines(cell_lines, current_cell_type)
        cells.append(new_cell)

    cells = remove_shebang(cells)

    notebook = new_notebook(cells=cells)

    return notebook


def remove_shebang(cells: List[NotebookNode]) -> List[NotebookNode]:
    """
    Args:
        cells - list of notebook cells
    """
    if cells and cells[0].source.startswith("#!"):
        cells = cells[1:]
    return cells


def create_cell_from_lines(cell_lines: List[str], cell_type: CellType) -> NotebookNode:
    """
    Args:
        cell_lines: List of content in each line
        cell_type: Recognized cell type

    Returns:
        NotebookNode which contains specified type and merged content
    """
    source = os.linesep.join(cell_lines)
    processed_source = TYPE2PREPROCESSOR[cell_type](source)
    cell = TYPE2FUNC[cell_type](processed_source)

    if cell_type == CellType.CODE:
        cell = handle_cell_tag(cell, REMOVE_OUTPUT_TAG)

    return cell


def handle_cell_tag(cell: NotebookNode, tag: str) -> NotebookNode:
    """
    Adds information about tag into cell data and remove line with tag from text in the cell
    Args:
        cell: Cell to be analyzed
        tag: Tag we look for

    Returns:
        Updated cell
    """
    if tag in cell.source:
        cell.metadata.update({"tags": [tag]})
        cell = remove_from_cell_source(cell, f"# {tag}")
    return cell


def remove_from_cell_source(cell: NotebookNode, string_to_remove: str) -> NotebookNode:
    cell.source = os.linesep.join(
        [
            line for line in cell.source.splitlines()
            if string_to_remove not in line
        ]
    )
    return cell
