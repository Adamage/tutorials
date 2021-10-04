from src.exporter.markdown_exporter import MarkdownExporterWrapper
from src.format_converter import py_to_ipynb


def test_when_only_copyright_in_code_then_remove_cell():
    md_exporter = MarkdownExporterWrapper()

    notebook = py_to_ipynb('\n# Copyright \n""" \nMarkdown cell \n"""" ')
    md_exporter.update_first_copyright_node(notebook)
    assert len(notebook.cells) == 1


def test_when_only_copyright_in_md_then_remove_cell():
    md_exporter = MarkdownExporterWrapper()

    notebook = py_to_ipynb('\n"""\n\n# Copyright \n\n\n""" \nCode cell \n')
    md_exporter.update_first_copyright_node(notebook)
    print(notebook)
    assert len(notebook.cells) == 1


def test_when_not_only_copyright_in_code_then_not_remove_cell():
    md_exporter = MarkdownExporterWrapper()

    notebook = py_to_ipynb('\n# Copyright\nprint(x)\n\n""" \nMarkdown cell \n"""" ')
    md_exporter.update_first_copyright_node(notebook)
    assert len(notebook.cells) == 2
    assert 'Copyright' not in notebook.cells[0].source


def test_when_not_only_copyright_in_md_then_not_remove_cell():
    md_exporter = MarkdownExporterWrapper()

    notebook = py_to_ipynb('\n"""\n\n# Copyright\n#print(x) \n\n\n""" \nCode cell \n')
    md_exporter.update_first_copyright_node(notebook)
    print(notebook)
    assert len(notebook.cells) == 2
    assert 'Copyright' not in notebook.cells[0].source


def test_when_not_only_copyright_in_source_check_only_first_delete():
    md_exporter = MarkdownExporterWrapper()

    notebook = py_to_ipynb('\n# copyright\nCopyright\n\n""" \nMarkdown cell \n""""\n# copyright')
    md_exporter.update_first_copyright_node(notebook)
    assert len(notebook.cells) == 2
    assert 'copyright' in notebook.cells[1].source


def test_when_not_only_copyright_in_md_check_only_first_delete():
    md_exporter = MarkdownExporterWrapper()

    notebook = py_to_ipynb('\n"""\n\n# copyright\n#Copyright \n\n\n""" \nCode cell \n"""\ncopyright\n"""')
    md_exporter.update_first_copyright_node(notebook)
    print(notebook)
    assert len(notebook.cells) == 2
    assert 'copyright' in notebook.cells[1].source
