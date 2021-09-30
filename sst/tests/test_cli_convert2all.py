# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from src.utils.click import print_exception
from sst import cli
from tests.test_utils.path import get_unit_test_static_files_dir


TRIVIAL_MAPPING_SOURCE_PATH = get_unit_test_static_files_dir() / "trivial_mapping_md_code_md.py"


@pytest.fixture
def cli_runner_instance():
    return CliRunner()


def test_cli_convert2all_when_input_file_is_not_py(cli_runner_instance):
    with pytest.raises(AssertionError) as e_info:
        result = cli_runner_instance.invoke(cli, ['convert2all', '--source', 'input.txt'])
        if result.exception:
            raise result.exception


def test_cli_convert2all_when_no_output_dir(cli_runner_instance, tmp_path):
    my_file = tmp_path / 'nested/my_file.py'
    my_file.parent.mkdir(exist_ok=False, parents=True)
    my_file.write_text("print('hello')")
    outfile_path = tmp_path / 'nested' / 'my_file'

    result = cli_runner_instance.invoke(cli, ['convert2all', '--source', my_file])
    print_exception(result)

    assert os.path.exists(outfile_path.with_suffix('.md'))
    assert os.path.exists(outfile_path.with_suffix('.ipynb'))
    assert os.path.exists(outfile_path.with_name('my_file_pure.py'))


def test_cli_convert2all_when_correct_input(cli_runner_instance, tmp_path):
    result = cli_runner_instance.invoke(cli, ['convert2all', '--source', TRIVIAL_MAPPING_SOURCE_PATH, '--output-dir', tmp_path])
    print_exception(result)

    outfile_path = tmp_path / Path('trivial_mapping_md_code_md')

    assert os.path.exists(outfile_path.with_suffix('.md'))
    assert "#!" not in outfile_path.with_suffix('.md').read_text()

    assert os.path.exists(outfile_path.with_suffix('.ipynb'))
    assert "#!" not in outfile_path.with_suffix('.ipynb').read_text()

    assert os.path.exists(outfile_path.with_name('trivial_mapping_md_code_md_pure.py'))
    assert "#!" in outfile_path.with_name('trivial_mapping_md_code_md_pure.py').read_text()

    assert result.exit_code == 0
