# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
CELL_SEPARATOR = '"""'
SHEBANG_MARKER = "#!"
COPYRIGHT_TAG = 'copyright'


SST_HIDE_OUTPUT_TAG = 'sst_hide_output'
SST_IGNORE_JUPYTER_MD_TAG = 'sst_ignore_jupyter_md'
ALLOWED_TAGS = [SST_HIDE_OUTPUT_TAG, SST_IGNORE_JUPYTER_MD_TAG]

NBCONVERT_RESOURCE_OUTPUT_EXT_KEY = 'output_extension'
NBCONVERT_RESOURCE_OUTPUT_DIR_KEY = 'output_files_dir'
NBCONVERT_RESOURCE_OUTPUTS_KEY = 'outputs'
IMAGES_DIR = 'outputs'

CODE_SUFFIX = '_code_only'
README_FILE_NAME = 'README.md'

REGEX_COPYRIGHT_PATTERN = r'[\S\s]*(copyright)[\S\s]*'
