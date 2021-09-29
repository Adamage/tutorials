# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
from traitlets.config import Config

from src.constants import SST_HIDE_OUTPUT_TAG


def configure_tag_removal_preprocessor(c: Config):
    c.TagRemovePreprocessor.remove_all_outputs_tags = (SST_HIDE_OUTPUT_TAG,)
    c.TagRemovePreprocessor.enabled = True
    return c


def configure_extract_outputs_preprocessor(c: Config):
    c.ExtractOutputsPreprocessor.enabled = True
    return c
