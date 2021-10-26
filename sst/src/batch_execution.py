# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from collections import namedtuple
from pathlib import Path
from typing import List, Dict

import yaml

SSTConversionConfig = namedtuple("SSTConversionConfig", "name source markdown_name include_code_only")


def batch_config(config_path: Path) -> List[SSTConversionConfig]:
    parsed_configs = _parse_config(config_path).get("files", [])
    if not parsed_configs:
        raise AttributeError("Provided configuration file has no specified files to convert!")

    sst_conversion_configs = [
        SSTConversionConfig(
            name=conversion_config.get("name", "no_name"),
            source=conversion_config.get("source", None),
            markdown_name=conversion_config.get("markdown_name", None),
            include_code_only=conversion_config.get("include_code_only", True)
        )
        for conversion_config in parsed_configs
    ]

    if any(list(tc.source is None for tc in sst_conversion_configs)):
        raise AttributeError(f"Not all configs specify source path! {sst_conversion_configs}")

    return sst_conversion_configs


def _parse_config(config_path: Path) -> Dict:
    with open(config_path) as f:
        return yaml.load(f,  yaml.SafeLoader)
