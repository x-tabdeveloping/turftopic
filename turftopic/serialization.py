import json
import warnings
from pathlib import Path
from typing import Union

import joblib
import pkg_resources
from huggingface_hub import snapshot_download

IMPORTANT_PACKAGES = [
    "scikit-learn",
    "sentence-transformers",
    "turftopic",
    "joblib",
]

README_TEMPLATE = """
---
pipeline_tag: text-classification
library_name: turftopic
tags:
- turftopic
- topic-modelling
---

# {model_path}

This repository contains a topic model trained with the [Turftopic](https://github.com/x-tabdeveloping/turftopic) Python library.

To load and use the model run the following piece of code:

```python
from turftopic import load_model

model = load_model({model_path})
model.print_topics()
```

## Model Structure

The model is structured as follows:

```
{model_structure}
```

## Topics
The topics discovered by the model are the following:

{topics_table}

## Package versions

The model in this repo was trained using the following package versions:

{package_version_table}

We recommend that you install the same, or compatible versions of these packages locally, before trying to load a model.

"""


def get_package_versions() -> dict[str, str]:
    return {
        package: pkg_resources.get_distribution(package).version
        for package in IMPORTANT_PACKAGES
    }


def validate_package_versions(remote_versions: dict[str, str]):
    local_versions = get_package_versions()
    for package in IMPORTANT_PACKAGES:
        local = local_versions.get(package, None)
        remote = remote_versions.get(package, None)
        if local != remote:
            warnings.warn(
                f"Local version of {package} ({local}) does not match the version in the model you are trying to load ({remote}). This might cause some issues."
            )


def create_readme(model, model_path: str) -> str:
    model_structure = str(model)
    topics_table = model.export_topics(format="markdown", top_k=10)
    local_versions = get_package_versions()
    lines = ["| Package | Version |", "| - | - |"]
    for package in IMPORTANT_PACKAGES:
        version = local_versions.get(package, "")
        lines.append(f"| {package} | {version} |")
    package_version_table = "\n".join(lines)
    return README_TEMPLATE.format(
        model_path=model_path,
        model_structure=model_structure,
        topics_table=topics_table,
        package_version_table=package_version_table,
    )


def load_model(repo_id_or_path: Union[str, Path]):
    """Loads topic model from local directory or HuggingFace Hub repository.

    Parameters
    ----------
    repo_id_or_path: str | Path
        Path to local directory or HuggingFace repository ID.
    """
    path = Path(repo_id_or_path)
    if path.is_dir():
        with path.joinpath("package_versions.json").open() as ver_file:
            remote_versions = json.loads(ver_file.read())
        validate_package_versions(remote_versions)
        model = joblib.load(path.joinpath("model.joblib"))
        return model
    else:
        in_dir = snapshot_download(repo_id=repo_id_or_path)
        return load_model(in_dir)
