# Copyright © 2025 Apple Inc.

import sys
from pathlib import Path

from setuptools import setup

package_dir = Path(__file__).parent / "mlx_interp"
with open(package_dir / "requirements.txt") as fid:
    requirements = [l.strip() for l in fid.readlines()]

sys.path.append(str(package_dir))
from _version import __version__

setup(
    name="mlx-interp",
    version=__version__,
    description="LLMs on Apple silicon with MLX and the Hugging Face Hub",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    readme="README.md",
    author_email="mlx@group.apple.com",
    author="MLX Contributors",
    url="https://github.com/ml-explore/mlx-examples",
    license="MIT",
    install_requires=requirements,
    packages=["mlx_interp"],
    python_requires=">=3.8",
    extras_require={
    },
    entry_points={
        "console_scripts": [
            # "mlx_interp.chat = mlx_interp.chat:main",
            # "mlx_lm.convert = mlx_lm.convert:main",
            # "mlx_lm.evaluate = mlx_lm.evaluate:main",
            # "mlx_lm.fuse = mlx_lm.fuse:main",
            "mlx_interp.generate = mlx_interp.generate:main",
            # "mlx_lm.server = mlx_lm.server:main",
        ]
    },
)
