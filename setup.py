#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Alexandre Bittar <abittar@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#

from distutils.core import setup

import setuptools

setup(
    name="sparse",
    version="1.0",
    description="Spike-driven Physiologically-inspired Approach towards Realistic Speech Encoding",
    author="Alexandre Bittar",
    author_email="abittar@idiap.ch",
    packages=setuptools.find_packages(),
)
