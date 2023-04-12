#!/usr/bin/env python3

from distutils.core import setup

from pip._internal.network.session import PipSession
from pip._internal.req import parse_requirements


def load_requirements(fname):
    reqs = parse_requirements(fname, session=PipSession())
    return [ir.requirement for ir in reqs]


setup(
    name="SiLK Keypoint Library",
    version="1.0",
    description="[FAIR] SiLK - Simple Learned Keypoints",
    author="",  # TODO(Pierre) : Add team authors in alphabetical order or use team name and team email.
    author_email="",
    url="",  # TODO(Pierre) Update link.
    packages=["silk"],
    install_requires=load_requirements("requirements.txt"),
)
