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
    author="Pierre Gleize, Weiyao Wang, Matt Feiszli",
    author_email="",
    url="https://github.com/facebookresearch/silk",
    packages=["silk"],
    install_requires=load_requirements("requirements.txt"),
)
