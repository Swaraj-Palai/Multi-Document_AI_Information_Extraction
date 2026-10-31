from setuptools import setup, find_packages

with open("req.txt") as f:
    requirement = f.read().splitlines()

setup(
    packages=find_packages(),
    install_requires=requirement
)

