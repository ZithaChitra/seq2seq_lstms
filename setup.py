from setuptools import setup, find_packages
import pathlib
from os import path


__version__ = "0.1.0"

cwd = pathlib.Path.cwd()

# get dependencies and installs
with open(path.join(cwd, "requirements.txt"), encoding="utf-8") as f:
	all_reqs = f.read().split("\n")

install_requires = [x.strip() for x in all_reqs if "git+" not in x]
dependency_links = [x.strp().replace("git+", "") for x in all_reqs if x.startswith("git+")]



setup(
	name="ManyThings",
	version=__version__,
	packages=find_packages(),
	install_requires=install_requires,
	dependency_links=dependency_links,
)