import os
from setuptools import setup, find_packages

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "deepton",
    version = "0.0.2",
    author = "Moctar Diallo",
    author_email = "moctarjallo@gmail.com",
    description = ("Introduction to Artificial Intelligence and Python Programming \
                    for beginner learners"),
    license = "MIT",
    keywords = "python biginner learner learn programming artificial-intelligence deep-learning",
    packages=find_packages(),
    package_dir={'deepton': 'deepton'},
    url='https://github.com/kajande/deepton',

)