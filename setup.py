import os
import sys
from setuptools import setup, find_packages
PACKAGE_NAME = 'CandC_Framework'
MINIMUM_PYTHON_VERSION = 3, 5


def check_python_version():
    """Exit when the Python version is too low."""
    if sys.version_info < MINIMUM_PYTHON_VERSION:
        sys.exit("Python {}.{}+ is required.".format(*MINIMUM_PYTHON_VERSION))


def read_package_variable(key):
    """Read the value of a variable from the package without importing."""
    module_path = os.path.join(PACKAGE_NAME, '__init__.py')
    with open(module_path) as module:
        for line in module:
            parts = line.strip().split(' ')
            if parts and parts[0] == key:
                return parts[-1].strip("'")
    assert 0, "'{0}' not found in '{1}'".format(key, module_path)


check_python_version()
setup(
    name='CandC_Framework',
    version=read_package_variable('__version__'),
    description='A Python Library for gathering certainty and competence scores, and using them to perform out-of-distribution detection',
    author='Alexander Berenbeim',
    author_email='alexander.berenbeim@westpoint.edu',
    packages=find_packages(),
    install_requires=['torch>=1.6.0', 'numpy', 'pickle', 'statsmodels','plotly','keras>=2.0', 'tqdm','pytorch_ood','gc','math','pandas','hamiltorch','itertools','datetime','sklearn','typing'],
    url='https://github.com/Army-Cyber-Institute/UCQ',
    classifiers=['Development Status :: 4 - Beta', 'License :: OSI Approved :: BSD License', 'Programming Language :: Python :: 3.10'],
    license='BSD',
    keywords='pytorch certainty competence uncertainty quantification bnn',
)
