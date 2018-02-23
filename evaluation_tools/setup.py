from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=['evaluation_tools'],
    package_dir={'':'python'},
    scripts=[])

setup(**setup_args)