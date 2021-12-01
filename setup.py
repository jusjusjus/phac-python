from os.path import join
import setuptools
#
# We expect "mffpy/version.py" to be very simple:
#
# > __version__ = "x.y.z"
#
# (from one of the answers in
# https://stackoverflow.com/questions/458550/standard-way-to-embed-version-into-python-package)
__version__ = ''
exec(open(join('phac', 'version.py')).read())
requirements = open('requirements.txt').read().split('\n')

setup(
    name='phac',
    version=__version__,
    packages=['phac', 'phac.models'],
    install_requires=requirements,
)
