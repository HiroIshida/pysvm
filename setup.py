from setuptools import setup, find_packages
setup(
    name='svm',
    version='0.0.1',
    description='Sample package for Python-Guide.org',
    license=license,
    install_requires=['numpy'],
    packages=find_packages(exclude=('tests', 'docs'))
)
