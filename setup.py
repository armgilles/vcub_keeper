from setuptools import find_packages, setup
import vcub_keeper

VERSION = vcub_keeper.__version__

setup(
    name='vcub_keeper',
    packages=find_packages(),
    version=VERSION,
    install_requires=[
        'numpy==1.23.3',
        'pandas==1.1.1',
        'scikit-learn==0.23.2',
        'requests==2.28.0',
        'plotly==4.9.0',
        'matplotlib==3.3.1',
        'seaborn==0.11.0',
        'keplergl==0.2.2',
        'setuptools==61.3.0',
        'pytest==6.2.5'
    ],
    description="Alerter les stations Vcub qui sont hors service",
    url="https://github.com/armgilles/vcub_keeper",
    author='GILLES Armand',
    author_email='armand.gilles@sub-data.fr',
    license='MIT',
)
