from setuptools import find_packages, setup

setup(
    name='vcub_keeper',
    packages=find_packages(),
    version='1.0',
    install_requires=[
        'pandas==1.1.1',
        'scikit-learn==0.23.2',
        'requests==2.24.0',
        'plotly==4.9.0',
        'matplotlib==3.3.1',
        'seaborn==0.11.0'
    ],
    description="Alerter les stations Vcub qui sont hors service",
    url="https://github.com/armgilles/vcub_keeper",
    author='GILLES Armand',
    author_email='armand.gilles@sub-data.fr',
    license='MIT',
)
