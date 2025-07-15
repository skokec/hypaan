import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name='hypaan',
    version='0.2',
    scripts=['bin/hypaan'],
    author="Domen Tabernik",
    author_email="domen.tabernik@fri.uni-lj.si",
    description="A Hyper-Parameter Analyzer tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/skokec/hypaan",
    packages=setuptools.find_packages(),
    install_requires=required,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
