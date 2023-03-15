from setuptools import setup, find_packages

setup(
    name="QubitSim2",
    version="1.0",
    author="Chenyin Sun",
    author_email="sunchenyin@mail.ustc.edu.cn",
    description="Qubit Simulation",
    packages=["QubitSim2"],
    install_requires=['numpy', 'progressbar', 'matplotlib', 'scipy']
)
