from setuptools import setup, find_packages

setup(
    name="hilbert-causal-discovery",
    version="1.0.0",
    author="Shiqian Zhu",
    author_email="souhu2013@gmail.com",
    description="Causal discovery using distance correlation and MDL",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/souhu2013/hilbert-causal-discovery",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
    ],
)
