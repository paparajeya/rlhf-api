from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="RLHF Library",
    version="0.1.0",
    author="Dr Prashant Aparajeya",
    author_email="p.aparajeya@aisimply.uk",
    description="A comprehensive RLHF library for training language models with human feedback",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/paparajeya/rlhf-api",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "gpu": [
            "torch>=1.9.0",
            "torchvision>=0.10.0",
            "torchaudio>=0.9.0",
        ],
    },
    include_package_data=True,
    package_data={
        "rlhf": ["py.typed"],
    },
    entry_points={
        "console_scripts": [
            "rlhf-train=rlhf.cli.train:main",
            "rlhf-evaluate=rlhf.cli.evaluate:main",
            "rlhf-collect=rlhf.cli.collect:main",
        ],
    },
) 