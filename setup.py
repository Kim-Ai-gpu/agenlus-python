from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="agenlus-hub",
    version="0.2.0",
    description="Python client for Agenlus model upload pipeline.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Agenlus Team",
    url="https://github.com/Kim-Ai-gpu/agenlus-python",
    project_urls={
        "Bug Tracker": "https://github.com/Kim-Ai-gpu/agenlus-python/issues",
        "Source Code": "https://github.com/Kim-Ai-gpu/agenlus-python",
    },
    packages=find_packages(),
    install_requires=[
        "requests",
        "torch",
        "onnx",
        "onnxscript",
        "huggingface_hub"
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
