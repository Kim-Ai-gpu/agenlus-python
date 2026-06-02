from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="agenlus-hub",
    version="0.1.4",
    description="Python client for Agenlus model upload pipeline.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Agenlus Team",
    packages=find_packages(),
    install_requires=[
        "requests",
        "torch",
        "onnx",
        "onnxscript",
        "huggingface_hub"
    ],
    python_requires=">=3.7",
)
