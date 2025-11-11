from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_desc = f.read()
    
setup(
    name="conda_torch",
    version="0.1.0",
    description="Conda optimizer implemented in PyTorch",
    author="Junjie Wang",
    author_email="wangjunjie25@stu.pku.edu.cn",
    packages=find_packages(include=["conda_torch", "conda_torch.*"]),
    python_requires=">=3.8",
)