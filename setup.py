from setuptools import setup, find_packages

setup(
    name='dv4',
    version='0.1.0',
    author='Peter Norman',
    author_email='research@twoswans.com.au',
    description='DV4: Topic-Conditional Weight Reinterpretation via Ternary Encoding with Flip Bits',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/StoneColdLlama/dv4',
    packages=find_packages(),
    python_requires='>=3.12',
    install_requires=[
        'torch>=2.0',
        'transformers>=4.0',
        'datasets>=2.0',
        'huggingface_hub',
    ],
)
