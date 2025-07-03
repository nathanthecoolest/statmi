from setuptools import setup, find_packages

setup(
    name='statmi',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-learn',
        'scipy'
    ],
    author='Nathan Burton',
    description='Mutual information, conditional mutual information, and entropy estimation using binning or KDE.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
