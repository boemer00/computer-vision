from setuptools import setup, find_packages

setup(
    name='ship-detection',
    version='1.0',
    description='A computer vision project for ship detection using ResNet50 and MLflow',
    author='Renato Boemer',
    author_email='boemer00@-----.com', # to prevent scraping (but use gmail)
    url='https://github.com/boemer00/ship-detection',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch',
        'torchvision',
        'Pillow',
        'mlflow',
        'pytest'
    ],
    entry_points={
        'console_scripts': [
            'train=src.train:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10.6',
)
