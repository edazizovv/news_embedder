import os
import sys
import json
import subprocess
import setuptools
from setuptools import setup


metadata = {'name': 'news_embedder',
            'maintainer': 'Edward Aziz',
            'maintainer_email': 'edazizovv@gmail.com',
            'description': 'A naive wrapper for some NLP tools & packages',
            'license': 'MIT',
            'url': 'https://github.com/redjerdai/news_embedder',
            'download_url': 'https://github.com/redjerdai/news_embedder',
            'packages': setuptools.find_packages(),
            'include_package_data': True,
            'version': '0.1.1.12',
            'long_description': '',
            'python_requires': '<3.8',
            'install_requires': ['numpy==1.18.1', 'xlrd', 'openpyxl', 'pandas', 'mkl']}


setup(**metadata)


