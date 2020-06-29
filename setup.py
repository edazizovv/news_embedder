#
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
            'version': '0.2.1.1',
            'long_description': '',
            'python_requires': '<3.9',
            'install_requires': ['mkl', 'numpy', 'pandas']}


setup(**metadata)


