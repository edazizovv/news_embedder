import os
import sys
import json
import subprocess
from setuptools import setup


metadata = {'name': 'news_embedder',
            'maintainer': 'Edward Aziz',
            'maintainer_email': 'edazizovv@gmail.com',
            'description': 'A naive wrapper for some NLP tools & packages',
            'license': 'mit',
            'url': 'https://github.com/redjerdai/news_embedder',
            'download_url': 'https://github.com/redjerdai/news_embedder',
            'project_urls': {},
            'version': '0.1.0',
            'long_description': '',
            'classifiers': [],
            'python_requires': '>=3.7',
            'install_requires': ['xlrd', 'openpyxl', 'pandas']}


setup(**metadata)

tools_env = os.path.dirname(__file__) + 'environments\\env\\hazard_'
tools_dir = os.path.dirname(__file__) + 'environments\\requirements\\hazard_'


tools = ['deeppavlov', 'flair', 'nltk', 'pattern', 'sister', 'spacy', 'textblob', 'use']

virtual = {}
for tool in tools:
    virtual[tool] = tools_env + tool
    subprocess.call([sys.executable, '-m', 'venv', (tools_env + tool)])
    requirements_txt = tools_dir + tool + '.txt'
    python_exe = tools_env + tool + '\\Scripts\\python.exe'
    subprocess.call([python_exe, '-m', 'pip', 'install', '-r', (tools_dir + requirements_txt)])

data = {"text": "Text",
        "opened": "./data/source.xlsx",
        "closed": "./data/gained.xlsx"}

adds = {}

with open(os.path.join(os.path.dirname(__file__), 'settings\\virtual.json'), 'w') as js:
    json.dump(virtual, js)
with open(os.path.join(os.path.dirname(__file__), 'settings\\virtual.json'), 'w') as js:
    json.dump(data, js)
with open(os.path.join(os.path.dirname(__file__), 'settings\\virtual.json'), 'w') as js:
    json.dump(adds, js)
