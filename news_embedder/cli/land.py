import os
import site
from conda.cli.python_api import Commands, run_command
import subprocess
import json


def land(wd):

    # deeppavlov

    tdd = os.path.join(wd, 'deeppavlov')
    tddx = os.path.join(tdd, 'python.exe')
    print('DEEPPAVLOV')
    print(tdd)
    print(tddx)

    run_command(Commands.CREATE, "--prefix", tdd, "python=3.7", "conda")
    run_command(Commands.INSTALL, "-c", "anaconda", "numpy", "--prefix", tdd)
    run_command(Commands.INSTALL, "-c", "conda-forge", "cython", "--prefix", tdd)
    subprocess.run([tddx, "-m", "pip", "install", "deeppavlov"])
    subprocess.run([tddx, "-m", "deeppavlov", "install", "ner_ontonotes"])
    subprocess.run([tddx, "-m", "deeppavlov", "install", "ner_ontonotes_bert"])
    subprocess.run([tddx, "-m", "deeppavlov", "install", "ner_ontonotes_bert_mult"])
    subprocess.run([tddx, "-m", "deeppavlov", "install", "ner_conll2003"])
    subprocess.run([tddx, "-m", "deeppavlov", "install", "ner_conll2003_bert"])
    run_command(Commands.INSTALL, "-c", "anaconda", "xlrd", "--prefix", tdd)
    run_command(Commands.INSTALL, "-c", "anaconda", "openpyxl", "--prefix", tdd)
    run_command(Commands.INSTALL, "-c", "anaconda", "pandas", "--prefix", tdd)

    # flair

    tdd = os.path.join(wd, 'flair')
    tddx = os.path.join(tdd, 'python.exe')
    print('FLAIR')
    print(tdd)
    print(tddx)

    run_command(Commands.CREATE, "--prefix", tdd, "conda")
    run_command(Commands.INSTALL, "pytorch", "torchvision", "cpuonly", "-c", "pytorch", "--prefix", tdd)
    subprocess.run([tddx, "-m", "pip", "install", "flair"])
    run_command(Commands.INSTALL, "-c", "anaconda", "xlrd", "--prefix", tdd)
    run_command(Commands.INSTALL, "-c", "anaconda", "openpyxl", "--prefix", tdd)
    run_command(Commands.INSTALL, "-c", "anaconda", "pandas", "--prefix", tdd)

    # nltk

    tdd = os.path.join(wd, 'nltk')
    tddx = os.path.join(tdd, 'python.exe')
    print('NLTK')
    print(tdd)
    print(tddx)

    run_command(Commands.CREATE, "--prefix", tdd, "conda")
    run_command(Commands.INSTALL, "-c", "anaconda", "nltk", "--prefix", tdd)
    run_command(Commands.INSTALL, "-c", "anaconda", "xlrd", "--prefix", tdd)
    run_command(Commands.INSTALL, "-c", "anaconda", "openpyxl", "--prefix", tdd)
    run_command(Commands.INSTALL, "-c", "anaconda", "pandas", "--prefix", tdd)


    # pattern

    tdd = os.path.join(wd, 'pattern')
    tddx = os.path.join(tdd, 'python.exe')
    print('PATTERN')
    print(tdd)
    print(tddx)

    run_command(Commands.CREATE, "--prefix", tdd, "python=3.7", "conda")
    run_command(Commands.INSTALL, "-c", "anaconda", "numpy", "--prefix", tdd)
    run_command(Commands.INSTALL, "-c", "conda-forge", "pattern", "--prefix", tdd)
    run_command(Commands.INSTALL, "-c", "anaconda", "xlrd", "--prefix", tdd)
    run_command(Commands.INSTALL, "-c", "anaconda", "openpyxl", "--prefix", tdd)
    run_command(Commands.INSTALL, "-c", "anaconda", "pandas", "--prefix", tdd)

    # sister

    tdd = os.path.join(wd, 'sister')
    tddx = os.path.join(tdd, 'python.exe')
    print('SISTER')
    print(tdd)
    print(tddx)

    run_command(Commands.CREATE, "--prefix", tdd, "conda")
    subprocess.run([tddx, "-m", "pip", "install", "sister"])
    run_command(Commands.INSTALL, "-c", "anaconda", "xlrd", "--prefix", tdd)
    run_command(Commands.INSTALL, "-c", "anaconda", "openpyxl", "--prefix", tdd)
    run_command(Commands.INSTALL, "-c", "anaconda", "pandas", "--prefix", tdd)

    # spacy

    tdd = os.path.join(wd, 'spacy')
    tddx = os.path.join(tdd, 'python.exe')
    print('SPACY')
    print(tdd)
    print(tddx)

    run_command(Commands.CREATE, "--prefix", tdd, "python=3.7", "conda")
    run_command(Commands.INSTALL, "-c", "anaconda", "numpy", "--prefix", tdd)
    run_command(Commands.INSTALL, "-c", "conda-forge", "spacy", "--prefix", tdd)
    subprocess.run([tddx, "-m", "spacy", "download", "en_core_web_sm"])
    subprocess.run([tddx, "-m", "spacy", "download", "en_core_web_md"])
    subprocess.run([tddx, "-m", "spacy", "download", "en_core_web_lg"])
    run_command(Commands.INSTALL, "-c", "anaconda", "xlrd", "--prefix", tdd)
    run_command(Commands.INSTALL, "-c", "anaconda", "openpyxl", "--prefix", tdd)
    run_command(Commands.INSTALL, "-c", "anaconda", "pandas", "--prefix", tdd)


    # textblob

    tdd = os.path.join(wd, 'textblob')
    tddx = os.path.join(tdd, 'python.exe')
    print('TEXTBLOB')
    print(tdd)
    print(tddx)

    run_command(Commands.CREATE, "--prefix", tdd, "conda")
    run_command(Commands.INSTALL, "-c", "conda-forge", "textblob", "--prefix", tdd)
    run_command(Commands.INSTALL, "-c", "anaconda", "xlrd", "--prefix", tdd)
    run_command(Commands.INSTALL, "-c", "anaconda", "openpyxl", "--prefix", tdd)
    run_command(Commands.INSTALL, "-c", "anaconda", "pandas", "--prefix", tdd)

    # use

    tdd = os.path.join(wd, 'use')
    tddx = os.path.join(tdd, 'python.exe')
    print('USE')
    print(tdd)
    print(tddx)

    run_command(Commands.CREATE, "--prefix", tdd, "python=3.7", "conda")
    run_command(Commands.INSTALL, "-c", "anaconda", "numpy", "--prefix", tdd)
    run_command(Commands.INSTALL, "-c", "conda-forge", "tensorflow", "--prefix", tdd)
    run_command(Commands.INSTALL, "-c", "conda-forge", "tensorflow-hub", "--prefix", tdd)
    run_command(Commands.INSTALL, "-c", "anaconda", "xlrd", "--prefix", tdd)
    run_command(Commands.INSTALL, "-c", "anaconda", "openpyxl", "--prefix", tdd)
    run_command(Commands.INSTALL, "-c", "anaconda", "pandas", "--prefix", tdd)

    #

    td = os.path.join(site.getsitepackages()[1], 'news_embedder/settings')

    virtual_config = {'deeppavlov': os.path.join(wd, 'deeppavlov', 'python.exe'),
                      'flair': os.path.join(wd, 'flair', 'python.exe'),
                      'nltk': os.path.join(wd, 'nltk', 'python.exe'),
                      'pattern': os.path.join(wd, 'pattern', 'python.exe'),
                      'sister': os.path.join(wd, 'sister', 'python.exe'),
                      'spacy': os.path.join(wd, 'spacy', 'python.exe'),
                      'textblob': os.path.join(wd, 'textblob', 'python.exe'),
                      'use': os.path.join(wd, 'use', 'python.exe')}

    with open((os.path.join(td, 'virtual.json')), 'w') as outfile:
        json.dump(virtual_config, outfile)

