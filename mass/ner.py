import pandas
import subprocess


def flair_ner_cell(data, config, *args):
    data.to_excel((config.paths.store + config.paths.opened), index=False)
    subprocess.call([config.virtual.flair, './mass/low_level/ner_flair.py'])
    data = pandas.read_excel((config.paths.store + config.paths.closed))
    return data
