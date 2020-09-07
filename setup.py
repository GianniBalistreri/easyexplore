import setuptools
import subprocess
import sys

from easyexplore.text_miner import LANG_MODELS

# Install complete dask library for handling big data sets using parallel computing:
subprocess.run(['python{} -m pip install "dask[complete]"'.format('3' if sys.platform.find('win') != 0 else '')], shell=True)

# Install jupyter notebook extensions for using EasyExplore_examples.ipynb more conveniently:
subprocess.run(['python{} -m pip install jupyter_contrib_nbextensions && jupyter contrib nbextension install'.format('3' if sys.platform.find('win') != 0 else '')], shell=True)

# Install spacy language models:
subprocess.run('python{} -m pip install spacy'.format('3' if sys.platform.find('win') != 0 else ''), shell=True)
for lang in LANG_MODELS.keys():
    for model in LANG_MODELS[lang]['model']['spacy'].keys():
        subprocess.run('python{} -m spacy download {}'.format('3' if sys.platform.find('win') != 0 else '',
                                                              LANG_MODELS[lang]['model']['spacy'][model]
                                                              ),
                       shell=True)

with open('README.md', 'r') as _read_me:
    long_description = _read_me.read()

with open('requirements.txt', 'r') as _requirements:
    requires = _requirements.read()

requires = [r.strip() for r in requires.split('\n') if ((r.strip()[0] != "#") and (len(r.strip()) > 3) and "-e git://" not in r)]

setuptools.setup(
    name='easyexplore',
    version='0.4.2',
    author='Gianni Francesco Balistreri',
    author_email='gbalistreri@gmx.de',
    description='Toolbox for easy and effective data exploration',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords='data exploration interactive visualization machine learning text mining nlp',
    license='GNU',
    url='https://github.com/GianniBalistreri/easyexplore',
    include_package_data=True,
    packages=setuptools.find_packages(),
    package_data={'easyexplore': ['LICENSE',
                                  'README.md',
                                  'requirements.txt',
                                  'setup.py',
                                  'EasyExplore_examples.ipynb'
                                  ]
                  },
    data_file=[('test', ['test/test_anomaly_detector.py',
                         'test/test_data.csv',
                         'test/test_data_explorer.py',
                         'test/test_data_import_export.py',
                         'test/test_data_visualizer.py',
                         'test/test_interactive_visualizer.py',
                         'test/test_text_miner.py',
                         'test/test_utils.py'
                         ]
                )],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=requires
)
