import setuptools
import subprocess

# Install jupyter notebook extensions:
subprocess.run('python3 -m pip install jupyter_contrib_nbextensions && jupyter contrib nbextension install')

with open('README.md', 'r') as _read_me:
    long_description = _read_me.read()

setuptools.setup(
    name='easyexplore',
    version='0.0.1',
    author='Gianni Francesco Balistreri',
    author_email='gbalistreri@gmx.de',
    description='Toolbox for easy and effective data exploration',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/pypa/sampleproject',
    packages=setuptools.find_packages(),
    package_data={'easyexplore': ['LICENCE',
                                  'README.md',
                                  'requirements.txt',
                                  'setup.py',
                                  'EasyExplore_examples.ipynb'
                                  ]
                  },
    include_package_data=True,
    scripts=['src/anomaly_detector.py',
             'src/data_explorer.py',
             'src/data_import_export.py',
             'src/data_visualizer.py',
             'src/interactive_visualizer.py',
             'src/utils.py',
             'src/test/test_anomaly_detector.py',
             'src/test/test_data_explorer.py',
             'src/test/test_data_import_export.py',
             'src/test/test_data_visualizer.py',
             'src/test/test_interactive_visualizer.py',
             'src/test/test_utils.py'
             ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: GNU GENERAL PUBLIC LICENSE',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=['geojson>=2.5.0',
                      'ipywidgets>=0.5.1',
                      'joblib>=0.14.1',
                      'networkx>=2.2',
                      'numpy>=1.18.1',
                      'pandas==0.25.3',
                      'plotly>=4.5.4',
                      'pyod>=0.7.7.1',
                      'psutil>=5.5.1',
                      'scipy>=1.4.1',
                      'sqlalchemy>=1.3.15',
                      'statsmodels>=0.9.0',
                      'xlrd>=1.2.0'
                      ]
)
