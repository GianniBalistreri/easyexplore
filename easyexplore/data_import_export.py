import csv
import json
import os
import pandas as pd
import pickle
import sqlite3
import zipfile

from sqlalchemy import create_engine
from typing import List


class FileUtilsException(Exception):
    """
    Class for handling exceptions for class FileUtils
    """
    pass


class FileUtils:
    """
    Class for handling files
    """
    def __init__(self, file_path: str, create_dir: bool = True):
        """
        :param file_path: str
            File path

        :param create_dir
            Boolean indicating whether to create directories if they are not existed
        """
        if len(file_path) == 0:
            raise FileUtilsException('No file path found')
        self.full_path: str = file_path.replace('\\', '/')
        self.file_name: str = self.full_path.split('/')[-1]
        self.file_path: str = self.full_path.replace(self.file_name, '')
        _file_type = self.file_name.split('.')
        if len(_file_type) > 0:
            self.file_type = _file_type[len(_file_type) - 1]
        else:
            self.file_type = None
        self.create_dir = create_dir
        if self.create_dir:
            if self.full_path.find('/') >= 0:
                self.make_dir()

    def make_dir(self, other_dir: str = None):
        """
        Create directory if it not exists

        :param other_dir: str
            Name of the additional directory to create
        """
        if not os.path.exists(self.file_path):
            try:
                os.mkdir(path=self.file_path)
            except FileNotFoundError:
                raise FileUtilsException('Invalid file path ({})'.format(self.file_path))
        if other_dir is not None:
            if len(other_dir) > 0:
                os.mkdir(path=other_dir)

    def kill(self):
        """
        Kill a file if it exists
        """
        if os.path.isfile(self.full_path):
            os.remove(self.full_path)
        else:
            raise FileUtilsException('File ({}) not exists in directory ({}) !'.format(self.file_name, self.full_path))


class DataImporter(FileUtils):
    """
    Class for import data from external file types
    """
    def __init__(self,
                 file_path: str,
                 as_data_frame: bool = True,
                 create_dir: bool = True,
                 sep: str = ',',
                 **kwargs: dict
                 ):
        """
        :param file_path: str
            String containing the file path

        :param as_data_frame: bool
            Import data set as pandas data frame or not

        :param create_dir: bool
            Boolean indicating whether to create directories if they are not existed

        :param sep: str
            File separator

        :param kwargs: dict
            Additional key word arguments
        """
        super().__init__(file_path=file_path, create_dir=create_dir)
        self.as_df: bool = as_data_frame
        self.sep: str = sep
        self.user_kwargs = kwargs
        self.kwargs = None
        self._config_args()

    def _config_args(self):
        """
        Set configuration setting for the data import as Pandas DataFrame
        """
        self.kwargs = {'filepath': self.full_path,
                       'sep': self.sep,
                       'decimal': '.',
                       'header': 0,
                       'encoding': 'utf-8',
                       'skip_blank_lines': True,
                       'na_values': None,
                       'keep_default_na': True,
                       'parse_dates': True,
                       'quotechar': '|',
                       'quoting': csv.QUOTE_NONE,
                       'doublequote': False,
                       'sheet_name': 0,
                       'names': None,
                       'index_col': None,
                       'usecols': None,
                       'squeeze': False,
                       'prefix': None,
                       'mangle_dup_cols': True,
                       'dtype': None,
                       'engine': None,
                       'converters': None,
                       'true_values': None,
                       'false_values': None,
                       'skipinitialspace': False,
                       'skiprows': None,
                       'skipfooter': 0,
                       'nrows': None,
                       'na_filter': True,
                       'verbose': False,
                       'infer_datetime_format': True,
                       'keep_date_col': False,
                       'date_parser': None,
                       'dayfirst': False,
                       'iterator': False,
                       'chunksize': None,
                       'compression': 'infer',
                       'thousands': None,
                       'float_precision': None,
                       'lineterminator': None,
                       'escapechar': None,
                       'comment': None,
                       'dialect': None,
                       'error_bad_lines': False,
                       'warn_bad_lines': True,
                       'low_memory': True,
                       'memory_map': False
                       }
        if self.user_kwargs is not None:
            for kwarg, value in self.user_kwargs.items():
                if kwarg in self.kwargs.keys():
                    self.kwargs[kwarg] = value
                else:
                    print('Key word argument ({}) not supported !'.format(kwarg))

    def _excel_as_df(self) -> pd.DataFrame:
        """
        Import excel file as Pandas DataFrame

        :return: pd.DataFrame
            Pandas DataFrame containing the content of the html file
        """
        return pd.read_excel(io=self.kwargs.get('filepath'),
                             sheet_name=self.kwargs.get('sheet_name'),
                             header=self.kwargs.get('header'),
                             names=self.kwargs.get('names'),
                             index_col=self.kwargs.get('index_col'),
                             usecols=self.kwargs.get('usecols'),
                             squeeze=self.kwargs.get('squeeze'),
                             prefix=self.kwargs.get('prefix'),
                             #mangle_dupe_cols=self.kwargs.get('mangle_dup_cols'),
                             dtype=self.kwargs.get('dtype'),
                             engine=self.kwargs.get('engine'),
                             converters=self.kwargs.get('converters'),
                             true_values=self.kwargs.get('true_values'),
                             false_values=self.kwargs.get('false_values'),
                             skipinitialspace=self.kwargs.get('skipinitialspace'),
                             skiprows=self.kwargs.get('skiprows'),
                             skipfooter=self.kwargs.get('skipfooter'),
                             nrows=self.kwargs.get('nrows'),
                             na_values=self.kwargs.get('na_values'),
                             keep_default_na=self.kwargs.get('keep_default_na'),
                             na_filter=self.kwargs.get('na_filter'),
                             verbose=self.kwargs.get('verbose'),
                             skip_blank_lines=self.kwargs.get('skip_blank_lines'),
                             parse_dates=self.kwargs.get('parse_dates'),
                             infer_datetime_format=self.kwargs.get('infer_datetime_format'),
                             keep_date_col=self.kwargs.get('keep_date_col'),
                             date_parser=self.kwargs.get('date_parser'),
                             dayfirst=self.kwargs.get('dayfirst'),
                             iterator=self.kwargs.get('iterator'),
                             #chunksize=self.kwargs.get('chunksize'),
                             #compression=self.kwargs.get('compression'),
                             thousands=self.kwargs.get('thousands'),
                             decimal=self.kwargs.get('decimal'),
                             float_precision=self.kwargs.get('float_precision'),
                             lineterminator=self.kwargs.get('lineterminator'),
                             quotechar=self.kwargs.get('quotechar'),
                             quoting=self.kwargs.get('quoting'),
                             doublequote=self.kwargs.get('doublequote'),
                             escapechar=self.kwargs.get('escapechar'),
                             comment=self.kwargs.get('comment'),
                             encoding=self.kwargs.get('encoding'),
                             error_bad_lines=self.kwargs.get('error_bad_lines'),
                             warn_bad_lines=self.kwargs.get('warn_bad_lines'),
                             low_memory=self.kwargs.get('low_memory'),
                             #memory_map=self.kwargs.get('memory_map')
                             )

    def _file(self):
        """
        Import file

        :return: object
            File content
        """
        with open(file=self.file_path,
                  mode='r' if self.kwargs.get('mode') is None else self.kwargs.get('mode'),
                  encoding='utf-8' if self.kwargs.get('encoding') is None else self.kwargs.get('encoding')
                  ) as file:
            return file.read()

    def _html(self):
        """
        Import parsed text content from html file
        """
        raise NotImplementedError('Import data from html file not supported')

    def _html_as_df(self) -> List[pd.DataFrame]:
        """
        Import html file as Pandas DataFrame

        :return: List[pd.DataFrame]
            Contents of the html file as pandas data frames
        """
        return pd.read_html(io=None,
                            match=None,
                            flavor=self.kwargs.get('flavor'),
                            header=self.kwargs.get('header'),
                            index_col=self.kwargs.get('index_col'),
                            skiprows=self.kwargs.get('skiprows'),
                            attrs=self.kwargs.get('attrs'),
                            parse_dates=self.kwargs.get('parse_dates'),
                            #tuplesize_cols=self.kwargs.get('tupleize_cols'),
                            thousands=self.kwargs.get('thousands'),
                            encoding=self.kwargs.get('encoding'),
                            decimal=self.kwargs.get('decimal'),
                            converters=self.kwargs.get('converters'),
                            na_values=self.kwargs.get('na_values'),
                            keep_default_na=self.kwargs.get('keep_default_na'),
                            displayed_only=self.kwargs.get('displayed_only')
                            )

    def _json(self) -> json.load:
        """
        Import json file

        :return: json.load
            Content of the json file
        """
        with open(file=self.file_path,
                  mode='r' if self.kwargs.get('mode') is None else self.kwargs.get('mode'),
                  encoding='utf-8' if self.kwargs.get('encoding') is None else self.kwargs.get('encoding')
                  ) as json_file:
            return json.load(fp=json_file,
                             cls=self.kwargs.get('cls'),
                             object_hook=self.kwargs.get('object_hook'),
                             parse_float=self.kwargs.get('parse_float'),
                             parse_int=self.kwargs.get('parse_int'),
                             parse_constant=self.kwargs.get('parse_constant'),
                             object_pairs_hook=self.kwargs.get('object_pairs_hook')
                             )

    def _json_as_df(self) -> pd.DataFrame:
        """
        Import json file as Pandas DataFrame

        :return: pd.DataFrame
            Content of the json file
        """
        return pd.read_json(path_or_buf=self.kwargs.get('filepath'),
                            orient=None,
                            typ='frame',
                            dtype=True,
                            convert_axes=True,
                            convert_dates=True,
                            keep_default_dates=True,
                            numpy=False,
                            precise_float=False,
                            date_unit=None,
                            encoding=None,
                            lines=False,
                            chunksize=None,
                            compression=self.kwargs.get('compression'))

    def _pickle(self) -> pickle.load:
        """
        Import pickle file

        :return: pickle.load
            Content of pickle file
        """
        with open(self.file_path, 'rb') as file:
            return pickle.load(file=file)

    def _pickle_as_df(self) -> pd.DataFrame:
        """
        Import pickle file as Pandas DataFrame

        :return: pd.DataFrame
            Content of the pickle file
        """
        return pd.read_pickle(path=self.full_path, compression=self.kwargs.get('compression'))

    def _text_as_df(self) -> pd.DataFrame:
        """
        Import text file (csv, txt) as Pandas DataFrame

        :return: pd.DataFrame
            Content of the text file
        """
        return pd.read_csv(filepath_or_buffer=self.kwargs.get('filepath'),
                           sep=self.kwargs.get('sep'),
                           header=self.kwargs.get('header'),
                           names=self.kwargs.get('names'),
                           index_col=self.kwargs.get('index_col'),
                           usecols=self.kwargs.get('usecols'),
                           squeeze=self.kwargs.get('squeeze'),
                           prefix=self.kwargs.get('prefix'),
                           mangle_dupe_cols=self.kwargs.get('mangle_dup_cols'),
                           dtype=self.kwargs.get('dtype'),
                           engine=self.kwargs.get('engine'),
                           converters=self.kwargs.get('converters'),
                           true_values=self.kwargs.get('true_values'),
                           false_values=self.kwargs.get('false_values'),
                           skipinitialspace=self.kwargs.get('skipinitialspace'),
                           skiprows=self.kwargs.get('skiprows'),
                           skipfooter=self.kwargs.get('skipfooter'),
                           nrows=self.kwargs.get('nrows'),
                           na_values=self.kwargs.get('na_values'),
                           keep_default_na=self.kwargs.get('keep_default_na'),
                           na_filter=self.kwargs.get('na_filter'),
                           verbose=self.kwargs.get('verbose'),
                           skip_blank_lines=self.kwargs.get('skip_blank_lines'),
                           parse_dates=self.kwargs.get('parse_dates'),
                           infer_datetime_format=self.kwargs.get('infer_datetime_format'),
                           keep_date_col=self.kwargs.get('keep_date_col'),
                           date_parser=self.kwargs.get('date_parser'),
                           dayfirst=self.kwargs.get('dayfirst'),
                           iterator=self.kwargs.get('iterator'),
                           chunksize=self.kwargs.get('chunksize'),
                           compression=self.kwargs.get('compression'),
                           thousands=self.kwargs.get('thousands'),
                           decimal=self.kwargs.get('decimal'),
                           float_precision=self.kwargs.get('float_precision'),
                           lineterminator=self.kwargs.get('lineterminator'),
                           quotechar=self.kwargs.get('quotechar'),
                           quoting=self.kwargs.get('quoting'),
                           doublequote=self.kwargs.get('doublequote'),
                           escapechar=self.kwargs.get('escapechar'),
                           comment=self.kwargs.get('comment'),
                           encoding=self.kwargs.get('encoding'),
                           error_bad_lines=self.kwargs.get('error_bad_lines'),
                           warn_bad_lines=self.kwargs.get('warn_bad_lines'),
                           low_memory=self.kwargs.get('low_memory'),
                           memory_map=self.kwargs.get('memory_map')
                           )

    def file(self):
        """
        Import data from file

        :return: object
            File content
        """
        if self.file_type in ['csv', 'tsv', 'txt']:
            return self._text_as_df() if self.as_df else self._file()
        elif self.file_type in ['p', 'pkl', 'pickle']:
            return self._pickle_as_df() if self.as_df else self._pickle()
        elif self.file_type == 'json':
            return self._json_as_df() if self.as_df else self._json()
        elif self.file_type == 'html':
            return self._html_as_df() if self.as_df else self._file()
        elif self.file_type in ['xls', 'xlsx']:
            return self._excel_as_df() if self.as_df else self._file()
        else:
            raise FileUtilsException('File type ({}) not supported'.format(self.file_type))

    def zip(self, files: List[str]) -> dict:
        """
        Import data file from compressed zip collection

        :param files: List[str]
            File to look for in zip file

        :return: dict
            Detected file names and file objects
        """
        _zip_content: dict = {self.file_name: {}}
        _zip = zipfile.ZipFile(file=self.full_path,
                               mode='r' if self.kwargs.get('mode') is None else self.kwargs.get('mode'),
                               compression=zipfile.ZIP_STORED if self.kwargs.get('compression') is None else self.kwargs.get('compression'),
                               allowZip64=True if self.kwargs.get('allowZip64') is None else self.kwargs.get('allowZip64')
                               )
        for file in files:
            try:
                with _zip.open(name=file,
                               mode='r' if self.kwargs.get('mode') is None else self.kwargs.get('mode'),
                               pwd=self.kwargs.get('pwd'),
                               force_zip64=False if self.kwargs.get('force_zip64') is None else self.kwargs.get('force_zip64')
                               ) as uncompressed_file:
                    _zip_content[self.file_name].update({file: uncompressed_file.read()})
            except Exception as e:
                FileUtilsException('Could not open file ({}) because of the following error\n{}'.format(file, e))
        return _zip_content


class DataExporter(FileUtils):
    """
    Class for export data to local files
    """
    def __init__(self,
                 obj,
                 file_path: str,
                 create_dir: bool = True,
                 overwrite: bool = False,
                 **kwargs
                 ):
        """
        :param obj: object
            Object to export

        :param file_path: str
            File path

        :param create_dir: bool
            Whether to create directories if they are not existed

        :param overwrite: bool
            Whether to overwrite an existing file or not

        :param kwargs: dict
            Additional key word arguments
        """
        super().__init__(file_path=file_path, create_dir=create_dir)
        self.obj = obj
        if self.create_dir:
            self.make_dir()
        if not overwrite:
            self._avoid_overwriting()
        self.user_kwargs = kwargs

    def _avoid_overwriting(self):
        """
        Generate file name extension to avoid overwriting of existing files
        """
        _i: int = 1
        while os.path.isfile(self.full_path):
            _i += 1
            if _i <= 2:
                self.full_path = self.full_path.replace('.{}'.format(self.file_type), '({}).{}'.format(_i, self.file_type))
            else:
                self.full_path = self.full_path.replace('({}).{}'.format(_i - 1, self.file_type), '({}).{}'.format(_i, self.file_type))

    def _html(self):
        """
        Export data as json file
        """
        with open(self.full_path, 'w', encoding='utf-8') as file:
            file.write(self.obj)

    def _gitignore(self):
        """
        Export data as .gitignore file
        """
        with open(self.file_path, 'w', encoding='utf-8') as file:
            file.write(self.obj)

    def _json(self):
        """
        Export data as json file
        """
        with open(self.full_path, 'w', encoding='utf-8') as file:
            json.dump(self.obj, file, ensure_ascii=False)

    def _pickle(self):
        """
        Export data as pickle file
        """
        with open(self.full_path, 'wb') as _output:
            pickle.dump(self.obj, _output, pickle.HIGHEST_PROTOCOL)

    def _py(self):
        """
        Export data as python file
        """
        with open(self.full_path, 'w') as file:
            file.write(self.obj)

    def _text(self):
        """
        Export data as text (txt, csv) file
        """
        _txt = open(self.full_path, 'wb')
        _txt.write(self.obj)
        _txt.close()

    def file(self):
        """
        Export data as file object
        """
        if self.file_type in ['csv', 'txt']:
            return self._text()
        elif self.file_type in ['', 'p', 'pkl', 'pickle']:
            return self._pickle()
        elif self.file_type == 'json':
            return self._json()
        elif self.file_type == 'py':
            return self._py()
        elif self.file_type == 'gitignore':
            return self._gitignore()
        else:
            return self._text()


class DBUtilsException(Exception):
    """
    Class for handling exceptions for class DBUtils
    """
    pass


class DBUtils:
    """
    Class for importing / exporting data from / to database
    """
    def __init__(self,
                 df: pd.DataFrame = None,
                 table_name: str = None,
                 database: str = 'sqlite',
                 env_var: List[str] = None,
                 con=None,
                 file_path: str = None
                 ):
        """
        :param df
            Data set

        :param table_name
            Name of the table

        :param con
            Data base connection

        :param database
            Name of the database
                -> sqlite: SQLite3 (local db)
                -> postgresql: Postgres db

        :param file_path
            File path of the local SQLite3 database

        """
        self.df = df
        self.table_name = table_name
        self.con = con
        self.database = database
        self.env_var: List[str] = env_var
        self.file_path: str = file_path

    def _get_creds(self) -> str:
        """
        Get database credentials from environment variables

        :return: str
            Database credentials
        """
        if self.env_var is None:
            return '{}://{}:{}@{}:{}/{}'.format(self.database,
                                                os.environ['DB_USER'],
                                                os.environ['DB_PWD'],
                                                os.environ['DB_HOST'],
                                                os.environ['DB_PORT'],
                                                os.environ['DB_NAME']
                                                )
        else:
            if len(self.env_var) == 1:
                return '{}'.format(os.environ[self.env_var[0]])
            elif len(self.env_var) == 5:
                return '{}://{}:{}@{}:{}/{}'.format(self.database,
                                                    os.environ[self.env_var[0]],
                                                    os.environ[self.env_var[1]],
                                                    os.environ[self.env_var[2]],
                                                    os.environ[self.env_var[3]],
                                                    os.environ[self.env_var[4]]
                                                    )
            else:
                raise DBUtilsException('Environment variables ({}) not supported'.format(self.env_var))

    def create_connection(self):
        """
        Create connection to SQLite3 database

        :return: object
            Database connection
        """
        try:
            if self.database == 'sqlite':
                self.con = sqlite3.connect(self.file_path)
            else:
                self._get_creds()
                self.con = create_engine(self._get_creds())
        except sqlite3.Error as e:
            self.con = None
            print(e)
        finally:
            return self.con

    def get_table(self, query: str = 'SELECT * from ') -> pd.DataFrame:
        """
        Fetch table from SQLite3 database

        :param query: str
            SQL query for fetching data from table

        :return: pd.DataFrmae
            Table data set
        """
        return pd.read_sql_query("{}'{}'".format(query, self.table_name), self.con)

    def update_table(self):
        """
        Update table
        """
        self.df.to_sql(name=self.table_name, con=self.con, if_exists='replace')

    def create_table(self):
        """
        Create table
        """
        self.df.to_sql(name=self.table_name, con=self.con, if_exists='fail')

    def drop_table(self):
        """
        Drop existing table
        """
        cursor = self.con.cursor()
        cursor.execute("DROP TABLE '{}'".format(self.table_name))
