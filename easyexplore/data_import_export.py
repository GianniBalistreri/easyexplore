import boto3
import io
import json
import os
import pandas as pd
import pickle
import sqlite3
import zipfile

from dask import dataframe as dd
from google.cloud import storage
from sqlalchemy import create_engine
from typing import List

CLOUD_PROVIDER: List[str] = ['aws', 'google']


class FileUtilsException(Exception):
    """
    Class for handling exceptions for class FileUtils
    """
    pass


class FileUtils:
    """
    Class for handling file utilities
    """
    def __init__(self,
                 file_path: str,
                 create_dir: bool = False,
                 cloud: str = None,
                 bucket_name: str = None,
                 region: str = None
                 ):
        """
        :param file_path: str
            File path

        :param create_dir: bool
            Create directories if they are not existed

        :param cloud: str
            Name of the cloud provider:
                -> google: Google Cloud Storage
                -> aws: AWS Cloud

        :param bucket_name: str
            Name of the bucket of the cloud provider

        :param region: str
            Name of the region (AWS S3-Bucket)
        """
        if len(file_path) == 0:
            raise FileUtilsException('No file path found')
        self.full_path: str = file_path.replace('\\', '/')
        self.file_name: str = self.full_path.split('/')[-1]
        self.file_path: str = self.full_path.replace(self.file_name, '')
        _file_type = self.file_name.split('.')
        if len(_file_type) > 1:
            self.file_type = _file_type[len(_file_type) - 1]
        else:
            self.file_type = None
        self.create_dir = create_dir
        self.cloud: str = cloud
        self.bucket_name: str = bucket_name
        self.region: str = region
        self.aws_s3_file_name: str = None
        self.aws_s3_file_path: str = None
        self.google_cloud_file_name: str = None
        self.google_cloud_file_path: str = None
        if self.cloud is not None:
            if self.cloud not in CLOUD_PROVIDER:
                raise FileUtilsException('Cloud provider ({}) not supported'.format(self.cloud))
        if self.cloud == 'google':
            self.google_cloud_file_name = '/'.join(self.full_path.split('//')[1].split('/')[1:])
            self.google_cloud_file_path = '/'.join(self.full_path.split('//')[1].split('/')[1:-1])
        if self.cloud == 'aws':
            self.aws_s3_file_name = '/'.join(self.full_path.split('//')[1].split('/')[1:])
            self.aws_s3_file_path = '/'.join(self.full_path.split('//')[1].split('/')[1:-1])
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
                 use_dask: bool = True,
                 create_dir: bool = False,
                 sep: str = ',',
                 cloud: str = None,
                 bucket_name: str = None,
                 region: str = None,
                 **kwargs: dict
                 ):
        """
        :param file_path: str
            String containing the file path

        :param as_data_frame: bool
            Import data set as pandas data frame or not

        :param use_dask: bool
            Use dask library for parallel computation

        :param create_dir: bool
            Create directories if they are not existed

        :param sep: str
            File separator

        :param cloud: str
            Name of the cloud provider:
                -> google: Google Cloud Storage
                -> aws: AWS Cloud

        :param bucket_name: str
            Name of the bucket of the cloud provider

        :param region: str
            Name of the region (AWS S3-Bucket)

        :param kwargs: dict
            Additional key word arguments
        """
        super().__init__(file_path=file_path,
                         create_dir=create_dir,
                         cloud=cloud,
                         bucket_name=bucket_name,
                         region=region
                         )
        self.as_df: bool = as_data_frame
        self.use_dask: bool = use_dask
        self.partitions: int = 4 if kwargs.get('npartitions') is None else kwargs.get('npartitions')
        self.sep: str = sep
        self.kwargs: dict = kwargs
        self._config_args()

    def _config_args(self):
        """
        Set configuration setting for the data import as Pandas DataFrame
        """
        self.kwargs.update({'sep': self.sep,
                            'decimal': '.' if self.kwargs.get('decimal') is None else self.kwargs.get('decimal'),
                            'header': 0 if self.kwargs.get('header') is None else self.kwargs.get('header'),
                            'encoding': 'utf-8' if self.kwargs.get('encoding') is None else self.kwargs.get('encoding'),
                            'skip_blank_lines': True if self.kwargs.get('skip_blank_lines') is None else self.kwargs.get('skip_blank_lines'),
                            'na_values': self.kwargs.get('na_values'),
                            'keep_default_na': True if self.kwargs.get('keep_default_na') is None else self.kwargs.get('keep_default_na'),
                            'parse_dates': True if self.kwargs.get('parse_dates') is None else self.kwargs.get('parse_dates'),
                            'quotechar': '"' if self.kwargs.get('quotechar') is None else self.kwargs.get('quotechar'),
                            'quoting': 0 if self.kwargs.get('quoting') is None else self.kwargs.get('quotechar'), #csv.QUOTE_NONE,
                            'doublequote': True if self.kwargs.get('doublequote') is None else self.kwargs.get('doublequote'),
                            'sheet_name': 0 if self.kwargs.get('sheet_name') is None else self.kwargs.get('sheet_name'),
                            'names': self.kwargs.get('names'),
                            'index_col': self.kwargs.get('index_col'),
                            'usecols': self.kwargs.get('usecols'),
                            'squeeze': False if self.kwargs.get('squeeze') is None else self.kwargs.get('squeeze'),
                            'prefix': self.kwargs.get('prefix'),
                            'mangle_dup_cols': True if self.kwargs.get('mangle_dup_cols') is None else self.kwargs.get('mangle_dup_cols'),
                            'dtype': self.kwargs.get('dtype'),
                            'engine': self.kwargs.get('engine'),
                            'converters': self.kwargs.get('converters'),
                            'true_values': self.kwargs.get('true_values'),
                            'false_values': self.kwargs.get('false_values'),
                            'skipinitialspace': False if self.kwargs.get('skipinitialspace') is None else self.kwargs.get('skipinitialspace'),
                            'skiprows': self.kwargs.get('skiprows'),
                            'skipfooter': 0 if self.kwargs.get('skipfooter') is None else self.kwargs.get('skipfooter'),
                            'nrows': self.kwargs.get('nrows'),
                            'na_filter': True if self.kwargs.get('na_filter') is None else self.kwargs.get('na_filter'),
                            'verbose': False if self.kwargs.get('verbose') is None else self.kwargs.get('verbose'),
                            'infer_datetime_format': True if self.kwargs.get('infer_datetime_format') is None else self.kwargs.get('infer_datetime_format'),
                            'keep_date_col': False if self.kwargs.get('keep_date_col') is None else self.kwargs.get('keep_date_col'),
                            'date_parser': None if self.kwargs.get('date_parser') is None else self.kwargs.get('date_parser'),
                            'dayfirst': False if self.kwargs.get('dayfirst') is None else self.kwargs.get('dayfirst'),
                            'iterator': False if self.kwargs.get('iterator') is None else self.kwargs.get('iterator'),
                            'chunksize': self.kwargs.get('chunksize'),
                            'compression': 'infer' if self.kwargs.get('compression') is None else self.kwargs.get('compression'),
                            'thousands': self.kwargs.get('thousands'),
                            'float_precision': self.kwargs.get('float_precision'),
                            'lineterminator': self.kwargs.get('lineterminator'),
                            'escapechar': self.kwargs.get('escapechar'),
                            'comment': self.kwargs.get('comment'),
                            'dialect': self.kwargs.get('dialect'),
                            'error_bad_lines': False if self.kwargs.get('error_bad_lines') is None else self.kwargs.get('error_bad_lines'),
                            'warn_bad_lines': True if self.kwargs.get('warn_bad_lines') is None else self.kwargs.get('warn_bad_lines'),
                            'low_memory': True if self.kwargs.get('low_memory') is None else self.kwargs.get('low_memory'),
                            'memory_map': False if self.kwargs.get('memory_map') is None else self.kwargs.get('memory_map')
                            })

    def _excel_as_df(self):
        """
        Import excel file as Pandas DataFrame

        :return: pd.DataFrame
            Pandas DataFrame containing the content of the html file
        """
        if self.use_dask:
            return dd.from_pandas(data=pd.read_excel(io=self.full_path,
                                                     sheet_name=self.kwargs.get('sheet_name'),
                                                     header=self.kwargs.get('header'),
                                                     names=self.kwargs.get('names'),
                                                     index_col=self.kwargs.get('index_col'),
                                                     usecols=self.kwargs.get('usecols'),
                                                     squeeze=self.kwargs.get('squeeze'),
                                                     dtype=self.kwargs.get('dtype'),
                                                     engine=self.kwargs.get('engine'),
                                                     converters=self.kwargs.get('converters'),
                                                     true_values=self.kwargs.get('true_values'),
                                                     false_values=self.kwargs.get('false_values'),
                                                     skiprows=self.kwargs.get('skiprows'),
                                                     skipfooter=self.kwargs.get('skipfooter'),
                                                     nrows=self.kwargs.get('nrows'),
                                                     na_values=self.kwargs.get('na_values'),
                                                     keep_default_na=self.kwargs.get('keep_default_na'),
                                                     na_filter=self.kwargs.get('na_filter'),
                                                     verbose=self.kwargs.get('verbose'),
                                                     parse_dates=self.kwargs.get('parse_dates'),
                                                     date_parser=self.kwargs.get('date_parser'),
                                                     thousands=self.kwargs.get('thousands'),
                                                     comment=self.kwargs.get('comment')
                                                     ),
                                  npartitions=self.partitions,
                                  chunksize=self.kwargs.get('chunksize'),
                                  sort=True if self.kwargs.get('sort') is None else self.kwargs.get('sort'),
                                  name=self.kwargs.get('name')
                                  )
        return pd.read_excel(io=self.full_path,
                             sheet_name=self.kwargs.get('sheet_name'),
                             header=self.kwargs.get('header'),
                             names=self.kwargs.get('names'),
                             index_col=self.kwargs.get('index_col'),
                             usecols=self.kwargs.get('usecols'),
                             squeeze=self.kwargs.get('squeeze'),
                             prefix=self.kwargs.get('prefix'),
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
                             )

    def _file(self):
        """
        Import file

        :return: object
            File content
        """
        with open(file=self.full_path,
                  mode='r' if self.kwargs.get('mode') is None else self.kwargs.get('mode'),
                  encoding='utf-8' if self.kwargs.get('encoding') is None else self.kwargs.get('encoding')
                  ) as file:
            return file.read()

    def _google_cloud_storage(self):
        """
        Download files from Google Cloud Storage.
        """
        _client = storage.Client()
        _bucket = _client.get_bucket(bucket_or_name=self.bucket_name)
        _blob = _bucket.blob(blob_name=self.google_cloud_file_name)
        _blob.download_to_filename(filename=self.google_cloud_file_name.split('/')[-1])

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
        with open(file=self.full_path,
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

    def _json_as_df(self):
        """
        Import json file as Pandas DataFrame

        :return: Pandas DataFrame or dask dataframe
            Content of the json file
        """
        if self.use_dask:
            return dd.read_json(url_path=self.full_path,
                                orient='records' if self.kwargs.get('orient') is None else self.kwargs.get('orient'),
                                lines=self.kwargs.get('lines'),
                                storage_options=self.kwargs.get('storage_options'),
                                blocksize=self.kwargs.get('blocksize'),
                                sample=2 ** 20 if self.kwargs.get('sample') is None else self.kwargs.get('sample'),
                                encoding='utf-8' if self.kwargs.get('encoding') is None else self.kwargs.get('encoding'),
                                errors='strict' if self.kwargs.get('errors') is None else self.kwargs.get('errors'),
                                compression='infer' if self.kwargs.get('compression') is None else self.kwargs.get('compression'),
                                meta=self.kwargs.get('meta'),
                                engine=pd.read_json
                                )
        return pd.read_json(path_or_buf=self.full_path,
                            orient='records' if self.kwargs.get('orient') is None else self.kwargs.get('orient'),
                            typ='frame',
                            dtype=True if self.kwargs.get('dtype') is None else self.kwargs.get('dtype'),
                            convert_axes=True if self.kwargs.get('convert_axes') is None else self.kwargs.get('convert_axes'),
                            convert_dates=True if self.kwargs.get('convert_dates') is None else self.kwargs.get('convert_dates'),
                            keep_default_dates=True if self.kwargs.get('keep_default_dates') is None else self.kwargs.get('keep_default_dates'),
                            numpy=False if self.kwargs.get('numpy') is None else self.kwargs.get('numpy'),
                            precise_float=False if self.kwargs.get('precise_float') is None else self.kwargs.get('precise_float'),
                            date_unit=self.kwargs.get('date_unit'),
                            encoding='utf-8' if self.kwargs.get('encoding') is None else self.kwargs.get('encoding'),
                            lines=False if self.kwargs.get('lines') is None else self.kwargs.get('lines'),
                            chunksize=self.kwargs.get('chunksize'),
                            compression=self.kwargs.get('compression')
                            )

    def _parquet(self):
        """
        Import parquet file

        :return dask DataFrame
        """
        return dd.read_parquet(path=self.full_path,
                               columns=None,
                               filters=self.kwargs.get('filters'),
                               categories=self.kwargs.get('categories'),
                               index=self.kwargs.get('index'),
                               storage_options=self.kwargs.get('storage_options'),
                               engine='pyarrow',
                               gather_statistics=self.kwargs.get('gather_statistics'),
                               split_row_groups=self.kwargs.get('split_row_groups'),
                               chunksize=self.kwargs.get('chunksize')
                               )

    def _pickle(self) -> pickle.load:
        """
        Import pickle file

        :return: pickle.load
            Content of pickle file
        """
        if self.cloud is None:
            with open(self.full_path, 'rb') as file:
                return pickle.load(file=file)
        elif self.cloud == 'google':
            self._google_cloud_storage()
            with open(self.google_cloud_file_name.split('/')[-1], 'rb') as file:
                return pickle.load(file=file)
        elif self.cloud == 'aws':
            raise NotImplementedError('AWS not supported yet')

    def _pickle_as_df(self):
        """
        Import pickle file as Pandas DataFrame

        :return: Pandas DataFrame or dask dataframe
            Content of the pickle file
        """
        if self.use_dask:
            return dd.from_pandas(data=pd.read_pickle(filepath_or_buffer=self.full_path,
                                                      compression=self.kwargs.get('compression')
                                                      ),
                                  npartitions=self.partitions,
                                  chunksize=self.kwargs.get('chunksize'),
                                  sort=True if self.kwargs.get('sort') is None else self.kwargs.get('sort'),
                                  name=self.kwargs.get('name')
                                  )
        return pd.read_pickle(filepath_or_buffer=self.full_path, compression=self.kwargs.get('compression'))

    def _text_as_df(self):
        """
        Import text file (csv, tsv, txt) as Pandas or dask DataFrame

        :return: Pandas DataFrame or dask dataframe
            Content of the text file
        """
        if self.use_dask:
            return dd.read_csv(urlpath=self.full_path,
                               blocksize='default' if self.partitions > 1 else self.kwargs.get('blocksize'),
                               lineterminator=self.kwargs.get('lineterminator'),
                               #compression=self.kwargs.get('compression'),
                               sample=256000 if self.kwargs.get('sample') is None else self.kwargs.get('sample'),
                               enforce=False if self.kwargs.get('enforce') is None else self.kwargs.get('enforce'),
                               assume_missing=True if self.kwargs.get('assume_missing') is None else self.kwargs.get('assume_missing'),
                               storage_options=self.kwargs.get('storage_options'),
                               include_path_column=False if self.kwargs.get('include_path_column') is None else self.kwargs.get('include_path_column'),
                               dtype=str if self.kwargs.get('dtype') is None else self.kwargs.get('dtype')
                               )
        return pd.read_csv(filepath_or_buffer=self.full_path,
                           sep=self.kwargs.get('sep'),
                           header=self.kwargs.get('header'),
                           names=self.kwargs.get('names'),
                           index_col=self.kwargs.get('index_col'),
                           usecols=self.kwargs.get('usecols'),
                           squeeze=self.kwargs.get('squeeze'),
                           prefix=self.kwargs.get('prefix'),
                           mangle_dupe_cols=self.kwargs.get('mangle_dup_cols'),
                           dtype=str if self.kwargs.get('dtype') is None else self.kwargs.get('dtype'),
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

    def file(self, table_name: str = None):
        """
        Import data from file

        :param table_name: str
            Name of the table of the local database file

        :return: object
            File content
        """
        if self.file_type is None:
            return self._parquet()
        elif self.file_type in ['csv', 'tsv', 'txt']:
            return self._text_as_df() if self.as_df else self._file()
        elif self.file_type in ['p', 'pkl', 'pickle']:
            return self._pickle_as_df() if self.as_df else self._pickle()
        elif self.file_type == 'json':
            return self._json_as_df() if self.as_df else self._json()
        elif self.file_type == 'html':
            return self._html_as_df() if self.as_df else self._file()
        elif self.file_type in ['xls', 'xlsx']:
            return self._excel_as_df() if self.as_df else self._file()
        elif self.file_type == 'db':
            _con = DBUtils(table_name=table_name, file_path=self.full_path)
            _con.create_connection()
            try:
                if self.use_dask:
                    _df: dd.DataFrame = dd.from_pandas(data=_con.get_table(), npartitions=self.partitions)
                    #_df: dd.DataFrame = dd.read_sql_table(table=table_name, uri=self.full_path, index_col='', divisions=self.partitions)
                else:
                    _df: pd.DataFrame = _con.get_table()
            except sqlite3.Error as e:
                _df: pd.DataFrame = pd.DataFrame()
                print(e)
            finally:
                _con.con.close()
            return _df
        elif self.file_type == 'parquet':
            return self._parquet()
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
                 create_dir: bool = False,
                 overwrite: bool = False,
                 cloud: str = None,
                 bucket_name: str = None,
                 region: str = None,
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

        :param cloud: str
            Name of the cloud provider:
                -> google: Google Cloud Storage
                -> aws: AWS Cloud

        :param bucket_name: str
            Name of the bucket of the cloud provider

        :param region: str
            Name of the region (AWS S3-Bucket)

        :param kwargs: dict
            Additional key word arguments
        """
        super().__init__(file_path=file_path,
                         create_dir=create_dir,
                         cloud=cloud,
                         bucket_name=bucket_name,
                         region=region
                         )
        self.obj = obj
        if self.create_dir:
            self.make_dir()
        if not overwrite:
            self._avoid_overwriting()
        self.kwargs = kwargs

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

    def _aws_s3(self, buffer: io.BytesIO):
        """
        Upload files to AWS S3 bucket

        :param buffer: io.BytesIO
            Object bytes
        """
        _aws_s3_client = boto3.client('s3', region_name=self.region)
        _aws_s3_client.put_object(Body=buffer.getvalue(), Bucket=self.bucket_name, Key=self.aws_s3_file_name)

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
        with open(self.full_path, 'w', encoding='utf-8') as file:
            file.write(self.obj)

    def _google_cloud_storage(self):
        """
        Upload files to Google Cloud Storage.
        """
        _client = storage.Client()
        _bucket = _client.get_bucket(bucket_or_name=self.bucket_name)
        _blob = _bucket.blob(blob_name=self.google_cloud_file_name)
        _blob.upload_from_filename(filename=self.google_cloud_file_name)

    def _json(self):
        """
        Export data as json file
        """
        with open(self.full_path, 'w', encoding='utf-8') as file:
            json.dump(obj=self.obj, fp=file, ensure_ascii=False)

    def _parquet(self):
        """
        Export data as parquet file
        """
        dd.to_parquet(df=self.obj,
                      path=self.full_path,
                      engine='auto',
                      compression='default' if self.kwargs.get('compression') is None else self.kwargs.get('compression'),
                      write_index=True if self.kwargs.get('write_index') is None else self.kwargs.get('write_index'),
                      append=False if self.kwargs.get('append') is None else self.kwargs.get('append'),
                      ignore_divisions=False if self.kwargs.get('ignore_divisions') is None else self.kwargs.get('ignore_divisions'),
                      partition_on=self.kwargs.get('partition_on'),
                      storage_options=self.kwargs.get('storage_options'),
                      write_metadata_file=True if self.kwargs.get('write_metadata_file') is None else self.kwargs.get('write_metadata_file'),
                      compute=True if self.kwargs.get('compute') is None else self.kwargs.get('compute'),
                      compute_kwargs=self.kwargs.get('compute_kwargs'),
                      schema='infer' if self.kwargs.get('schema') is None else self.kwargs.get('schema')
                      )

    def _pickle(self):
        """
        Export data as pickle file
        """
        if self.cloud is None:
            with open(self.full_path, 'wb') as _output:
                pickle.dump(self.obj, _output, pickle.HIGHEST_PROTOCOL)
        elif self.cloud == 'aws':
            _buffer: io.BytesIO = io.BytesIO()
            pickle.dump(obj=self.obj, file=_buffer, protocol=pickle.HIGHEST_PROTOCOL)
            self._aws_s3(buffer=_buffer)
        elif self.cloud == 'google':
            if not os.path.exists(self.google_cloud_file_path):
                os.makedirs(name=self.google_cloud_file_path, exist_ok=True)
            with open(self.google_cloud_file_name, 'wb') as _output:
                pickle.dump(obj=self.obj, file=_output, protocol=pickle.HIGHEST_PROTOCOL)
            self._google_cloud_storage()

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

    def _text_from_df(self):
        """
        Export data as text file from data frame
        """
        self.obj.to_csv(path_or_buf=self.full_path,
                        sep=self.kwargs.get('sep'),
                        na_rep="" if self.kwargs.get('na_rep') is None else self.kwargs.get('na_rep'),
                        float_format=self.kwargs.get('float_format'),
                        columns=self.kwargs.get('columns'),
                        header=True if self.kwargs.get('header') is None else self.kwargs.get('header'),
                        index=False if self.kwargs.get('index') is None else self.kwargs.get('index'),
                        index_label=self.kwargs.get('index_label'),
                        mode='w',
                        encoding=self.kwargs.get('encoding'),
                        compression='infer' if self.kwargs.get('compression') is None else self.kwargs.get('compression'),
                        quoting=self.kwargs.get('quoting'),
                        chunksize=self.kwargs.get('chunksize'),
                        date_format=self.kwargs.get('date_format'),
                        doublequote=True if self.kwargs.get('doublequote') is None else self.kwargs.get('doublequote'),
                        escapechar=self.kwargs.get('escapechar'),
                        decimal='.' if self.kwargs.get('decimal') is None else self.kwargs.get('decimal')
                        )

    def file(self):
        """
        Export data as file object
        """
        if self.file_type is None:
            return self._parquet()
        elif self.file_type in ['csv', 'tsv', 'txt']:
            if isinstance(self.obj, pd.DataFrame) or isinstance(self.obj, dd.DataFrame):
                return self._text_from_df()
            else:
                return self._text()
        elif self.file_type in ['', 'p', 'pkl', 'pickle']:
            return self._pickle()
        elif self.file_type == 'json':
            return self._json()
        elif self.file_type == 'py':
            return self._py()
        elif self.file_type == 'gitignore':
            return self._gitignore()
        elif self.file_type == 'parquet':
            return self._parquet()
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
