import copy
import dask.dataframe as dd
import numpy as np
import os
import pandas as pd

from .anomaly_detector import AnomalyDetector
from .data_import_export import DataImporter
from .data_visualizer import DataVisualizer
from .utils import Log, PERCENTILES, StatsUtils, EasyExploreUtils
from dask.array import from_array
from dask.distributed import Client
from typing import Dict, List, Tuple

# TODO:
#  Correlation -> Partial + Final Heat Map
#  Data Distribution: Color (overtime) + Annotations (continuous distribution parameter)


class DataExplorerException(Exception):
    """
    Class for handling exceptions for class DataExploration
    """
    pass


class DataExplorer:
    """
    Class for data exploration
    """
    def __init__(self,
                 df=None,
                 file_path: str = '',
                 table_name: str = None,
                 target: str = None,
                 seed: int = 1234,
                 include: List[str] = None,
                 exclude: List[str] = None,
                 ordinal: List[str] = None,
                 date: List[str] = None,
                 id_text: List[str] = None,
                 categorical: List[str] = None,
                 continuous: List[str] = None,
                 date_edges: Tuple[str, str] = None,
                 feature_types: Dict[str, List[str]] = None,
                 include_nan: bool = True,
                 plot: bool = True,
                 plot_type: str = None,
                 output_file_path: str = None,
                 **kwargs: dict
                 ):
        """
        :param df: Pandas DataFrame or dask dataframe
            Data set

        :param file_path: str
            Complete file path of data file

        :param table_name: str
            Name of the table to fetch from local database file (sqlite3)

        :param target: str
            Name of the target variable

        :param include: List[str]
            Name of the features to explore

        :param exclude: List[str]
            Name of the variables to ignore

        :param ordinal: List[str]
            Names of the ordinal features

        :param date: List[str]
            Names of the date features

        :param date_edges: tuple
            Datetime values defining the time frame of valid datetime data

        :param feature_types: Dict[str, List[str]]
            Feature type mapping

        :param plot: bool
            Whether to plot the exploration results or not

        :param plot_type: str
            Name of the plot type

        :param output_file_path: str
            Complete file path for the output

        :param kwargs: dict
            Key-word arguments
        """
        if len(file_path) == 0 and df is None:
            raise DataExplorerException('Neither data set nor file path found')
        self.dask_client: Client = EasyExploreUtils().dask_setup(client_name='explorer',
                                                                 client_address=kwargs.get('client_address'),
                                                                 mode='threads' if kwargs.get('client_mode') is None else kwargs.get('client_mode')
                                                                 )
        self.partitions: int = 4 if kwargs.get('npartitions') is None else kwargs.get('npartitions')
        if len(file_path) > 0:
            if os.path.isfile(file_path):
                self.df: dd.DataFrame = DataImporter(file_path=file_path,
                                                     as_data_frame=True,
                                                     use_dask=True,
                                                     sep=',',
                                                     **kwargs
                                                     ).file(table_name=table_name)
            else:
                raise DataExplorerException('File path ({}) is incomplete'.format(file_path))
        else:
            self.df = df
        if isinstance(df, pd.DataFrame):
            self.df = dd.from_pandas(data=df, npartitions=self.partitions)
            if len(self.df) < 10:
                if len(self.df) == 0:
                    raise DataExplorerException('Data set is empty')
                else:
                    Log(write=False, level='warn').log('Too few cases ({}) in data set'.format(len(self.df)))
            if len(self.df.columns) == 0:
                Log(write=False, level='warn').log('Too few features ({}) in data set'.format(len(self.df.columns)))
        elif isinstance(df, dd.core.DataFrame):
            self.df = df
        else:
            if self.df is None:
                raise DataExplorerException('Format of data set ({}) not supported. Use Pandas DataFrame or dask dataframe instead'.format(type(df)))
        if 'Unnamed: 0' in list(self.df.columns):
            del self.df['Unnamed: 0']
        Log(write=False).log(msg='Data set: {}\nCases: {}\nFeatures: {}'.format(file_path, len(self.df), len(self.df.columns)))
        self.seed: int = seed if seed > 0 else 1234
        if include is None:
            if exclude is not None:
                _exclude: List[str] = []
                for ex in exclude:
                    if ex in self.df.columns:
                        _exclude.append(ex)
                if len(_exclude) > 0:
                    self.df = self.df[_exclude]
        else:
            _include: List[str] = []
            for inc in include:
                if inc in self.df.columns:
                    _include.append(inc)
            if len(_include) > 0:
                self.df = self.df[_include]
        self.features: List[str] = list(self.df.columns)
        self.n_cases: int = len(self.df)
        self.n_features: int = len(self.df.columns)
        self.data_types: list = self.df.dtypes.tolist()
        self.data_index: list = list(self.df.index.values.compute())
        self.date_edges: Tuple[str, str] = date_edges
        self.date: List[str] = [] if date is None else date
        self.ordinal: List[str] = [] if ordinal is None else ordinal
        self.id_text: List[str] = [] if id_text is None else id_text
        self.categorical: List[str] = [] if categorical is None else categorical
        self.continuous: List[str] = [] if continuous is None else continuous
        if feature_types is None:
            self.feature_types: Dict[str, List[str]] = self._check_data_types()
        else:
            for ft in feature_types.keys():
                if ft not in ['continuous', 'categorical', 'ordinal', 'id_text', 'date']:
                    raise DataExplorerException('Feature type ({}) not supported'.format(ft))
            self.feature_types: Dict[str, List[str]] = feature_types
        self.target: str = target
        if self.target is not None:
            if self.target not in self.features:
                raise DataExplorerException('Target variable ({}) not in data set'.format(target))
            else:
                self.target_values: List[str] = self._target_values()
        self.include_nan: bool = include_nan
        self.plot: bool = plot
        self.plot_type: str = plot_type
        self.file_path: str = output_file_path
        self.invalid_values: List[str] = ['nan', 'NaN', 'NaT', 'none', 'None', 'inf', '-inf']
        self.df = self.df.replace(self.invalid_values, np.nan)

    def _check_data_types(self) -> Dict[str, List[str]]:
        """
        Get feature types

        :return: Dict[str, List[str]]
            Dictionary containing the names of the features and the regarding data types
        """
        return EasyExploreUtils().get_feature_types(df=self.df,
                                                    features=self.features,
                                                    dtypes=self.data_types,
                                                    continuous=self.continuous,
                                                    categorical=self.categorical,
                                                    ordinal=self.ordinal,
                                                    date=self.date,
                                                    id_text=self.id_text,
                                                    date_edges=self.date_edges,
                                                    print_msg=True
                                                    )

    def _check_duplicates(self, by_row: bool = True, by_col: bool = True) -> Dict[str, list]:
        """
        Get duplicated cases and features

        :param by_row: bool
            Find duplicates among cases

        :param by_col: bool
            Find duplicates among features

        :return: Dict[str, list]:
            List of index values of the duplicated cases and list of names of the duplicated features
        """
        _duplicates: dict = dict(cases=[], features=[])
        if by_row:
            _duplicates['cases'] = list(self.df.loc[self.df.duplicated(), :].index.values.compute())
        if by_col:
            _transposed_dask_array = self.df.to_dask_array().transpose().compute()
            _duplicates['features'] = [self.df.columns[f] for f, feature in enumerate(dd.from_array(x=_transposed_dask_array).compute().duplicated()) if feature]
        return _duplicates

    def _check_invariant_features(self) -> List[int]:
        """
        Get invariant features

        :return: List[int]
            Indices of the invariant features
        """
        _invariant_features: list = []
        for ft in self.features:
            _unique_values: np.ndarray = self.df[ft].unique()
            if len(_unique_values) <= 2:
                if len(_unique_values) == 1:
                    _invariant_features.append(ft)
                else:
                    if any(self.df[ft].isnull()):
                        _invariant_features.append(ft)
        return _invariant_features

    def _check_outliers(self, meth: str = 'iqr', outlier_threshold: float = 0.15) -> List[int]:
        """
        Get univariate outliers or extreme values

        :param meth: str
            Name of the method to use

        :param outlier_threshold: float
            Threshold of outliers for cutting them off

        :return: List of integers containing the index values of outlier cases
        """
        _cases: list = []
        _o: float = outlier_threshold if (outlier_threshold > 0) and (outlier_threshold < 1) else 0.15
        if meth == 'iqr':
            for feature in self.feature_types['continuous']:
                _values: np.ndarray = self.df[feature].values.compute()
                _iqr = np.quantile(a=_values, q=0.75) - np.quantile(a=_values, q=0.25)
                _lower = _values < (np.quantile(a=_values, q=0.25) - (1.5 * _iqr))
                _lower_cases = np.where(_lower)[0].tolist()
                _upper = _values > (np.quantile(a=_values, q=0.75) + (1.5 * _iqr))
                _upper_cases = np.where(_upper)[0].tolist()
                _cases = _cases + _lower_cases + _upper_cases
        else:
            raise DataExplorerException('Method ({}) not supported'.format(meth))
        return _cases

    def _nan_case_summary(self, nan_case_df: pd.Series) -> pd.DataFrame:
        """
        Generate missing data case-wise summary

        :param nan_case_df: pd.DataFrame
            Pandas Series containing the missing data case-wise relative frequency table

        :return: pd.DataFrame
            Pandas DataFrame containing the missing data case-wise summary table
        """
        _rec = dict()
        _rec[0] = self.n_cases - nan_case_df.shape[0] if self.n_cases >= nan_case_df.shape[0] else 0
        for val in nan_case_df:
            if (val > 0) and (val <= 10):
                _rec[val] = 1
            elif (val > 10) and (val <= 25):
                _rec[val] = 2
            elif (val > 25) and (val <= 50):
                _rec[val] = 3
            elif (val > 50) and (val <= 75):
                _rec[val] = 4
            elif (val > 75) and (val < 100):
                _rec[val] = 5
            else:
                _rec[val] = 6
        _df_mis_case_bin = nan_case_df.replace(to_replace=_rec)
        _df_mis_case_bin = _df_mis_case_bin.astype(dtype=int)
        _df_freq_bin = _df_mis_case_bin.value_counts(normalize=False, sort=True, ascending=False, bins=None)
        if hasattr(_df_freq_bin, 'columns'):
            _df_freq_bin = _df_freq_bin.rename(columns={'rel': 0})
        _df_freq_rel_bin = _df_freq_bin.apply(lambda x: 100 * round(x / len(self.df), 4))
        if hasattr(_df_freq_bin, 'columns'):
            _df_freq_bin = _df_freq_rel_bin.rename(columns={'rel': 1})
        _df: pd.DataFrame = pd.DataFrame()
        _df[0] = _df_freq_bin
        _df[1] = _df_freq_rel_bin
        _df = _df.transpose()
        _df = _df.rename(index={0: 'N', 1: '%'},
                         columns={0: 'Cases containing 0 % Missings',
                                  1: 'Cases containing less than 10 % Missings',
                                  2: 'Cases containing 10 % - 25 % Missings',
                                  3: 'Cases containing 25 % - 50 % Missings',
                                  4: 'Cases containing 50 % - 75 % Missings',
                                  5: 'Cases containing more than 75 % Missings',
                                  6: 'Cases containing 100 % Missings'
                                  }
                         )
        _df = _df.transpose()
        _df['N'] = _df['N'].astype('int')
        return _df

    def _target_values(self) -> np.array:
        """
        Get unique values of the target variable

        :return: np.array
            Unique values of the target variable
        """
        return self.df[self.target].unique().compute()

    def break_down(self, plot_type: str = 'violin', **kwargs):
        """
        Generate univariate statistics of continuous features grouped by categorical features

        :param plot_type: str
            Name of the visualization type:
                -> radar: Radar Chart for level 1 overview
                -> parcoords: Parallel Coordinate Chart for level 2 overview
                -> sunburst: Sunburst Chart for level 2 overview
                -> tree: Treemap Chart for level 2 overview
                -> hist: Histogram Chart for level 2 overview
                -> violin: Violin Chart for level 3 overview

        :param kwargs: dict
            Key-word arguments

        :return Pandas DataFrame or dask dataframe
            Breakdown statistics
        """
        _features: List[str] = []
        if kwargs.get('include') is not None:
            for include in list(set(kwargs.get('include'))):
                if include in self.features:
                    _features.append(include)
        if kwargs.get('exclude') is not None:
            _features: List[str] = self.features
            for exclude in list(set(kwargs.get('exclude'))):
                if exclude in _features:
                    del _features[_features.index(exclude)]
        if len(_features) == 0:
            _features = self.features
        _break_down_stats: pd.DataFrame = pd.DataFrame()
        if plot_type not in ['radar', 'parcoords', 'sunburst', 'tree', 'hist', 'violin']:
            raise DataExplorerException('Plot type ({}) for visualizing categorical breakdown not supported'.format(plot_type))
        _cat_features: List[str] = [feature for feature in self.feature_types.get('categorical') + self.feature_types.get('ordinal') if feature in _features]
        _continuous_features: List[str] = [conti for conti in self.feature_types.get('continuous') if conti in _features]
        if len(_cat_features) == 0:
            Log(write=False, level='info').log(msg='No categorical features found to breakdown')
        else:
            if len(_continuous_features) > 0:
                _agg: dict = {conti: ['count', 'min', 'mean', 'max', 'sum'] for conti in _continuous_features}
                try:
                    _break_down_stats = self.df.groupby(by=_cat_features).aggregate(_agg).compute()
                except TypeError:
                    Log(write=False).log(msg='Categorical break down statistics could not be calculated. Please check input values of categorical features')
                if self.plot:
                    DataVisualizer(title='Breakdown Statistics',
                                   df=self.df,
                                   features=_continuous_features + _cat_features,
                                   group_by=_cat_features,
                                   plot_type=plot_type,
                                   melt=False,
                                   interactive=True,
                                   height=500,
                                   width=500,
                                   render=True if self.file_path is None else False,
                                   file_path=self.file_path
                                   ).run()
            else:
                Log(write=False, level='info').log(msg='No continuous features found to aggregate group by statistics')
        return _break_down_stats

    def cor(self,
            marginal: bool = True,
            partial: bool = True,
            marginal_meth: str = 'pearson',
            min_obs: int = 1,
            decimals: int = 2
            ) -> dict:
        """
        Calculate correlation coefficients

        :param marginal: bool
            Calculate marginal (classical) correlation

        :param partial: bool
            Calculate partial correlation

        :param marginal_meth: str
            Name of the method to be used as marginal correlation coefficient
                -> pearson: Marginal Correlation based on Pearson's r
                -> kendall: Rank Correlation based on Kendall
                -> spearman: Rank Correlation based on Spearman

        :param min_obs: int
            Minimum amount of valid observations

        :param decimals: int
            Amount of decimal digits to show

        :return: dict
            Containing the marginal / partial correlation coefficient
        """
        _cor: dict = dict(marginal={}, partial={}, diff={})
        _cor_plot: dict = {}
        _min_obs: int = min_obs if min_obs > 0 else 1
        _decimals: int = decimals if decimals > 0 else 2
        if partial:
            _ar: np.array = self.df[self.feature_types.get('continuous')].dropna().values
            if _ar.shape[0] == 0:
                Log(write=False).log('Partial correlation not possible because of high amount of missing values')
            else:
                #_n_features = _ar.shape[1]
                #_partial_cor: np.array = np.zeros((_n_features, _n_features), dtype=np.float)
                #for i in range(_n_features):
                #    _partial_cor[i, i] = 1
                #    for j in range(i + 1, _n_features):
                #        _idx = np.ones(_n_features, dtype=np.bool)
                #        _idx[i] = False
                #        _idx[j] = False
                #        _beta_i = linalg.lstsq(_ar[:, _idx], _ar[:, j])[0]
                #        _beta_j = linalg.lstsq(_ar[:, _idx], _ar[:, i])[0]
                #        _res_j = _ar[:, j] - _ar[:, _idx].dot(_beta_i)
                #        _res_i = _ar[:, i] - _ar[:, _idx].dot(_beta_j)
                #        _corr = stats.pearsonr(_res_i, _res_j)[0]
                #        _partial_cor[i, j] = _corr
                #        _partial_cor[j, i] = _corr
                #_cor_matrix: pd.DataFrame = pd.DataFrame(data=_partial_cor, columns=self.feature_types.get('continuous'), index=self.feature_types.get('continuous'))
                _partial_cor_matrix = StatsUtils(data=self.df, features=self.feature_types.get('continuous')).correlation(meth='partial')
                #print(_partial_cor_matrix)
                _cor['partial'].update(_partial_cor_matrix.to_dict())
                _cor_plot.update({'Partial Correlation': dict(data=_partial_cor_matrix,
                                                              features=self.feature_types.get('continuous'),
                                                              plot_type='heat',
                                                              kwargs=dict(layout={})
                                                              )
                                  })
        if marginal:
            if marginal_meth not in ['pearson', 'kendall', 'spearman']:
                _features: List[str] = self.feature_types.get('continuous')
                _cor_matrix: pd.DataFrame = self.df[_features].corr(method='pearson', min_periods=_min_obs).round(decimals=_decimals)
                Log(write=False, level='error').log('Correlation method ({}) not supported; Calculate marginal correlation using pearsons r instead'.format(marginal_meth))
            else:
                if marginal_meth == 'pearson':
                    _features: List[str] = self.feature_types.get('continuous')
                    _cor_matrix: pd.DataFrame = self.df[_features].corr(method=marginal_meth, min_periods=_min_obs).round(decimals=_decimals)
                else:
                    _features: List[str] = self.feature_types.get('ordinal')
                    _cor_matrix: pd.DataFrame = self.df[_features].corr(method=marginal_meth, min_periods=_min_obs).round(decimals=_decimals)
            _cor['marginal'].update(_cor_matrix.to_dict())
            _cor_plot.update({'Marginal Correlation': dict(data=_cor_matrix,
                                                           features=_features,
                                                           plot_type='heat',
                                                           kwargs=dict(layout={})
                                                           )
                              })
        if partial and marginal:
            _diff: pd.DataFrame = _cor_matrix - _partial_cor_matrix
            _cor['diff'].update(_diff.to_dict())
            _cor_plot.update({'Difference: Partial vs. Marginal Correlation': dict(data=_cor_matrix,
                                                                                   features=self.feature_types.get('continuous'),
                                                                                   plot_type='heat',
                                                                                   kwargs=dict(layout={})
                                                                                   )
                              })
        if self.plot:
            DataVisualizer(subplots=_cor_plot,
                           feature_types=self.feature_types,
                           plot_type='heat',
                           interactive=True,
                           height=500,
                           width=500,
                           render=True if self.file_path is None else False,
                           file_path=self.file_path
                           ).run()
        return _cor

    def data_distribution(self,
                          categorical: bool = True,
                          continuous: bool = True,
                          over_time: bool = False,
                          **kwargs
                          ) -> dict:
        """
        Check data distribution of different data types

        :param categorical: bool
            Calculate distribution of categorical features

        :param continuous: bool
            Calculate distribution of continuous features

        :param over_time: bool
            Calculate distribution of continuous features over time period

        :param kwargs: dict
            Key-word arguments

        :return: dict
            Distribution parameter of each features
        """
        _features: List[str] = []
        if kwargs.get('include') is not None:
            for include in list(set(kwargs.get('include'))):
                if include in self.features:
                    _features.append(include)
        if kwargs.get('exclude') is not None:
            _features: List[str] = self.features
            for exclude in list(set(kwargs.get('exclude'))):
                if exclude in _features:
                    del _features[_features.index(exclude)]
        if len(_features) == 0:
            _features = self.features
        _subplots: dict = {}
        _distribution: dict = {}
        _supported_cat_plot_types: List[str] = ['bar', 'pie']
        _supported_conti_plot_types: List[str] = ['box', 'histo', 'violin']
        if categorical:
            _categorical_features: List[str] = [feature for feature in self.feature_types.get('ordinal') + self.feature_types.get('categorical') if feature in _features]
            if len(_categorical_features) > 0:
                for ft in _categorical_features:
                    try:
                        _distribution[ft] = self.df[ft].value_counts(sort=True, ascending=False, dropna=self.include_nan).compute()
                    except TypeError:
                        Log(write=False).log(msg='Distribution of categorical feature "{}" not possible. Please check unique values'.format(ft))
                _subplots.update({'Categorical Features': dict(data=self.df,
                                                               features=_categorical_features,
                                                               plot_type='bar',
                                                               melt=False
                                                               )
                                  })
        if continuous:
            _continuous_features: List[str] = []
            for conti in self.feature_types.get('continuous'):
                if conti in _features:
                    if str(self.df[conti].dtype).find('float') < 0:
                        self.df[conti] = self.df[conti].astype(float)
                    _continuous_features.append(conti)
            if len(_continuous_features) > 0:
                _desc: dict = self.df[_continuous_features].describe(percentiles=PERCENTILES).compute()
                #_norm: dict = StatsUtils(data=self.df, features=self.feature_types.get('continuous')).normality_test(alpha=0.05, meth='shapiro-wilk')
                #_skew: dict = StatsUtils(data=self.df, features=self.feature_types.get('continuous')).skewness_test(axis='col')
                #_annotations: List[dict] = []
                #for ft in _desc.keys():
                #    _annotations.append(dict(text='Mean={}<br></br>Median={}<br></br>Std={}<br></br>Normality:{}<br></br>Skewness:{}'.format(self.df[ft].mean().compute(), np.median(self.df[ft].values.compute()), self.df[ft].std().compute(), _norm.get(ft), _skew.get(ft)),
                #                             align='left',
                #                             showarrow=False,
                #                             x=0.5,
                #                             y=0.9,
                #                             xref='paper',
                #                             yref='paper',
                #                             bordercolor='black',
                #                             borderwidth=0
                #                             )
                #                        )
                #    _distribution[ft] = _desc.get(ft)
                _subplots.update({'Continuous Features': dict(data=self.df,
                                                              features=self.feature_types.get('continuous'),
                                                              plot_type='hist',
                                                              melt=False,
                                                              #kwargs=dict(layout=dict(annotations=_annotations))
                                                              )
                                  })
        if over_time:
            if len(self.feature_types.get('date')) == 0:
                Log(write=False, level='error').log('No time feature found in data set')
            else:
                __continuous_features: List[str] = []
                for conti in self.feature_types.get('continuous'):
                    if conti in _features:
                        if str(self.df[conti].dtype).find('float') < 0:
                            self.df[conti] = self.df[conti].astype(float)
                        __continuous_features.append(conti)
                _date_features: List[str] = []
                for date in self.feature_types.get('date'):
                    if date in _features:
                        if str(self.df[date].dtype).find('object') >= 0:
                            self.df[date] = dd.from_array(x=pd.to_datetime(self.df[date].values.compute(), errors='coerce'))
                        _date_features.append(date)
                for ft in _date_features:
                    _distribution[ft] = self.df[ft].value_counts(sort=True, ascending=False, dropna=self.include_nan).compute()
                if len(_date_features) > 0 and len(__continuous_features) > 0:
                    _subplots.update({'Distribution over Time': dict(data=self.df,
                                                                     features=__continuous_features,
                                                                     time_features=_date_features,
                                                                     plot_type='ridgeline'
                                                                     )
                                      })
        if self.plot:
            if len(_subplots) > 0:
                DataVisualizer(title='Data Distribution',
                               feature_types=self.feature_types,
                               subplots=_subplots,
                               interactive=True,
                               height=500,
                               width=500,
                               render=True if self.file_path is None else False,
                               file_path=self.file_path
                               ).run()
        return _distribution

    def data_health_check(self,
                          sparsity: bool = True,
                          invariant: bool = True,
                          duplicate_cases: bool = False,
                          duplicate_features: bool = True,
                          nan_heat_map: bool = True,
                          nan_threshold: float = 0.95,
                          other_mis: list = None,
                          **kwargs
                          ) -> Dict[str, list]:
        """
        Check the quality of the data set in terms of sparsity, anomalies, duplicates, invariance

        :param sparsity: bool
            Check sparsity of the data

        :param invariant: bool
            Check whether the data is invariant

        :param duplicate_cases: bool
            Check whether cases are duplicated

        :param duplicate_features: bool
            Check whether features are duplicated

        :param nan_heat_map: bool
            Generate heat map for missing data visualization

        :param nan_threshold: float
            Threshold of missing values for cutting them off

        :param other_mis: list
            List of (other missing) values to convert to missing value NaN

        :param kwargs: dict
            Key-word arguments

        :return: Mapping[str, list]
            Results of the data health check
        """
        _features: List[str] = []
        if kwargs.get('include') is not None:
            for include in list(set(kwargs.get('include'))):
                if include in self.features:
                    _features.append(include)
        if kwargs.get('exclude') is not None:
            _features: List[str] = self.features
            for exclude in list(set(kwargs.get('exclude'))):
                if exclude in _features:
                    del _features[_features.index(exclude)]
        if len(_features) == 0:
            _features = self.features
        _index: list = []
        _cases: list = []
        __features: list = []
        _info_table: dict = {}
        _tables: dict = {}
        _subplots: dict = {}
        _nan_threshold: float = 100 * nan_threshold
        _data_health: dict = {'sparsity': {'cases': [], 'features': []},
                              'invariant': [],
                              'duplicate': {'cases': [], 'features': []}
                              }
        if not sparsity and not invariant and not duplicate_cases and not duplicate_features:
            raise DataExplorerException('No method for analyzing data health enabled')
        if other_mis is not None:
            self.df = self.df.replace(other_mis, np.nan)
        if sparsity:
            _mis_analysis = {'features': {'abs': {}, 'rel': {}}, 'cases': {'abs': {}, 'rel': {}}}
            _mis: dict = dict(cases=[], features=[])
            for ft in _features:
                _feature_mis: list = np.where(pd.isnull(self.df[ft].values.compute()))[0].tolist()
                _mis['cases'].extend(_feature_mis)
                if len(_feature_mis) > 0:
                    _mis['features'].append(ft)
            if len(_mis['cases']) > 0:
                for case in list(set(_mis['cases'])):
                    _df_case_wise = copy.deepcopy(self.df.loc[case, :].values.compute())
                    _nan_case_wise = len(np.where(pd.isnull(_df_case_wise))[0])
                    _mis_analysis['cases']['abs'].update({case: _nan_case_wise})
                    _mis_analysis['cases']['rel'].update({case: 100 * round(_nan_case_wise / len(_features), 6)})
                del _df_case_wise
                for feature in list(set(_mis['features'])):
                    _df_feature_wise = self.df.loc[:, feature].values.compute()
                    _nan_feature_wise = len(np.where(pd.isnull(_df_feature_wise))[0])
                    _mis_analysis['features']['abs'].update({feature: _nan_feature_wise})
                    _mis_analysis['features']['rel'].update({feature: 100 * round(_nan_feature_wise / len(_df_feature_wise), 6)})
                del _df_feature_wise
            _nan_cases = len(_mis_analysis['cases']['abs'].keys())
            _nan_features = len(_mis_analysis['features']['abs'].keys())
            _data_health['sparsity']['cases'] = list(EasyExploreUtils().subset_dict(d=_mis_analysis['cases']['rel'],
                                                                                    threshold=_nan_threshold
                                                                                    ).keys()
                                                     )
            _data_health['sparsity']['features'] = [self.features[ft] for ft in EasyExploreUtils().subset_dict(d=_mis_analysis['features']['rel'], threshold=_nan_threshold).keys()]
            _info_table['sparsity_cases'] = '{} ({} %)'.format(_nan_cases,
                                                               str(100 * round(_nan_cases / self.n_cases, 3))
                                                               )
            _info_table['sparsity_features'] = '{} ({} %)'.format(_nan_features,
                                                                  str(100 * round(_nan_features / len(_features), 3))
                                                                  )
            _index.append('Amount of missing data case-wise')
            _index.append('Amount of missing data feature-wise')
            if self.plot:
                _df_feature_mis = pd.DataFrame(data=_mis_analysis['features'])
                _df_feature_mis = _df_feature_mis.rename(columns={'abs': 'N', 'rel': '%'},
                                                         index={val: k for k, val in _mis_analysis['features']['rel'].items()}
                                                         )
                if any(self.df.isnull().compute()):
                    _df_case_mis = pd.DataFrame(data=_mis_analysis['cases'])
                    _df_all_data = self.df.applymap(lambda x: 1 if x == x else 0).compute()
                    _df_nan_case_sum: pd.DataFrame = self._nan_case_summary(nan_case_df=_df_case_mis['rel'])
                    _subplots.update({'Sparsity of data set': dict(data=_df_nan_case_sum,
                                                                   features=[],
                                                                   plot_type='pie',
                                                                   kwargs=dict(values=[_df_all_data.sum().sum(),
                                                                                       (self.n_cases * len(_features)) - _df_all_data.sum().sum()
                                                                                       ],
                                                                               labels=['Valid Data', 'Missing Data']
                                                                               )
                                                                   ),
                                      'Missing Data Distribution Case-wise': dict(data=_df_nan_case_sum,
                                                                                  features=[],
                                                                                  plot_type='pie',
                                                                                  kwargs=dict(values=list(_df_nan_case_sum.loc[:, 'N'].values),
                                                                                              labels=list(_df_nan_case_sum.loc[:, 'N'].index.values)
                                                                                              )
                                                                                  ),
                                      'Missing Data Distribution Features-wise': dict(data=_df_feature_mis,
                                                                                      features=[],
                                                                                      plot_type='pie',
                                                                                      kwargs=dict(values=list(_df_feature_mis.loc[:, 'N'].values),
                                                                                                  labels=list(_df_feature_mis.loc[:, 'N'].index.values)
                                                                                                  )
                                                                                      ),
                                      'Sparsity of the features': dict(data=_df_feature_mis,
                                                                       features=[],
                                                                       plot_type='table',
                                                                       kwargs=dict(index_title='Features with Missing Data')
                                                                       )
                                      })
                    if nan_heat_map:
                        _subplots.update({'Missing Data Heatmap': dict(data=pd.DataFrame(data=self.df[_features].values.compute(),
                                                                                         columns=_features
                                                                                         ),
                                                                       features=None,
                                                                       plot_type='heat',
                                                                       kwargs=dict(z=self.df.isnull().astype(int).values.compute(),
                                                                                   colorbar=dict(title='Value Range',
                                                                                                 tickvals=['0', '1'],
                                                                                                 ticktext=['Valid',
                                                                                                           'Missing'])
                                                                                   )
                                                                       )
                                          })
        if invariant:
            _data_health['invariant'] = self._check_invariant_features()
            _i = len(_data_health['invariant'])
            _info_table['invariant'] = '{} ({} %)'.format(_i, str(100 * round(_i / len(_features), 4)))
            _index.append('Amount of invariant features')
            if self.plot:
                _label: List[str] = []
                for ft in self.features:
                    if ft in _data_health.get('invariant'):
                        _label.append('invariant')
                    else:
                        _label.append('variant')
                _subplots.update({'Invariant Features': dict(data=pd.DataFrame(data=dict(features=self.features, invariant=_label)),
                                                             features=['invariant'],
                                                             plot_type='pie'
                                                             )
                                  })
        if duplicate_features or duplicate_cases:
            _data_health['duplicate'] = EasyExploreUtils().get_duplicates(df=self.df, cases=duplicate_cases, features=duplicate_features)
            if duplicate_cases:
                _dc = len(_data_health['duplicate']['cases'])
                _info_table['duplicate_cases'] = '{} ({} %)'.format(_dc, str(100 * round(_dc / self.n_cases, 4)))
                _index.append('Amount of duplicated cases')
            _d = len(_data_health['duplicate']['features'])
            _info_table['duplicate_features'] = '{} ({} %)'.format(_d, str(100 * round(_d / len(_features), 4)))
            _index.append('Amount of duplicated features')
            if self.plot:
                _duplicate_cases: List[str] = []
                _duplicate_features: List[str] = []
                for i in range(0, self.n_cases, 1):
                    if i in _data_health['duplicate']['cases']:
                        _duplicate_cases.append('duplicate')
                    else:
                        _duplicate_cases.append('unique')
                for ft in _features:
                    if ft in _data_health['duplicate']['features']:
                        _duplicate_features.append('duplicate')
                    else:
                        _duplicate_features.append('unique')
                if duplicate_cases:
                    _subplots.update({'Duplicate Cases': dict(data=pd.DataFrame(data=dict(duplicate_cases=_duplicate_cases)),
                                                              features=['duplicate_cases'],
                                                              plot_type='pie',
                                                              interactive=True
                                                              )
                                      })
                if duplicate_features:
                    _subplots.update({'Duplicate Features': dict(data=pd.DataFrame(data=dict(features=_features, duplicate_features=_duplicate_features)),
                                                                 features=['duplicate_features'],
                                                                 plot_type='pie',
                                                                 interactive=True
                                                                 )
                                      })
        for mis_feature in _data_health['sparsity']['features']:
            __features.append(mis_feature)
        for inv_feature in _data_health['invariant']:
            __features.append(inv_feature)
        for dup_feature in _data_health['duplicate']['features']:
            __features.append(dup_feature)
        if duplicate_cases:
            for dup_case in _data_health['duplicate']['cases']:
                _cases.append(dup_case)
        for mis_case in _data_health['sparsity']['cases']:
            _cases.append(mis_case)
        if self.plot:
            _results_after_cleaning: pd.DataFrame = pd.DataFrame(data=np.nan, columns=_features, index=self.data_index)
            _results_after_cleaning = _results_after_cleaning.fillna(0)
            if _data_health.get('sparsity') is not None:
                for mis in _data_health['sparsity'].get('features'):
                    _results_after_cleaning.loc[:, mis] = 1
            if _data_health.get('invariant') is not None:
                for inv in _data_health.get('invariant'):
                    _results_after_cleaning.loc[:, inv] = 2
            if _data_health.get('duplicate') is not None:
                for dup_ft in _data_health['duplicate'].get('features'):
                    _results_after_cleaning.loc[:, dup_ft] = 3
                for dup_cases in _data_health['duplicate'].get('cases'):
                    _results_after_cleaning.loc[dup_cases, :] = 3
            for ft in _results_after_cleaning.columns:
                _results_after_cleaning[ft] = _results_after_cleaning[ft].astype(dtype='int32')
            _summary: pd.DataFrame = pd.DataFrame(data=_info_table.values(), columns=['N (%)'], index=_index)
            _subplots.update({'Data Structure Describing Data Health': dict(data=_results_after_cleaning,
                                                                            features=[],
                                                                            plot_type='heat',
                                                                            kwargs=dict(z=_results_after_cleaning.values,
                                                                                        colorbar=dict(title='Value Range',
                                                                                                      tickvals=['0', '1', '2', '3'],
                                                                                                      ticktext=['Valid', 'Missing', 'Invariant', 'Duplicate']
                                                                                                      )
                                                                                        )
                                                                            ),
                              'Data Health Check Summary': dict(data=_summary,
                                                                features=[],
                                                                plot_type='table',
                                                                kwargs=dict(index_title='Attributes of Data Health')
                                                                )
                              })
            DataVisualizer(title='Data Health Check',
                           feature_types=self.feature_types,
                           subplots=_subplots,
                           interactive=True,
                           width=500,
                           height=500,
                           render=True if self.file_path is None else False,
                           file_path=self.file_path
                           ).run()
        del _results_after_cleaning
        return dict(cases=list(set(_cases)), features=list(set(__features)))

    def data_typing(self) -> dict:
        """
        Check typing of each feature

        :return: dict
            Typing and the data typing of and recommendation for each feature
        """
        _feature: list = []
        _typing: dict = {}
        _table: Dict[str, list] = {'feature_type': [], 'data_type': [], 'rec': []}
        for i in range(0, len(self.features), 1):
            if any(self.df[self.features[i]].isnull().compute()):
                if len(self.df[self.features[i]].unique().compute()) == 1:
                    _table['feature_type'].append('float')
                    _table['data_type'].append('unknown')
                    _table['rec'].append('Drop feature (no valid data)')
                    _feature.append(self.features[i])
                    continue
            if str(self.data_types[i]).find('bool') >= 0:
                _table['feature_type'].append('boolean')
                _table['data_type'].append('categorical')
                _table['rec'].append('Convert to integer')
                _feature.append(self.features[i])
                _typing[self.features[i]] = 'int'
            elif str(self.data_types[i]).find('float') >= 0:
                if self.features[i] in self.feature_types.get('date'):
                    _table['feature_type'].append('float')
                    _table['data_type'].append('date')
                    _table['rec'].append('Convert to datetime')
                    _feature.append(self.features[i])
                    _typing[self.features[i]] = 'date'
                elif self.features[i] in self.feature_types.get('ordinal'):
                    _table['feature_type'].append('float')
                    _table['data_type'].append('categorical')
                    _table['rec'].append('Convert to integer')
                    _feature.append(self.features[i])
                    _typing[self.features[i]] = 'int'
                elif self.features[i] in self.feature_types.get('continuous'):
                    if any(self.df[self.features[i]].isnull().compute()):
                        _table['feature_type'].append('float')
                        _table['data_type'].append('continuous')
                        _table['rec'].append('Handle missing data')
                        _feature.append(self.features[i])
                else:
                    _table['feature_type'].append('float')
                    _feature.append(self.features[i])
                    if self.features[i] in self.feature_types.get('categorical'):
                        _table['data_type'].append('categorical')
                        if any(self.df[self.features[i]].isnull().compute()):
                            _table['rec'].append('Handle missing data and convert to integer')
                        else:
                            _table['rec'].append('Convert to integer')
                        _typing[self.features[i]] = 'int'
                    else:
                        _table['data_type'].append('id_text')
                        _table['rec'].append('Convert to string')
            elif str(self.data_types[i]).find('int') >= 0:
                if self.features[i] not in self.feature_types.get('categorical'):
                    _table['feature_type'].append('integer')
                    _feature.append(self.features[i])
                    if self.features[i] in self.feature_types.get('ordinal'):
                        continue
                    elif self.features[i] in self.feature_types.get('date'):
                        _table['data_type'].append('date')
                        _table['rec'].append('Convert to datetime')
                        _typing[self.features[i]] = 'datetime'
                    elif self.features[i] in self.feature_types.get('text'):
                        _table['data_type'].append('text')
                        _table['rec'].append('Convert to string')
                        _typing[self.features[i]] = 'str'
            elif str(self.data_types[i]).find('object') >= 0:
                if self.features[i] in self.feature_types.get('date'):
                    _table['feature_type'].append('text')
                    _feature.append(self.features[i])
                    _table['data_type'].append('date')
                    _table['rec'].append('Convert to datetime')
                    _typing[self.features[i]] = 'datetime'
                elif self.features[i] in self.feature_types.get('continuous'):
                    _table['feature_type'].append('text')
                    _feature.append(self.features[i])
                    _table['data_type'].append('continuous')
                    _typing[self.features[i]] = 'float'
                    if any(self.df[self.features[i]].isnull().compute()):
                        _table['rec'].append('Handle missing data and convert to float')
                    else:
                        _table['rec'].append('Convert to float')
                elif self.features[i] in self.feature_types.get('categorical') or self.features[i] in self.feature_types.get('ordinal'):
                    _table['feature_type'].append('text')
                    _feature.append(self.features[i])
                    _table['data_type'].append('categorical')
                    _typing[self.features[i]] = 'int'
                    if any(self.df[self.features[i]].isnull().compute()):
                        _table['rec'].append('Handle missing data and convert to integer by label encoding')
                    else:
                        _table['rec'].append('Convert to integer by label encoding')
        if self.plot:
            _df: pd.DataFrame = pd.DataFrame(data=_table)
            if _df.shape[0] == 0:
                Log(write=False).log('All features and data points are correctly typed')
            else:
                _df = _df.rename(columns={'feature_type': 'Feature Type', 'data_type': 'Data Type', 'rec': 'Recommendation'},
                                 index={idx: name for idx, name in zip(range(len(_feature)), _feature)}
                                 )
                _kwargs: dict = dict(index_title='Features')
                DataVisualizer(title='Data Type Check',
                               df=_df,
                               feature_types=self.feature_types,
                               plot_type='table',
                               interactive=True,
                               height=500,
                               width=500,
                               render=True if self.file_path is None else False,
                               file_path=self.file_path,
                               **_kwargs
                               ).run()
        return _typing

    def get_feature_types(self) -> Dict[str, List[str]]:
        """
        Get and return data types of each feature

        :return: dict
            Names of the features and the regarding data types
        """
        return self.feature_types

    def geo_stats(self,
                  geo_features: List[str] = None,
                  lat: str = None,
                  lon: str = None,
                  val: str = None,
                  plot_type: str = 'density'
                  ) -> pd.DataFrame:
        """
        Calculate statistics based on geographical area

        :param geo_features: List[str]
            Geographical features for generate group by statistics

        :param lat: str
            Name of the latitude feature for visualization

        :param lon: str
            Name of the longitude feature for visualization

        :param val: str
            Name of the continuous value feature for visualization

        :param plot_type: str
            Name of the plot type to use:
                -> geo: Geomap
                -> density: Density map

        :return: pd.DataFrame
            Statistics based on geographical features
        """
        if geo_features is None:
            _geo_stats: pd.DataFrame = pd.DataFrame()
        else:
            if len(geo_features) > 0:
                _geo_features: List[str] = []
                for geo in geo_features:
                    if geo in self.df.columns:
                        _geo_features.append(geo)
                if len(_geo_features) > 0:
                    _agg: dict = {conti: ['count', 'min', 'quantile', 'median', 'mean', 'max', 'sum'] for conti in self.feature_types.get('continuous')}
                    _categorical_features: List[str] = self.feature_types.get('categorical') + self.feature_types.get('ordinal')
                    _agg.update({cat: ['count'] for cat in _categorical_features if cat not in _geo_features})
                    _geo_stats: pd.DataFrame = self.df[_geo_features].groupby().aggregate(_agg).compute()
                else:
                    _geo_stats: pd.DataFrame = pd.DataFrame()
            else:
                _geo_stats: pd.DataFrame = pd.DataFrame()
        if self.plot:
            if lat is None:
                raise DataExplorerException('No latitude feature found')
            else:
                if lat not in self.df.columns:
                    raise DataExplorerException('Latitude feature not found in data set')
            if lon is None:
                raise DataExplorerException('No longitude feature found')
            else:
                if lon not in self.df.columns:
                    raise DataExplorerException('Longitude feature not found in data set')
            if val is None:
                raise DataExplorerException('No continuous value feature found')
            else:
                if val not in self.df.columns:
                    raise DataExplorerException('Continuous value feature not found in data set')
            if plot_type in ['geo', 'density']:
                _plot_type: str = plot_type
            else:
                _plot_type: str = 'density'
            if str(self.df[val].dtype).find('object') >= 0:
                self.df[val] = self.df[val].astype(float)
            if str(self.df[lat].dtype).find('object') >= 0:
                self.df[lat] = self.df[lat].astype(float)
            if str(self.df[lon].dtype).find('object') >= 0:
                self.df[lon] = self.df[lon].astype(float)
            _df: dd.DataFrame = self.df[[val, lat, lon]]
            _df = _df.loc[~_df[val].isnull(), :]
            _df = _df.loc[~_df[lat].isnull(), :]
            _df = _df.loc[~_df[lon].isnull(), :]
            _df[val] = _df[val].astype(dtype='float64')
            _df[lat] = _df[lat].astype(dtype='float64')
            _df[lon] = _df[lon].astype(dtype='float64')
            DataVisualizer(title='Geo Map "{}" (n={})'.format(val, len(_df)),
                           df=_df.compute(),
                           features=[val],
                           feature_types=self.feature_types,
                           plot_type=_plot_type,
                           interactive=True,
                           height=750,
                           width=750,
                           render=True if self.file_path is None else False,
                           file_path=self.file_path,
                           **dict(lat=lat, lon=lon)
                           ).run()
            del _df
        return _geo_stats

    def outlier_detector(self,
                         kind: str = 'uni',
                         multi_meth: List[str] = None,
                         contour: bool = False,
                         **kwargs
                         ) -> Dict[str, List[int]]:
        """
        Detect univariate or multivariate outliers

        :param kind: str
            String containing the type of the outlier detection
                -> uni: Univariate
                -> bi: Bivariate
                -> multi: Multivariate

        :param multi_meth: List[str]
            Algorithms for running multivariate outlier detection
                -> if: Isolation Forest
                -> knn: K-Nearest Neighbor

        :param contour: bool
            Generate contour chart

        :param kwargs: dict
            Key-word arguments

        :return: dict
            Detected outliers
        """
        _features: List[str] = []
        if kwargs.get('include') is not None:
            for include in list(set(kwargs.get('include'))):
                if include in self.features:
                    _features.append(include)
        if kwargs.get('exclude') is not None:
            _features: List[str] = self.features
            for exclude in list(set(kwargs.get('exclude'))):
                if exclude in _features:
                    del _features[_features.index(exclude)]
        if len(_features) == 0:
            _features = self.features
        _subplots: dict = {}
        _outlier: Dict[str, List[int]] = {}
        _continuous_features: List[str] = []
        for conti in self.feature_types.get('continuous'):
            if conti in _features:
                if str(self.df[conti].dtype).find('float') < 0:
                    self.df[conti] = self.df[conti].astype(float)
                _continuous_features.append(conti)
        if len(_continuous_features) > 0:
            if kind == 'uni':
                _outlier.update({'uni': self._check_outliers()})
                _subplots.update({'Univariate Outlier Detection': dict(data=self.df,
                                                                       features=_continuous_features,
                                                                       plot_type='violin',
                                                                       melt=True,
                                                                       kwargs=dict(layout=dict(points='outliers'))
                                                                       )
                                  })
            elif kind in ['bi', 'multi']:
                if len(_continuous_features) < 2:
                    raise DataExplorerException('Not enough features for running a multivariate outlier detection')
                _multi_meth: List[str] = ['knn'] if multi_meth is None else multi_meth
                _df: dd.DataFrame = self.df[_continuous_features]
                _df = _df.repartition(npartitions=1)
                for ft in _df.columns:
                    _df = _df.loc[~_df[ft].isnull(), :]
                _anomaly_detection: dict = AnomalyDetector(df=_df.compute(),
                                                           feature_types=self.feature_types
                                                           ).multivariate(contour_plot=contour)
                for meth in _multi_meth:
                    _outlier.update({'multi': _anomaly_detection.get('cases')})
                    _df['outlier'] = from_array(x=_anomaly_detection[meth].get('pred'))
                    if contour:
                        _multi: dict = ({'contour': dict(x=_anomaly_detection[meth].get('space'),
                                                         y=_anomaly_detection[meth].get('space'),
                                                         z=_anomaly_detection[meth].get('anomaly_score'),
                                                         colorscale='Blues',
                                                         hoverinfo='none'
                                                         )
                                         })
                    else:
                        _multi: dict = {}
                    if len(_continuous_features) == 2:
                        _multi.update(
                            {'scatter_inlier': dict(x=_df.loc[_df['outlier'] == 0, _continuous_features[0]].values.compute(), #features[0]
                                                    y=_df.loc[_df['outlier'] == 0, _continuous_features[1]].values.compute(), #_features[1]
                                                    mode='markers',
                                                    name='inlier',
                                                    marker=dict(color='rgba(255, 255, 255, 1)') if contour else dict(color='rgba(107, 142, 35, 1)'),
                                                    hoverinfo='text',
                                                    showlegend=False if contour else True
                                                    ),
                             'scatter_outlier': dict(x=_df.loc[_df['outlier'] == 1, _continuous_features[0]].values.compute(),
                                                     y=_df.loc[_df['outlier'] == 1, _continuous_features[1]].values.compute(),
                                                     mode='markers',
                                                     name='outlier',
                                                     marker=dict(color='rgba(0, 0, 0, 1)') if contour else dict(color='rgba(178, 34, 34, 1)'),
                                                     hoverinfo='text',
                                                     showlegend=False if contour else True
                                                     )
                             })
                        _subplots.update({'Multivariate Outlier Detection': dict(data=_df,
                                                                                 features=_continuous_features,
                                                                                 plot_type='multi',
                                                                                 kwargs=dict(multi=_multi,
                                                                                             layout=dict(
                                                                                                 xaxis=dict(title=_continuous_features[0]),
                                                                                                 yaxis=dict(title=_continuous_features[1])
                                                                                                 )
                                                                                             )
                                                                                 )
                                          })
                    else:
                        _pairs: List[tuple] = EasyExploreUtils().get_pairs(features=_continuous_features, max_features_each_pair=2)
                        for i, pair in enumerate(_pairs):
                            _multi.update({'scatter_inlier_{}'.format(i): dict(x=_df.loc[_df['outlier'] == 0, pair[0]].values.compute(),
                                                                               y=_df.loc[_df['outlier'] == 0, pair[1]].values.compute(),
                                                                               mode='markers',
                                                                               name='inlier ({} | {})'.format(pair[0], pair[1]),
                                                                               marker=dict(color='rgba(255, 255, 255, 1)') if contour else dict(color='rgba(107, 142, 35, 1)'),
                                                                               showlegend=True
                                                                               ),
                                           'scatter_outlier_{}'.format(i): dict(x=_df.loc[_df['outlier'] == 1, pair[0]].values.compute(),
                                                                                y=_df.loc[_df['outlier'] == 1, pair[1]].values.compute(),
                                                                                mode='markers',
                                                                                name='outlier ({} | {})'.format(pair[0], pair[1]),
                                                                                marker=dict(color='rgba(0, 0, 0, 1)') if contour else dict(color='rgba(178, 34, 34, 1)'),
                                                                                showlegend=True
                                                                                )
                                           })
                            if kind == 'bi':
                                DataVisualizer(title='Bivariate Outlier Detection',
                                               df=self.df,
                                               feature_types=self.feature_types,
                                               plot_type='multi',
                                               interactive=True,
                                               height=500,
                                               width=500,
                                               render=True if self.file_path is None else False,
                                               file_path=None if self.file_path is None else '{}/{}_{}'.format(self.file_path, pair[0], pair[1]),
                                               **dict(multi={'scatter_inlier_{}'.format(i): _multi.get('scatter_inlier_{}'.format(i)),
                                                             'scatter_outlier_{}'.format(i): _multi.get('scatter_outlier_{}'.format(i))
                                                             },
                                                      layout=dict(xaxis=dict(title=pair[0]),
                                                                  yaxis=dict(title=pair[1])
                                                                  )
                                                      )
                                               ).run()
                        if kind == 'multi':
                            _subplots.update({'Multivariate Outlier Detection': dict(data=self.df,
                                                                                     features=_continuous_features,
                                                                                     plot_type='multi',
                                                                                     render=True if self.file_path is None else False,
                                                                                     file_path=self.file_path,
                                                                                     kwargs=dict(multi=_multi)
                                                                                     )
                                              })
            else:
                raise DataExplorerException('Type of outlier detection ({}) not supported'.format(type))
            if self.plot:
                if kind != 'bi':
                    DataVisualizer(subplots=_subplots,
                                   feature_types=self.feature_types,
                                   interactive=True,
                                   height=500,
                                   width=500,
                                   render=True if self.file_path is None else False,
                                   file_path=self.file_path
                                   ).run()
        else:
            Log(write=False, level='info').log(msg='No continuous features found to run outlier detection')
        return _outlier
