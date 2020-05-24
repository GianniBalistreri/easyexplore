import numpy as np
import os
import pandas as pd

from .anomaly_detector import AnomalyDetector
from .data_visualizer import DataVisualizer
from .utils import Log, PERCENTILES, StatsUtils, EasyExploreUtils
from typing import Dict, List, Tuple

# TODO:
#  Correlation -> Partial + Final Heat Map
#  Data Distribution: Color + Annotations
#  Move MissingDataAnalysis class to utils.py
#  Parallelization


class DataExplorerException(Exception):
    """
    Class for setting up exceptions for class DataExploration, MissingDataAnalysis
    """
    pass


class DataExplorer:
    """
    Class for data exploration
    """
    def __init__(self,
                 df: pd.DataFrame,
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
                 file_path: str = None
                 ):
        """
        :param df: pd.DataFrame
            Data set

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
        """
        if df.shape[0] < 25:
            if df.shape[0] == 0:
                raise DataExplorerException('Data set is empty')
            else:
                Log(write=False, level='warn').log('Too few cases ({}) in data set'.format(df.shape[0]))
        if df.shape[1] < 2:
            Log(write=False, level='warn').log('Too few features ({}) in data set'.format(df.shape[1]))
        self.seed: int = seed if seed > 0 else 1234
        self.cpu_cores: int = os.cpu_count() - 1 if os.cpu_count() > 1 else 1
        self.max_cats: int = 500
        self.df: pd.DataFrame = df
        if include is None:
            if exclude is not None:
                _exclude: List[str] = []
                for ex in exclude:
                    if ex in self.df.keys():
                        _exclude.append(ex)
                if len(_exclude) > 0:
                    self.df = self.df[_exclude]
        else:
            _include: List[str] = []
            for inc in include:
                if inc in self.df.keys():
                    _include.append(inc)
            if len(_include) > 0:
                self.df = self.df[_include]
        self.features: List[str] = list(self.df.keys())
        self.n_cases: int = self.df.shape[0]
        self.n_features: int = self.df.shape[1]
        self.data_types: list = self.df.dtypes.tolist()
        self.data_index: list = self.df.index.values.tolist()
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
                if ft not in ['continuous', 'categorical', 'ordinal', 'text', 'date']:
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
        self.file_path: str = file_path
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
                                                    max_cats=self.max_cats,
                                                    date_edges=self.date_edges
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
            _duplicates['cases'] = self.df.loc[self.df.duplicated(), :].index.values.tolist()
        if by_col:
            _duplicates['features'] = self.df.loc[:, self.df.transpose().duplicated()].keys().tolist()
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
                _iqr = np.quantile(a=self.df[feature].values, q=0.75) - np.quantile(a=self.df[feature].values, q=0.25)
                _lower = self.df[feature].values < (np.quantile(a=self.df[feature].values, q=0.25) - (1.5 * _iqr))
                _lower_cases = np.where(_lower)[0].tolist()
                _upper = self.df[feature].values > (np.quantile(a=self.df[feature].values, q=0.75) + (1.5 * _iqr))
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
        _df_freq_bin = _df_mis_case_bin.value_counts(normalize=False, sort=True, ascending=False, bins=None)
        _df_freq_bin = _df_freq_bin.rename(columns={'rel': 0})
        _df_freq_rel_bin = _df_freq_bin.apply(func=lambda x: 100 * round(x / self.df.shape[0], 4))
        _df_freq_rel_bin = _df_freq_rel_bin.rename(columns={'rel': 1})
        _df: pd.DataFrame = pd.DataFrame(data=[_df_freq_bin, _df_freq_rel_bin])
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
        return self.df[self.target].unique()

    def break_down(self, include_cat: bool = True, plot_type: str = 'violin') -> dict:
        """
        Generate univariate statistics of continuous features grouped by categorical features

        :param include_cat: bool
            Whether to include categorical features into the statistical analysis or just as group by features

        :param plot_type: str
            Name of the visualization type:
                -> radar: Radar Chart for level 1 overview
                -> parcoords: Parallel Coordinate Chart for level 2 overview
                -> sunburst: Sunburst Chart for level 2 overview
                -> tree: Treemap Chart for level 2 overview
                -> hist: Histogram Chart for level 2 overview
                -> violin: Violin Chart for level 3 overview

        :return dict
            Breakdown statistics
        """
        _break_down_stats: dict = {}
        if plot_type not in ['radar', 'parcoords', 'sunburst', 'tree', 'hist', 'violin']:
            raise DataExplorerException('Plot type ({}) for visualizing categorical breakdown not supported'.format(plot_type))
        if len(self.feature_types.get('categorical')) == 0:
            raise DataExplorerException('No categorical features found to breakdown')
        _features: List[str] = self.feature_types.get('categorical') + self.feature_types.get('ordinal')
        for cat in _features:
            _cats: List[str] = _features
            del _cats[_cats.index(cat)]
            if cat in self.df.keys():
                for val in self.df[cat].unique():
                    _break_down_stats.update({'continuous': {cat: {val: self.df.loc[self.df[cat] == val, self.feature_types.get('continuous')].describe(PERCENTILES).to_dict()}}})
                    if include_cat:
                        _break_down_stats.update({'categorical': {cat: {val: self.df.loc[self.df[cat] == val, _features].describe(PERCENTILES).to_dict()}}})
        if self.plot:
            DataVisualizer(title='Breakdown Statistics',
                           df=self.df,
                           features=self.feature_types.get('continuous') + _features,
                           group_by=self.feature_types.get('categorical'),
                           plot_type=plot_type,
                           melt=False,
                           interactive=True,
                           height=500,
                           width=500,
                           render=True if self.file_path is None else False,
                           file_path=self.file_path
                           ).run()
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
                print(_partial_cor_matrix)
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
                          grouping: bool = True
                          ) -> dict:
        """
        Check data distribution of different data types

        :param categorical: bool
            Calculate distribution of categorical features

        :param continuous: bool
            Calculate distribution of continuous features

        :param over_time: bool
            Calculate distribution of continuous features over time period

        :return: dict
            Distribution parameter of each features
        """
        _subplots: dict = {}
        _distribution: dict = {}
        _supported_cat_plot_types: List[str] = ['bar', 'pie']
        _supported_conti_plot_types: List[str] = ['box', 'histo', 'violin']
        if categorical:
            _categorical_features: List[str] = self.feature_types.get('ordinal') + self.feature_types.get('categorical')
            for ft in _categorical_features:
                _distribution[ft] = self.df[ft].value_counts(normalize=False, sort=True, ascending=False, bins=None, dropna=self.include_nan).to_dict()
            _subplots.update({'Categorical Features': dict(data=self.df,
                                                           features=_categorical_features,
                                                           plot_type='bar',
                                                           melt=False
                                                           )
                              })
        if continuous:
            _desc: dict = self.df[self.feature_types.get('continuous')].describe(percentiles=PERCENTILES).to_dict()
            _norm: dict = StatsUtils(data=self.df, features=self.feature_types.get('continuous')).normality_test(alpha=0.05, meth='shapiro-wilk')
            _skew: dict = StatsUtils(data=self.df, features=self.feature_types.get('continuous')).skewness_test(axis='col')
            _annotations: List[dict] = []
            for ft in _desc.keys():
                _annotations.append(dict(text='Mean={}<br></br>Median={}<br></br>Std={}<br></br>Normality:{}<br></br>Skewness:{}'.format(self.df[ft].mean(), self.df[ft].median(), self.df[ft].std(), _norm.get(ft), _skew.get(ft)),
                                         align='left',
                                         showarrow=False,
                                         x=0.5,
                                         y=0.9,
                                         xref='paper',
                                         yref='paper',
                                         bordercolor='black',
                                         borderwidth=0
                                         )
                                    )
                _distribution[ft] = _desc.get(ft)
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
                for ft in self.feature_types.get('date'):
                    _distribution[ft] = self.df[ft].value_counts(normalize=False, sort=True, ascending=False, bins=None, dropna=self.include_nan).to_dict()
                _subplots.update({'Distribution over Time': dict(data=self.df,
                                                                 features=self.feature_types.get('continuous'),
                                                                 time_features=self.feature_types.get('date'),
                                                                 plot_type='ridgeline'
                                                                 )
                                  })
        if self.plot:
            DataVisualizer(title='Data Distribution',
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
                          outliers_univariate: bool = False,
                          nan_heat_map: bool = True,
                          nan_threshold: float = 0.95,
                          other_mis: list = None,
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

        :param outliers_univariate: bool
            Check whether the data has duplicated

        :param nan_heat_map: bool
            Generate heat map for missing data visualization

        :param nan_threshold: float
            Threshold of missing values for cutting them off

        :param other_mis: list
            List of (other missing) values to convert to missing value NaN

        :return: Mapping[str, list]
            Results of the data health check
        """
        _index: list = []
        _cases: list = []
        _features: list = []
        _info_table: dict = {}
        _tables: dict = {}
        _subplots: dict = {}
        _nan_threshold: float = 100 * nan_threshold
        _data_health: dict = {'sparsity': {'cases': [], 'features': []},
                              'invariant': [],
                              'duplicate': {'cases': [], 'features': []}
                              }
        if not sparsity and not invariant and not duplicate_cases and not duplicate_features and not outliers_univariate:
            raise DataExplorerException('No method for analyzing data health enabled')
        if other_mis is not None:
            self.df = self.df.replace(other_mis, np.nan)
        if sparsity:
            _mis_analysis = MissingDataAnalysis(data=self.df.to_numpy(), features=self.features).freq_nan()
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
                                                                  str(100 * round(_nan_features / self.n_features, 3))
                                                                  )
            _index.append('Amount of missing data case-wise')
            _index.append('Amount of missing data feature-wise')
            if self.plot:
                _df_feature_mis = pd.DataFrame(data=_mis_analysis['features'])
                _df_feature_mis = _df_feature_mis.rename(columns={'abs': 'N', 'rel': '%'},
                                                         index=EasyExploreUtils().replace_dict_keys(d=_mis_analysis['features']['rel'],
                                                                                                    new_keys=self.features
                                                                                                    )
                                                         )
                if any(self.df.isnull()):
                    _df_case_mis = pd.DataFrame(data=_mis_analysis['cases'])
                    _df_all_data: pd.DataFrame = self.df.applymap(lambda x: 1 if x == x else 0)
                    _df_nan_case_sum: pd.DataFrame = self._nan_case_summary(nan_case_df=_df_case_mis['rel'])
                    _subplots.update({'Sparsity of data set': dict(data=_df_nan_case_sum,
                                                                   features=[],
                                                                   plot_type='pie',
                                                                   kwargs=dict(values=[_df_all_data.sum().sum(), (self.n_cases * self.n_features) - _df_all_data.sum().sum()],
                                                                               labels=['Valid Data', 'Missing Data']
                                                                               )
                                                                   ),
                                      'Missing Data Distribution Case-wise': dict(data=_df_nan_case_sum,
                                                                                  features=[],
                                                                                  plot_type='pie',
                                                                                  kwargs=dict(values=_df_nan_case_sum.loc[:, 'N'].values.tolist(),
                                                                                              labels=_df_nan_case_sum.loc[:, 'N'].index.values.tolist()
                                                                                              )
                                                                                  ),
                                      'Missing Data Distribution Features-wise': dict(data=_df_feature_mis,
                                                                                      features=[],
                                                                                      plot_type='pie',
                                                                                      kwargs=dict(values=_df_feature_mis.loc[:, 'N'].values.tolist(),
                                                                                                  labels=_df_feature_mis.loc[:, 'N'].index.values.tolist()
                                                                                                  )
                                                                                      ),
                                      'Sparsity of the features': dict(data=_df_feature_mis,
                                                                       features=[],
                                                                       plot_type='table',
                                                                       kwargs=dict(index_title='Features with Missing Data')
                                                                       )
                                      })
                    if nan_heat_map:
                        _subplots.update({'Missing Data Heatmap': dict(data=self.df,
                                                                       features=[],
                                                                       plot_type='heat',
                                                                       kwargs=dict(z=self.df.isnull().astype(int).values,
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
            _info_table['invariant'] = '{} ({} %)'.format(_i, str(100 * round(_i / self.n_features, 4)))
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
        if duplicate_features:
            _data_health['duplicate'] = self._check_duplicates(by_row=True, by_col=True)
            if duplicate_cases:
                _dc = len(_data_health['duplicate']['cases'])
                _info_table['duplicate_cases'] = '{} ({} %)'.format(_dc, str(100 * round(_dc / self.n_cases, 4)))
                _index.append('Amount of duplicated cases')
            _d = len(_data_health['duplicate']['features'])
            _info_table['duplicate_features'] = '{} ({} %)'.format(_d, str(100 * round(_d / self.n_features, 4)))
            _index.append('Amount of duplicated features')
            if self.plot:
                _duplicate_cases: List[str] = []
                _duplicate_features: List[str] = []
                for i, ft in enumerate(self.features):
                    if i in _data_health['duplicate']['cases']:
                        _duplicate_cases.append('duplicate')
                    else:
                        _duplicate_cases.append('unique')
                    if ft in _data_health['duplicate']['features']:
                        _duplicate_features.append('duplicate')
                    else:
                        _duplicate_features.append('unique')
                _subplots.update({'Duplicate Cases': dict(data=pd.DataFrame(data=dict(duplicate_cases=_duplicate_cases)),
                                                          features=['duplicate_cases'],
                                                          plot_type='pie',
                                                          interactive=True
                                                          ),
                                  'Duplicate Features': dict(data=pd.DataFrame(data=dict(features=self.features, duplicate_features=_duplicate_features)),
                                                             features=['duplicate_features'],
                                                             plot_type='pie',
                                                             interactive=True
                                                             )
                                  })
        if outliers_univariate:
            _data_health['anomaly'] = self._check_outliers()
            _a = len(_data_health['anomaly'])
            _info_table['anomaly'] = '{} ({} %)'.format(_a, str(100 * round(_a / self.n_cases, 4)))
            _index.append('Amount of outlier cases')
            if self.plot:
                _outlier_cases: List[str] = (['outlier'] * _a) + (['inlier'] * (self.n_cases - _a))
                _subplots.update({'Cases containing univariate outliers': dict(data=pd.DataFrame(data=dict(uni_outlier=_outlier_cases)),
                                                                               features=['uni_outlier'],
                                                                               plot_type='pie'
                                                                               )
                                  })
        for mis_feature in _data_health['sparsity']['features']:
            _features.append(mis_feature)
        for inv_feature in _data_health['invariant']:
            _features.append(inv_feature)
        for dup_feature in _data_health['duplicate']['features']:
            _features.append(dup_feature)
        if duplicate_cases:
            for dup_case in _data_health['duplicate']['cases']:
                _cases.append(dup_case)
        for mis_case in _data_health['sparsity']['cases']:
            _cases.append(mis_case)
        if self.plot:
            _results_after_cleaning: pd.DataFrame = pd.DataFrame(columns=self.features, index=self.data_index)
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
                           subplots=_subplots,
                           interactive=True,
                           width=500,
                           height=500,
                           render=True if self.file_path is None else False,
                           file_path=self.file_path
                           ).run()
        return dict(cases=list(set(_cases)),
                    features=list(set(_features))
                    )

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
            if any(self.df[self.features[i]].isnull()):
                if len(self.df[self.features[i]].unique()) == 1:
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
                    if any(self.df[self.features[i]].isnull()):
                        _table['feature_type'].append('float')
                        _table['data_type'].append('continuous')
                        _table['rec'].append('Handle missing data')
                        _feature.append(self.features[i])
                else:
                    _table['feature_type'].append('float')
                    _feature.append(self.features[i])
                    if self.features[i] in self.feature_types.get('categorical'):
                        _table['data_type'].append('categorical')
                        if any(self.df[self.features[i]].isnull()):
                            _table['rec'].append('Handle missing data and convert to integer')
                        else:
                            _table['rec'].append('Convert to integer')
                        _typing[self.features[i]] = 'int'
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
                    if any(self.df[self.features[i]].isnull()):
                        _table['rec'].append('Handle missing data and convert to float')
                    else:
                        _table['rec'].append('Convert to float')
                elif self.features[i] in self.feature_types.get('categorical') or self.features[i] in self.feature_types.get('ordinal'):
                    _table['feature_type'].append('text')
                    _feature.append(self.features[i])
                    _table['data_type'].append('categorical')
                    _typing[self.features[i]] = 'int'
                    if any(self.df[self.features[i]].isnull()):
                        _table['rec'].append('Handle missing data and convert to integer by label encoding')
                    else:
                        _table['rec'].append('Convert to integer by label encoding')
        if self.plot:
            _df: pd.DataFrame = pd.DataFrame(data=_table)
            if _df.shape[0] == 0:
                Log(write=False).log('All feature and data types are correct')
            else:
                _df = _df.rename(columns={'feature_type': 'Feature Type', 'data_type': 'Data Type', 'rec': 'Recommendation'},
                                 index={idx: name for idx, name in zip(range(len(_feature)), _feature)}
                                 )
                _kwargs: dict = dict(index_title='Features')
                DataVisualizer(title='Data Type Check',
                               df=_df,
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
                  geo_features: List[str],
                  lat: str = None,
                  lon: str = None,
                  minimum: bool = True,
                  maximum: bool = True,
                  mean: bool = True,
                  std: bool = True,
                  perc: List[float] = None
                  ) -> dict:
        """
        Calculate statistics based on geographical area

        :param geo_features: List[str]
            Geographical features

        :param lat: str
            Name of the latitude feature

        :param lon: str
            Name of the longitude feature

        :param minimum: bool
            Calculate minimum

        :param maximum: bool
            Calculate maximum

        :param mean: bool
            Calculate mean

        :param std: bool
            Calculate standard deviation

        :param perc: List[float]
            Percentiles

        :return: dict
            Statistics based on geographical location
        """
        _geo_stats: dict = {}
        for geo in geo_features:
            _geo_stats.update({geo: {}})
            for val in self.df[geo].unique():
                _geo_stats[geo].update({val: {}})
                if val in self.invalid_values:
                    _df: pd.DataFrame = self.df.loc[self.df[geo].isnull(), :]
                else:
                    _df: pd.DataFrame = self.df.loc[self.df[geo] == val, :]
                for ft in self.feature_types.get('continuous'):
                    _geo_stats[geo][val].update({ft: dict(n=_df.shape[0])})
                    if minimum:
                        _geo_stats[geo][val][ft].update({'min': _df[ft].min()})
                    if maximum:
                        _geo_stats[geo][val][ft].update({'max': _df[ft].max()})
                    if mean:
                        _geo_stats[geo][val][ft].update({'mean': _df[ft].mean()})
                    if std:
                        _geo_stats[geo][val][ft].update({'std': _df[ft].std()})
        if self.plot:
            if lat is None:
                raise DataExplorerException('No latitude feature found')
            else:
                if lat not in self.df.keys():
                    raise DataExplorerException('Latitude feature not found in data set')
            if lon is None:
                raise DataExplorerException('No longitude feature found')
            else:
                if lon not in self.df.keys():
                    raise DataExplorerException('Longitude feature not found in data set')
            DataVisualizer(title='Geo Statistics',
                           subplots={},
                           interactive=True,
                           height=500,
                           width=500,
                           render=True if self.file_path is None else False,
                           file_path=self.file_path
                           ).run()
        return _geo_stats

    def outlier_detector(self,
                         kind: str = 'uni',
                         multi_meth: List[str] = None,
                         contour: bool = False
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
                -> knn: K-Nearest Neightbor

        :param contour: bool
            Generate contour chart

        :return: dict
            Detected outliers
        """
        _subplots: dict = {}
        _outlier: Dict[str, List[int]] = {}
        if kind is 'uni':
            _outlier.update({'uni': self._check_outliers()})
            _subplots.update({'Univariate Outlier Detection': dict(data=self.df,
                                                                   features=self.feature_types.get('continuous'),
                                                                   plot_type='violin',
                                                                   melt=True,
                                                                   kwargs=dict(layout=dict(points='outliers'))
                                                                   )
                              })
        elif kind in ['bi', 'multi']:
            if len(self.features) < 2:
                raise DataExplorerException('Not enough features for running a multivariate outlier detection')
            _multi_meth: List[str] = ['knn'] if multi_meth is None else multi_meth
            _anomaly_detection: dict = AnomalyDetector(df=self.df,
                                                       feature_types=self.feature_types
                                                       ).multivariate(contour_plot=contour)
            for meth in _multi_meth:
                _outlier.update({'multi': _anomaly_detection.get('cases')})
                self.df['outlier'] = _anomaly_detection[meth].get('pred')
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
                if len(self.features) == 2:
                    _multi.update(
                        {'scatter_inlier': dict(x=self.df.loc[self.df['outlier'] == 0, self.features[0]].values,
                                                y=self.df.loc[self.df['outlier'] == 0, self.features[1]].values,
                                                mode='markers',
                                                name='inlier',
                                                marker=dict(color='rgba(255, 255, 255, 1)') if contour else dict(color='rgba(107, 142, 35, 1)'),
                                                hoverinfo='text',
                                                showlegend=False if contour else True
                                                ),
                         'scatter_outlier': dict(x=self.df.loc[self.df['outlier'] == 1, self.features[0]].values,
                                                 y=self.df.loc[self.df['outlier'] == 1, self.features[1]].values,
                                                 mode='markers',
                                                 name='outlier',
                                                 marker=dict(color='rgba(0, 0, 0, 1)') if contour else dict(color='rgba(178, 34, 34, 1)'),
                                                 hoverinfo='text',
                                                 showlegend=False if contour else True
                                                 )
                         })
                    _subplots.update({'Multivariate Outlier Detection': dict(data=self.df,
                                                                             features=self.feature_types.get('continuous'),
                                                                             plot_type='multi',
                                                                             kwargs=dict(multi=_multi,
                                                                                         layout=dict(
                                                                                             xaxis=dict(title=self.features[0]),
                                                                                             yaxis=dict(title=self.features[1])
                                                                                             )
                                                                                         )
                                                                             )
                                      })
                else:
                    _pairs: List[tuple] = EasyExploreUtils().get_pairs(features=self.features, max_features_each_pair=2)
                    for i, pair in enumerate(_pairs):
                        _multi.update({'scatter_inlier_{}'.format(i): dict(x=self.df.loc[self.df['outlier'] == 0, pair[0]].values,
                                                                           y=self.df.loc[self.df['outlier'] == 0, pair[1]].values,
                                                                           mode='markers',
                                                                           name='inlier ({} | {})'.format(pair[0], pair[1]),
                                                                           marker=dict(color='rgba(255, 255, 255, 1)') if contour else dict(color='rgba(107, 142, 35, 1)'),
                                                                           showlegend=True
                                                                           ),
                                       'scatter_outlier_{}'.format(i): dict(x=self.df.loc[self.df['outlier'] == 1, pair[0]].values,
                                                                            y=self.df.loc[self.df['outlier'] == 1, pair[1]].values,
                                                                            mode='markers',
                                                                            name='outlier ({} | {})'.format(pair[0], pair[1]),
                                                                            marker=dict(color='rgba(0, 0, 0, 1)') if contour else dict(color='rgba(178, 34, 34, 1)'),
                                                                            showlegend=True
                                                                            )
                                       })
                        if kind is 'bi':
                            DataVisualizer(title='Bivariate Outlier Detection',
                                           df=self.df,
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
                    if kind is 'multi':
                        _subplots.update({'Multivariate Outlier Detection': dict(data=self.df,
                                                                                 features=self.feature_types.get('continuous'),
                                                                                 plot_type='multi',
                                                                                 render=True if self.file_path is None else False,
                                                                                 file_path=self.file_path,
                                                                                 kwargs=dict(multi=_multi)
                                                                                 )
                                          })
        else:
            raise DataExplorerException('Type of outlier detection ({}) not supported'.format(type))
        if self.plot:
            if kind is not 'bi':
                DataVisualizer(subplots=_subplots,
                               interactive=True,
                               height=500,
                               width=500,
                               render=True if self.file_path is None else False,
                               file_path=self.file_path
                               ).run()
        return _outlier


class MissingDataAnalysis:
    """
    Class for missing data analysis
    """
    def __init__(self,
                 data: np.array,
                 features: List[str],
                 other_mis: list = None
                 ):
        """
        :param data: Numpy array containing the data set
        :param other_mis#: List of values to convert to Numpy missing values
        """
        self.data_set = data
        self.features = features
        self.other_mis = other_mis if other_mis is not None else []
        if len(self.other_mis) > 0:
            self.data_set = self._set_nan()

    def _set_nan(self) -> np.array:
        """

        Set missing data value

        :return: Numpy array containing pre-processed data set
        """
        for other_mis in self.other_mis:
            self.data_set[np.where(self.data_set == other_mis)] = np.nan
        return self.data_set

    def freq_nan(self) -> dict:
        """

        Frequency of missing data

        :return: Pandas DataFrame containing the absolute and relative frequency of missing data of each feature and case
        """
        _freq_nan = {'features': {'abs': {}, 'rel': {}}, 'cases': {'abs': {}, 'rel': {}}}
        _mis = np.where(pd.isnull(self.data_set))
        if len(_mis[0]) > 0 or len(_mis[1]) > 0:
            for case in np.unique(_mis[0]):
                _nan_case_wise = len(np.where(pd.isnull(self.data_set[case, :]))[0])
                _freq_nan['cases']['abs'].update({case: _nan_case_wise})
                _freq_nan['cases']['rel'].update({case: 100 * round(_nan_case_wise / self.data_set[case, :].shape[0], 6)})
            for feature in np.unique(_mis[1]):
                _nan_feature_wise = len(np.where(pd.isnull(self.data_set[:, feature]))[0])
                _freq_nan['features']['abs'].update({feature: _nan_feature_wise})
                _freq_nan['features']['rel'].update({feature: 100 * round(_nan_feature_wise / self.data_set[:, feature].shape[0], 6)})
        return _freq_nan
