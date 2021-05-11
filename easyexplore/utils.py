import collections
import copy
import dask
import geojson
import glob
import itertools
import logging
import os
import numpy as np
import networkx as nx
import pandas as pd
import re
import subprocess
import zipfile

from .data_import_export import DataExporter, FileUtils
from dask import dataframe as dd
from dask.distributed import Client
from datetime import datetime
from ipywidgets import FloatProgress
from IPython.display import display, HTML
from itertools import islice
from scipy.stats import anderson, chi2, chi2_contingency, f_oneway, friedmanchisquare, mannwhitneyu, normaltest, kendalltau,\
                        kruskal, kstest, pearsonr, powerlaw, shapiro, spearmanr, stats, ttest_ind, ttest_rel, wilcoxon
from statsmodels.stats.weightstats import ztest
from typing import Dict, List, Tuple, Union

INVALID_VALUES: list = ['nan', 'NaN', 'NaT', np.nan, 'none', 'None', 'inf', '-inf', np.inf, -np.inf] # None
PERCENTILES: List[float] = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
SPECIAL_CHARACTERS: List[str] = [' ', '^', '°', '!', '"', "'", '§', '$', '%', '&', '/', '(', ')', '=', '?', '`', '´',
                                 '<', '>', '|', '@', '€', '*', '+', '#', '-', '_', '.', ',', ':', ';'
                                 ]
SPECIAL_SEPARATORS: List[str] = ['!!', '%%', '==', '<<', '>>', '&&', '||', '##', '--', '__', '::', ';;']


class EasyExplore:
    """
    Class for handling logging activities
    """
    def __init__(self,
                 log_path: str = None,
                 show_msg: bool = True,
                 show_header: bool = False,
                 show_progress_bar: bool = False,
                 show_ram_usage: bool = True
                 ):
        """
        :param log_path: str
            Path of the log file

        :param show_msg: bool
            Enable and disable message display

        :param show_header: bool
            Enable and disable header display

        :param show_progress_bar: bool
            Enable and disable progress bar display

        :param show_ram_usage: bool
            Enable and disable ram usage display
        """
        self.display: bool = True
        self.data_shape: tuple = tuple([None, None])
        self.header_title: str = ''
        self.show_msg: bool = show_msg
        self.show_header: bool = show_header
        self.show_pb: bool = show_progress_bar
        self.show_ram: bool = show_ram_usage
        self.header: dict = dict(header=dict(id='easyexplore',
                                             color='olive',
                                             font_family='Courier',
                                             font_size='15px'
                                             ),
                                 title=dict(id='easyexplore',
                                            color='grey',
                                            font_family='Courier',
                                            font_size='15px'
                                            )
                                 )
        self.pb: FloatProgress = FloatProgress(value=0.0,
                                               min=0,
                                               max=100.0,
                                               step=0.1,
                                               description='EasyExplore',
                                               bar_style='success',
                                               orientation='horizontal'
                                               )
        self.pb_max: int = 100
        self.log = logging.getLogger(name=__name__)
        self.log.setLevel(level=logging.DEBUG)
        if log_path is None:
            self.log_path = '{}/log/'.format(os.getcwd().replace('\\', '/'))
        else:
            self.log_path = log_path
            if not os.path.exists(log_path):
                os.mkdir(log_path)
        self.invalid_values: List[str] = ['', 'nan', 'NaN', 'NaT', np.nan, 'none', 'None', None, 'inf', '-inf', np.inf, -np.inf]
        self.perc: List[float] = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.timestamp_format: str = '%Y-%m-%d %H:%M:%S'
        self.formatter = {'dev': '%(asctime)s - %(levelname)s - %(message)s',
                          'stage': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                          'prod': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
                          }

    def display_header(self, title: str, data_shape: Tuple[str, str]):
        """
        Display header

        :param title: str
            Title text

        :param data_shape: Tuple[str, str]
            Number of cases and features
        """
        _header: HTML = HTML('<p id="{}"><span style="font-size:{};color:{};font-family:{};"><i>EasyExplore</i></span></p>'.format(self.header['header'].get('id'),
                                                                                                                                   self.header['header'].get('color'),
                                                                                                                                   self.header['header'].get('font_family'),
                                                                                                                                   self.header['header'].get('font_size')
                                                                                                                                   )
                             )
        _title: HTML = HTML('<p id="{}"><span style="font-size:{};color:{};font-family:{};">{}</span><span style="font-size:15px;">Cases = {}  |  Features = {}</span></p>'.format(self.header['header'].get('id'),
                                                                                                                                                                                   self.header['title'].get('color'),
                                                                                                                                                                                   self.header['title'].get('font_family'),
                                                                                                                                                                                   self.header['title'].get('font_size'),
                                                                                                                                                                                   title,
                                                                                                                                                                                   data_shape[0],
                                                                                                                                                                                   data_shape[1]
                                                                                                                                                                                   )
                            )
        display(_title)

    def display_pb(self):
        """
        Display progress bar
        """
        display(self.pb, include=None, exclude=None, metadata=None, transient=None, display_id=None)

    def display_msg(self, msg: str):
        """
        Display messages as HTML widget

        :param msg: str
            Message to display
        """
        _msg: HTML = HTML(
            '<p id="{}"><span style="font-size:{};color:{};font-family:{};"><b>{}</b></span></p>'.format(
                self.header['header'].get('id'),
                self.header['header'].get('color'),
                self.header['header'].get('font_family'),
                self.header['header'].get('font_size'),
                msg
            )
        )
        display(_msg)

    def printer(self, msg: str):
        """
        Internal message printer
        """
        if self.show_msg:
            if self.show_ram:
                #print('\n{}: RAM usage: {}%: {}\n'.format(datetime.now().strftime(self.timestamp_format),
                #                                          psutil.virtual_memory().percent,
                #                                          msg
                #                                          )
                #      )
                print('\n{}: {}\n'.format(datetime.now().strftime(self.timestamp_format), msg))
            else:
                print('\n{}: {}\n'.format(datetime.now().strftime(self.timestamp_format), msg))


class Log:
    """
    Class for handling logging
    """
    def __init__(self,
                 write: bool = False,
                 level: str = 'info',
                 env: str = 'dev',
                 logger_file_path: str = None,
                 log_ram_usage: bool = True,
                 log_cpu_usage: bool = True
                 ):
        """
        :param write: bool
            Write logging file or not

        :param level: str
            Name of the logging level of the messge
                -> info: Logs any information
                -> warn: Logs warnings
                -> error: Logs errors including critical messages

        :param env: str
            Name of the logging environment to use
                -> dev: Development - Logs any information
                -> stage: Staging - Logs only warnings and errors including critical messages
                -> prod: Production - Logs only errors including critical messages

        :param logger_file_path: str
            Complete file path of the logger file

        :param log_ram_usage: bool
            Log RAM usage (in percent) or not

        :param log_cpu_usage: bool
            Log CPU usage (in percent) or not
        """
        self.write: bool = write
        self.timestamp_format: str = '%Y-%m-%d %H:%M:%S'
        if log_ram_usage:
            #self.ram: str = ' -> RAM {}%'.format(psutil.virtual_memory().percent)
            self.ram: str = ''
        else:
            self.ram: str = ''
        if log_cpu_usage:
            #self.cpu: str = ' -> CPU {}%'.format(psutil.cpu_percent(percpu=False))
            self.cpu: str = ''
        else:
            self.cpu: str = ''
        self.msg: str = '{}{}{} | '.format(datetime.now().strftime(self.timestamp_format), self.ram, self.cpu)
        if write:
            if logger_file_path is None:
                self.log_file_path: str = os.path.join(os.getcwd(), 'log.txt')
            else:
                self.log_file_path: str = logger_file_path
            FileUtils(file_path=self.log_file_path, create_dir=True).make_dir()
        else:
            self.log_file_path: str = None
        self.levels: List[str] = ['info', 'warn', 'error']
        _env: dict = dict(dev=0, stage=1, prod=2)
        if env in _env.keys():
            self.env: int = _env.get(env)
        else:
            self.env: int = _env.get('dev')
        if level in self.levels:
            self.level: int = self.levels.index(level)
        else:
            self.level: int = 0

    def _write(self):
        """
        Write log file
        """
        with open(file=self.log_file_path, mode='a', encoding='utf-8') as _log:
            _log.write('{}\n'.format(self.msg))

    def log(self, msg: str):
        """
        Log message

        :param msg: str
            Message to log
        """
        if self.level >= self.env:
            if self.level == 0:
                _level_description: str = ''
            elif self.level == 1:
                _level_description: str = 'WARNING: '
            elif self.level == 2:
                _level_description: str = 'ERROR: '
            self.msg = '{}{}'.format(self.msg, msg)
            if self.write:
                self._write()
            else:
                print(self.msg)


class StatsUtils:
    """
    Class for calculating univariate and multivariate statistics
    """
    def __init__(self, data, features: List[str], partitions: int = 1):
        """
        :param data: Pandas or dask DataFrame
            Data set

        :param features: List[str]
            Name of the features

        :param partitions: int
            Number of partitions to create using parallel computing framework dask
        """
        if isinstance(data, pd.DataFrame):
            self.df: dd.DataFrame = dd.from_pandas(data=data, npartitions=partitions)
        elif isinstance(data, dd.DataFrame):
            self.df: dd.DataFrame = data
        self.features = features
        self.n_features: int = len(features)
        self.n_cases: int = len(self.df)
        self.p: float = 0.95
        self.nan_policy = 'omit'

    def _anderson_darling_test(self, feature: str, sig_level: float = 0.05) -> float:
        """
        Anderson-Darling test for normality tests

        :param feature: str
            Name of the feature

        :param sig_level: float
            Level of significance

        :return: float
            Probability describing significance level
        """
        _stat = anderson(x=self.df[feature].values.compute(), dist='norm')
        try:
            _i: int = _stat.significance_level.tolist().index(100 * sig_level)
            p: float = _stat.critical_values[_i]
        except ValueError:
            p: float = _stat.critical_values[2]
        return p

    def _bartlette_sphericity_test(self) -> dict:
        """
        Bartlette's test for sphericity
        """
        _cor = self.df[self.features].corr('pearson').compute()
        _cor_det = np.linalg.det(_cor.values)
        _statistic: np.ndarray = -np.log(_cor_det) * (self.n_cases - 1 - (2 * self.n_features + 5) / 6)
        _dof = self.n_features * (self.n_features - 1) / 2
        return dict(statistic=_statistic, p=chi2.pdf(_statistic, _dof))

    def _dagostino_k2_test(self, feature: str) -> float:
        """
        D'Agostino K² test for normality

        :param feature: str
            Name of the feature

        :return: float
            Statistical probability value (p-value)
        """
        stat, p = normaltest(a=self.df[feature].values.compute(), axis=0, nan_policy='propagate')
        return p

    def _kaiser_meyer_olkin_test(self) -> dict:
        """
        Kaiser-Meyer-Olkin test for unobserved features
        """
        _cor = self.correlation(meth='pearson').values
        _partial_cor = self.correlation(meth='partial').values
        np.fill_diagonal(_cor, 0)
        np.fill_diagonal(_partial_cor, 0)
        _cor = _cor ** 2
        _partial_cor = _partial_cor ** 2
        _cor_sum = np.sum(_cor)
        _partial_cor_sum = np.sum(_partial_cor)
        _cor_sum_feature = np.sum(_cor, axis=0)
        _partial_cor_sum_feature = np.sum(_partial_cor, axis=0)
        return dict(kmo=_cor_sum / (_cor_sum + _partial_cor_sum),
                    kmo_per_feature=_cor_sum_feature / (_cor_sum_feature + _partial_cor_sum_feature),
                    )

    def _shapiro_wilk_test(self, feature: str) -> float:
        """
        Shapiro-Wilk test for normality tests

        :param feature: str
            Name of the feature

        :return: float
            Statistical probability value (p-value)
        """
        return shapiro(x=self.df[feature].values.compute())

    def curtosis_test(self) -> List[str]:
        """
        Test whether a distribution is tailed or not

        :return: List[str]
            Names of the tailed features
        """
        raise NotImplementedError('Method not supported yet')

    def correlation(self, meth: str = 'pearson', min_obs: int = 1) -> pd.DataFrame:
        """
        Calculate correlation coefficients

        :param meth: str
            Method to be used as correlation coefficient
                -> pearson: Marginal Correlation based on Pearson's r
                -> kendall: Rank Correlation based on Kendall
                -> spearman: Rank Correlation based on Spearman
                -> partial: Partial Correlation
        :param min_obs: int
            Minimum number of valid observations

        :return: pd.DataFrame
            Correlation matrix
        """
        if meth in ['pearson', 'kendall', 'spearman']:
            _cor: pd.DataFrame = self.df[self.features].corr(method=meth, min_periods=min_obs).compute()
        elif meth == 'partial':
            if len(self.df) - self.df.isnull().astype(dtype=int).sum().sum().compute() > 0:
                _cov: np.ndarray = np.cov(m=self.df[self.features].dropna())
                try:
                    assert np.linalg.det(_cov) > np.finfo(np.float32).eps
                    _inv_var_cov: np.ndarray = np.linalg.inv(_cov)
                except AssertionError:
                    _inv_var_cov: np.ndarray = np.linalg.pinv(_cov)
                    #warnings.warn('The inverse of the variance-covariance matrix '
                    #              'was calculated using the Moore-Penrose generalized '
                    #              'matrix inversion, due to its determinant being at '
                    #              'or very close to zero.')
                _std: np.ndarray = np.sqrt(np.diag(_inv_var_cov))
                _cov2cor: np.ndarray = _inv_var_cov / np.outer(_std, _std)
                _cor: pd.DataFrame = pd.DataFrame(data=np.nan_to_num(x=_cov2cor, copy=True) * -1,
                                                  columns=self.features,
                                                  index=self.features
                                                  )
            else:
                _cor: pd.DataFrame = pd.DataFrame()
                Log(write=False, level='info').log(msg='Can not calculate coefficients for partial correlation because of the high missing data rate')
        else:
            raise EasyExploreUtilsException('Method for calculating correlation coefficient ({}) not supported'.format(meth))
        return _cor

    def correlation_test(self,
                         x: str,
                         y: str,
                         meth: str = 'pearson',
                         freq_table: List[float] = None,
                         yates_correction: bool = True,
                         power_divergence: str = 'cressie_read'
                         ) -> dict:
        """
        :param x:
        :param y:
        :param meth: String defining the hypothesis test method for correlation
                        -> pearson:
                        -> spearman:
                        -> kendall:
                        -> chi-squared:
        :param freq_table:
        :param yates_correction:
        :param power_divergence: String defining the power divergence test method used in chi-squared independent test
                                    -> pearson: Pearson's chi-squared statistic
                                    -> log-likelihood: Log-Likelihood ratio (G-test)
                                    -> freeman-tukey: Freeman-tukey statistic
                                    -> mod-log-likelihood: Modified log-likelihood ratio
                                    -> neyman: Neyman's statistic
                                    -> cressie-read: Cressie-Read power divergence test statistic
        :return:
        """
        _reject = None
        if meth == 'pearson':
            _correlation_test = pearsonr(x=self.df[x].values.compute(), y=self.df[y].values.compute())
        elif meth == 'spearman':
            _correlation_test = spearmanr(a=self.df[x].values.compute(), b=self.df[y].values.compute(), axis=0, nan_policy=self.nan_policy)
        elif meth == 'kendall':
            _correlation_test = kendalltau(x=self.df[x].values.compute(), y=self.df[y].values.compute(), nan_policy=self.nan_policy)
        elif meth == 'chi-squared':
            _correlation_test = chi2_contingency(observed=freq_table, correction=yates_correction, lambda_=power_divergence)
        else:
            raise EasyExploreUtilsException('Method for correlation test not supported')
        if _correlation_test[1] <= self.p:
            _reject = False
        else:
            _reject = True
        return {'feature': ''.join(self.df.keys()),
                'cases': len(self.df.values),
                'test_statistic': _correlation_test[0],
                'p_value': _correlation_test[1],
                'reject': _reject}

    def factoriability_test(self, meth: str = 'kmo') -> dict:
        """
        Test whether a data set contains unobserved features required for factor analysis

        :param meth: str
            Name of the used method
                -> kmo: Kaiser-Meyer-Olkin Criterion
                -> bartlette: Bartlette's test of sphericity
        """
        _fac: dict = {}
        if meth == 'kmo':
            pass
        elif meth == 'bartlette':
            pass
        else:
            raise EasyExploreUtilsException('Method for testing "factoriability" ({}) not supported'.format(meth))
        return {}

    def non_parametric_test(self,
                            x: str,
                            y: str,
                            meth: str = 'kruskal-wallis',
                            continuity_correction: bool = True,
                            alternative: str = 'two-sided',
                            zero_meth: str = 'pratt',
                            *args
                            ):
        """
        :param x:
        :param y:
        :param meth: String defining the hypothesis test method for non-parametric tests
                        -> kruskal-wallis: Kruskal-Wallis H test to test whether the distributions of two or more
                                           independent samples are equal or not
                        -> mann-whitney: Mann-Whitney U test to test whether the distributions of two independent
                                         samples are equal or not
                        -> wilcoxon: Wilcoxon Signed-Rank test for test whether the distributions of two paired samples
                                     are equal or not
                        -> friedman: Friedman test for test whether the distributions of two or more paired samples
                                     are equal or not
        :param continuity_correction:
        :param alternative: String defining the type of hypothesis test
                            -> two-sided:
                            -> less:
                            -> greater:
        :param zero_meth: String defining the method to handle zero differences in the ranking process (Wilcoxon test)
                            -> pratt: Pratt treatment that includes zero-differences (more conservative)
                            -> wilcox: Wilcox tratment that discards all zero-differences
                            -> zsplit: Zero rank split, just like Pratt, but spliting the zero rank between positive
                                       and negative ones
        :param args:
        :return:
        """
        _reject = None
        if meth == 'kruskal-wallis':
            _non_parametric_test = kruskal(args, self.nan_policy)
        elif meth == 'mann-whitney':
            _non_parametric_test = mannwhitneyu(x=self.df[x].values.compute(),
                                                y=self.df[y].values.compute(),
                                                use_continuity=continuity_correction,
                                                alternative=alternative)
        elif meth == 'wilcoxon':
            _non_parametric_test = wilcoxon(x=self.df[x].values.compute(),
                                            y=self.df[y].values.compute(),
                                            zero_method=zero_meth,
                                            correction=continuity_correction)
        elif meth == 'friedman':
            _non_parametric_test = friedmanchisquare(args)
        else:
            raise ValueError('No non-parametric test found !')
        if _non_parametric_test[1] <= self.p:
            _reject = False
        else:
            _reject = True
        return {'feature': ''.join(list(self.df.columns)),
                'cases': self.n_cases,
                'test_statistic': _non_parametric_test[0],
                'p_value': _non_parametric_test[1],
                'reject': _reject
                }

    def normality_test(self, alpha: float = 0.05, meth: str = 'shapiro-wilk') -> dict:
        """
        Test whether a distribution is normal distributed or not

        :param alpha: float
            Threshold that indicates whether a hypothesis can be rejected or not

        :param meth: str
            Method to test normality
                -> shapiro-wilk:
                -> anderson-darling:
                -> dagostino:
        :return: dict
            Results of normality test (statistic, p-value, p > alpha)
        """
        _alpha = alpha
        _normality: dict = {}
        for feature in self.features:
            if meth == 'shapiro-wilk':
                _stat, _p = self._shapiro_wilk_test(feature=feature)
            elif meth == 'anderson-darling':
                _stat, _p = self._anderson_darling_test(feature=feature, sig_level=alpha)
            elif meth == 'dagostino':
                _stat, _p = self._dagostino_k2_test(feature=feature)
            else:
                raise EasyExploreUtilsException('Method ({}) for testing normality not supported'.format(meth))
            _normality.update({feature: dict(stat=_stat, p=_p, normality=_p > _alpha)})
        return _normality

    def parametric_test(self, x: str, y: str, meth: str = 't-test', welch_t_test: bool = True, *args):
        """
        :param x:
        :param y:
        :param meth: String defining the hypothesis test method for parametric tests
                        -> z-test:
                        -> t-test:
                        -> t-test-paired:
                        -> anova:
        :param welch_t_test:
        :param args: Arguments containing samples from two or more groups for anova test
        :return:
        """
        _reject = None
        if meth == 't-test':
            _parametric_test = ttest_ind(a=self.df[x].values.compute(), b=self.df[y].values.compute(),
                                         axis=0, equal_var=not welch_t_test, nan_policy=self.nan_policy)
        elif meth == 't-test-paired':
            _parametric_test = ttest_rel(a=self.df[x].values.compute(), b=self.df[y].values.compute(), axis=0, nan_policy=self.nan_policy)
        elif meth == 'anova':
            _parametric_test = f_oneway(args)
        elif meth == 'z-test':
            _parametric_test = ztest(x1=x, x2=y, value=0, alternative='two-sided', usevar='pooled', ddof=1)
        else:
            raise ValueError('No parametric test found !')
        if _parametric_test[1] <= self.p:
            _reject = False
        else:
            _reject = True
        return {'feature': ''.join(list(self.df.columns)),
                'cases': self.n_cases,
                'test_statistic': _parametric_test[0],
                'p_value': _parametric_test[1],
                'reject': _reject
                }

    def power_law_test(self,
                       tail_prob: List[float] = None,
                       shape_params: List[float] = None,
                       location_params: List[float] = None,
                       size: Tuple[int] = None,
                       moments: str = 'mvsk'
                       ):
        """
        :param tail_prob:
        :param shape_params:
        :param location_params:
        :param size:
        :param moments:
        :return:
        """
        raise NotImplementedError('Method not implemented yet')

    def skewness_test(self, axis: str = 'col', threshold_interval: Tuple[float, float] = (-0.5, 0.5)) -> dict:
        """
        Test whether a distribution is skewed or not

        :param axis: str
            Name of the axis of the data frame to use
                -> col: Test skewness of feature
                -> row: test skewness of cases

        :param threshold_interval: Tuple[float, float]
            Threshold interval for testing

        :return: dict
            Statistics regarding skewness of features
        """
        if axis == 'col':
            _axis = 0
        elif axis == 'row':
            _axis = 1
        else:
            raise EasyExploreUtilsException('Axis ({}) not supported'.format(axis))
        return self.df[self.features].compute().skew(axis=_axis).to_dict()


class EasyExploreUtilsException(Exception):
    """
    Class for setting up exceptions for class EasyExploreUtils
    """
    pass


class EasyExploreUtils:
    """
    Class for applying general utility methods
    """
    @staticmethod
    def _get_analytical_type(df: pd.DataFrame,
                             feature: str,
                             dtype: List[np.dtype],
                             continuous: List[str] = None,
                             categorical: List[str] = None,
                             ordinal: List[str] = None,
                             date: List[str] = None,
                             id_text: List[str] = None,
                             date_edges: Tuple[str, str] = None,
                             max_categories: int = 100
                             ) -> Dict[str, str]:
        """
        Get analytical data type of feature using dask for parallel computing

        :param df:
            Pandas or dask DataFrame

        :param feature: str
            Name of the feature

        :param dtype: List[np.dtype]
            Numpy dtypes of each feature

        :param continuous: List[str]
            Name of the continuous features

        :param categorical: List[str]
            Name of the categorical features

        :param ordinal: List[str]
            Name of the ordinal features

        :param date: List[str]
            Name of the date features

        :param id_text: List[str]
            Name of the identifier or text features

        :param max_categories: int
            Maximum number of categories for identifying feature as categorical

        :return Dict[str, str]:
            Analytical data type and feature name
        """
        if date is not None:
            if feature in date:
                return {'date': feature}
        if ordinal is not None:
            if feature in ordinal:
                return {'ordinal': feature}
        if categorical is not None:
            if feature in categorical:
                return {'categorical': feature}
        if continuous is not None:
            if feature in continuous:
                return {'continuous': feature}
        if id_text is not None:
            if feature in id_text:
                return {'id_text': feature}
        _feature_data = df.loc[~df[feature].isnull(), feature]
        if str(dtype).find('float') >= 0:
            _unique = _feature_data.unique()
            if any(_feature_data.isnull()):
                if any(_unique[~pd.isnull(_unique)] % 1) != 0:
                    return {'continuous': feature}
                else:
                    if len(str(_feature_data.min()).split('.')[0]) > 4:
                        try:
                            assert pd.to_datetime(_feature_data)
                            if date_edges is None:
                                return {'date': feature}
                            else:
                                if (date_edges[0] < pd.to_datetime(_unique.min())) or (
                                        date_edges[1] > pd.to_datetime(_unique.max())):
                                    return {'id_text': feature}
                                else:
                                    return {'date': feature}
                        except (TypeError, ValueError):
                            return {'id_text': feature}
                    else:
                        if len(_unique) > max_categories:
                            return {'ordinal': feature}
                        else:
                            return {'categorical': feature}
            else:
                if any(_unique % 1) != 0:
                    return {'continuous': feature}
                else:
                    if len(str(_feature_data.min()).split('.')[0]) > 4:
                        try:
                            assert pd.to_datetime(_feature_data)
                            if date_edges is None:
                                return {'date': feature}
                            else:
                                if (date_edges[0] < pd.to_datetime(_unique.min())) or (
                                        date_edges[1] > pd.to_datetime(_unique.max())):
                                    return {'id_text': feature}
                                else:
                                    return {'date': feature}
                        except (TypeError, ValueError):
                            return {'id_text': feature}
                    else:
                        if len(_feature_data) == len(_feature_data.unique()):
                            return {'id_text': feature}
                        if len(_unique) > max_categories:
                            return {'ordinal': feature}
                        else:
                            return {'categorical': feature}
        elif str(dtype).find('int') >= 0:
            if len(_feature_data) == len(_feature_data.unique()):
                return {'id_text': feature}
            else:
                if len(_feature_data.unique()) > max_categories:
                    return {'ordinal': feature}
                else:
                    return {'categorical': feature}
        elif str(dtype).find('object') >= 0:
            _unique: np.array = _feature_data.unique()
            _digits: int = 0
            _dot: bool = False
            _max_dots: int = 0
            for text_val in _unique:
                if text_val == text_val:
                    if (str(text_val).find('.') >= 0) or (str(text_val).replace(',', '').isdigit()):
                        _dot = True
                    if str(text_val).replace('.', '').replace('-', '').isdigit() or str(text_val).replace(',',
                                                                                                          '').replace(
                        '-', '').isdigit():
                        if (len(str(text_val).split('.')) == 2) or (len(str(text_val).split(',')) == 2):
                            _digits += 1
                    if len(str(text_val).split('.')) > _max_dots:
                        _max_dots = len(str(text_val).split('.'))
            if _digits >= (len(_unique[~pd.isnull(_unique)]) * 0.5):
                if _dot:
                    try:
                        if any(_unique[~pd.isnull(_unique)] % 1) != 0:
                            return {'continuous': feature}
                        else:
                            if _max_dots == 2:
                                return {'continuous': feature}
                            else:
                                return {'id_text': feature}
                    except (TypeError, ValueError):
                        if _max_dots == 2:
                            return {'continuous': feature}
                        else:
                            return {'id_text': feature}
                else:
                    if len(_feature_data) == len(_feature_data.unique()):
                        return {'id_text': feature}
                    _len_of_feature = pd.DataFrame()
                    _len_of_feature[feature] = _feature_data[~_feature_data.isnull()]
                    _len_of_feature['len'] = _len_of_feature[feature].str.len()
                    _unique_values: np.array = _len_of_feature['len'].unique()
                    if len(_feature_data.unique()) >= (len(_feature_data) * 0.5):
                        return {'id_text': feature}
                    else:
                        if len(_feature_data.unique()) > max_categories:
                            return {'ordinal': feature}
                        else:
                            return {'categorical': feature}
            else:
                try:
                    _potential_date = _feature_data[~_feature_data.isnull()]
                    _unique_years = pd.to_datetime(_potential_date).dt.year.unique()
                    _unique_months = pd.to_datetime(_potential_date).dt.isocalendar().week.unique()
                    _unique_days = pd.to_datetime(_potential_date).dt.day.unique()
                    _unique_cats: int = len(_unique_years) + len(_unique_months) + len(_unique_days)
                    if _unique_cats > 4:
                        return {'date': feature}
                    else:
                        if len(_feature_data) == len(_feature_data.unique()):
                            return {'id_text': feature}
                        if len(_feature_data.unique().values) <= 3:
                            return {'categorical': feature}
                        else:
                            _len_of_feature = pd.DataFrame()
                            _len_of_feature[feature] = _feature_data[~_feature_data.isnull()]
                            _len_of_feature['len'] = _len_of_feature[feature].str.len()
                            _unique_values: np.array = _len_of_feature['len'].unique()
                            for val in _unique_values:
                                if len(re.findall(pattern=r'[a-zA-Z]', string=str(val))) > 0:
                                    if len(_feature_data.unique()) >= (len(_feature_data) * 0.5):
                                        return {'id_text': feature}
                                    else:
                                        return {'categorical': feature}
                            if np.min(_unique_values) > 3:
                                if len(_feature_data.unique()) >= (len(_feature_data) * 0.5):
                                    return {'id_text': feature}
                                else:
                                    if len(_feature_data.unique().values) > max_categories:
                                        return {'ordinal': feature}
                                    else:
                                        return {'categorical': feature}
                            else:
                                return {'categorical': feature}
                except (TypeError, ValueError):
                    if len(_feature_data) == len(_unique):
                        return {'id_text': feature}
                    if len(_feature_data.unique()) <= 3:
                        return {'categorical': feature}
                    else:
                        _len_of_feature = _feature_data[~_feature_data.isnull()]
                        _len_of_feature['len'] = _len_of_feature.str.len()
                        _unique_values: np.array = _len_of_feature['len'].unique()
                        for val in _feature_data.unique():
                            if len(re.findall(pattern=r'[a-zA-Z]', string=str(val))) > 0:
                                if len(_feature_data.unique()) >= (len(_feature_data) * 0.5):
                                    return {'id_text': feature}
                                else:
                                    if len(_feature_data.unique()) > max_categories:
                                        return {'ordinal': feature}
                                    else:
                                        return {'categorical': feature}
                        for ch in SPECIAL_CHARACTERS:
                            if any(_len_of_feature.str.find(ch) > 0):
                                if len(_feature_data.unique()) >= (len(_feature_data) * 0.5):
                                    return {'id_text': feature}
                                else:
                                    return {'categorical': feature}
                        # if np.mean(_unique_values) == np.median(_unique_values):
                        #    return {'id_text': feature}
                        # else:
                        return {'categorical': feature}
        elif str(dtype).find('date') >= 0:
            return {'date': feature}
        elif str(dtype).find('bool') >= 0:
            return {'categorical': feature}

    def check_dtypes(self,
                     df: Union[dd.DataFrame, pd.DataFrame],
                     feature_types: Dict[str, List[str]] = None,
                     date_edges: Tuple[str, str] = None,
                     ignore_features: List[str] = None
                     ) -> dict:
        """
        Check if data types of Pandas DataFrame match with the analytical measurement of data

        :param df: Pandas DataFrame or dask dataframe
            Data set

        :param feature_types: Dict[str, List[str]]
            Pre-defined feature types
                -> id_text
                -> categorical
                -> ordinal
                -> continuous
                -> date

        :param date_edges: Tuple[str, str]
            Date edges for identifying feature as date feature

        :param ignore_features: List[str]
            Name of the features to ignore

        :return dict
            Data type conversion recommendation
        """
        _typing: dict = dict(meta={}, conversion={})
        _features: List[str] = list(df.columns)
        _dtypes: List[str] = [str(dt) for dt in df.dtypes.tolist()]
        if feature_types is None:
            _feature_types: Dict[str, List[str]] = self.get_feature_types(df=df,
                                                                          features=_features,
                                                                          dtypes=df.dtypes.tolist(),
                                                                          date_edges=date_edges
                                                                          )
        else:
            _feature_types: Dict[str, List[str]] = feature_types
        if isinstance(df, dd.DataFrame):
            _df: dd.DataFrame = df
        elif isinstance(df, pd.DataFrame):
            _df: dd.DataFrame = dd.from_pandas(data=df, npartitions=3)
        else:
            raise EasyExploreUtilsException('Format of data input ({}) not supported. Use Pandas or dask DataFrame instead'.format(type(df)))
        if ignore_features is not None:
            for ignore_feature in ignore_features:
                del _features[_features.index(ignore_feature)]
        _table: Dict[str, List[str]] = {'feature_type': [], 'data_type': [], 'rec': []}
        for i in range(0, len(_features), 1):
            if any(_df[_features[i]].isnull().compute()):
                if len(_df[_features[i]].unique().values.compute()) == 1:
                    _typing['meta'].update({_features[i]: dict(data_type='unknown',
                                                               feature_type='float',
                                                               rec='Drop feature (no valid data)'
                                                               )
                                            })
                    continue
            if str(_dtypes[i]).find('bool') >= 0:
                _typing['meta'].update({_features[i]: dict(data_type='categorical',
                                                           feature_type='bool',
                                                           rec='Convert to integer'
                                                           )
                                        })
                _typing['conversion'].update({_features[i]: 'int'})
            elif str(_dtypes[i]).find('float') >= 0:
                if _features[i] in _feature_types.get('date'):
                    _typing['meta'].update({_features[i]: dict(data_type='date',
                                                               feature_type='float',
                                                               rec='Convert to datetime'
                                                               )
                                            })
                    _typing['conversion'].update({_features[i]: 'date'})
                elif _features[i] in _feature_types.get('ordinal'):
                    _typing['meta'].update({_features[i]: dict(data_type='categorical',
                                                               feature_type='float',
                                                               rec='Convert to integer'
                                                               )
                                            })
                    _typing['conversion'].update({_features[i]: 'int'})
                elif _features[i] in _feature_types.get('continuous'):
                    if any(_df[_features[i]].isnull().compute()):
                        _typing['meta'].update({_features[i]: dict(data_type='continuous',
                                                                   feature_type='float',
                                                                   rec='Handle missing data'
                                                                   )
                                                })
                else:
                    _typing['meta'].update({_features[i]: dict(data_type='categorical',
                                                               feature_type='float'
                                                               )
                                            })
                    _typing['conversion'].update({_features[i]: 'int'})
                    if _features[i] in _feature_types.get('categorical'):
                        if any(_df[_features[i]].isnull().compute()):
                            _typing['meta'][_features[i]].update({'rec': 'Handle missing data and convert to integer'})
                        else:
                            _typing['meta'][_features[i]].update({'rec': 'Convert to integer'})
            elif str(_dtypes[i]).find('int') >= 0:
                if _features[i] not in _feature_types.get('categorical'):
                    _typing['meta'].update({_features[i]: dict(feature_type='int')})
                    if _features[i] in _feature_types.get('ordinal'):
                        continue
                    elif _features[i] in _feature_types.get('date'):
                        _typing['meta'].update({_features[i]: dict(data_type='date',
                                                                   rec='Convert to datetime'
                                                                   )})
                        _typing['conversion'].update({_features[i]: 'date'})
                    elif _features[i] in _feature_types.get('id_text'):
                        _typing['meta'].update({_features[i]: dict(data_type='text',
                                                                   rec='Convert to string'
                                                                   )})
                        _typing['conversion'].update({_features[i]: 'str'})
            elif str(_dtypes[i]).find('object') >= 0:
                if _features[i] in _feature_types.get('date'):
                    _typing['meta'].update({_features[i]: dict(data_type='date',
                                                               feature_type='text',
                                                               rec='Convert to datetime'
                                                               )
                                            })
                    _typing['conversion'].update({_features[i]: 'date'})
                elif _features[i] in _feature_types.get('continuous'):
                    _typing['meta'].update({_features[i]: dict(data_type='continuous',
                                                               feature_type='text'
                                                               )
                                            })
                    _typing['conversion'].update({_features[i]: 'float'})
                    if any(_df[_features[i]].isnull().compute()):
                        _typing['meta'][_features[i]].update({'rec': 'Handle missing data and convert to float'})
                    else:
                        _typing['meta'][_features[i]].update({'rec': 'Convert to float'})
                elif _features[i] in _feature_types.get('categorical') or _features[i] in _feature_types.get('ordinal'):
                    _typing['meta'].update({_features[i]: dict(data_type='categorical',
                                                               feature_type='text'
                                                               )
                                            })
                    _typing['conversion'].update({_features[i]: 'int'})
                    if any(_df[_features[i]].isnull().compute()):
                        _typing['meta'][_features[i]].update({'rec': 'Handle missing data and convert to integer by label encoding'})
                    else:
                        _typing['meta'][_features[i]].update({'rec': 'Convert to integer by label encoding'})
        return _typing

    @staticmethod
    def convert_jupyter(notebook_name: str, to: str = 'html'):
        """
        Convert Jupyter Notebook into several formats

        :param notebook_name: str
            Name of the jupyter notebook

        :param to: str
            Output format
                -> html: HTML
                -> pdf: PDF
                -> latex: Latex
                -> markdown: Markdown
                -> rst: reStructuredText
                -> script: Python / Julia / R script (depending on kernel settings of ipynb file)
        """
        if to in ['html', 'pdf', 'latex', 'markdown', 'rst', 'script']:
            subprocess.run(['jupyter nbconvert "{}" --to {}'.format(notebook_name, to)], shell=True)
        else:
            raise EasyExploreUtilsException('Jupyter notebook could not be converted into "{}" file'.format(to))

    @staticmethod
    def dask_setup(client_name: str, client_address: str = None, mode: str = 'threads', **kwargs) -> Client:
        """
        Setup dask framework for parallel computation

        :param client_name: str
            Name of the client

        :param client_address: str
            Address end point

        :param mode: str
            Parallel computation mode:
                threads: Multi-Threading
                processes: Multi-Processing
                single-threaded: Single thread and process

        :param kwargs: dict
            Key-word arguments for dask client implementation

        :return: Client
            Initialized dask client object
        """
        if mode == 'threads':
            dask.config.set(scheduler='threads')
        elif mode == 'processes':
            dask.config.set(scheduler='processes')
        else:
            dask.config.set(scheduler='single-threaded')
        if kwargs.get('memory_limit') is None:
            kwargs.update({'memory_limit': 'auto'})
        return Client(address=client_address,
                      loop=kwargs.get('loop'),
                      timeout=1000 if kwargs.get('timeout') is None else kwargs.get('timeout'),
                      set_as_default=True,
                      scheduler_file=kwargs.get('scheduler_file'),
                      security=kwargs.get('security'),
                      asynchronous=kwargs.get('asynchronous'),
                      name=client_name,
                      heartbeat_interval=kwargs.get('heartbeat_interval'),
                      serializers=kwargs.get('serializers'),
                      deserializers=kwargs.get('deserializers'),
                      direct_to_workers=kwargs.get('direct_to_workers'),
                      #connection_limit=512 if kwargs.get('connection_limit') else kwargs.get('connection_limit'),
                      processes=False if kwargs.get('processes') is None else kwargs.get('processes'),
                      **kwargs
                      )

    @staticmethod
    def extract_tuple_el_in_list(list_of_tuples: List[tuple], tuple_pos: int) -> list:
        """
        Extract specific tuple elements from list of tuples

        :param list_of_tuples: List[tuple]
            List of tuples

        :param tuple_pos: int
            Position of element in tuples to extract

        :return: list
            List of elements of tuple
        """
        if tuple_pos < 0:
            raise EasyExploreUtilsException('Position of element in tuple cannot be negative ({})'.format(tuple_pos))
        return next(islice(zip(*list_of_tuples), tuple_pos, None))

    @staticmethod
    def freedman_diaconis_bins(data_points: np.array) -> int:
        """
        Calculate the width of each bin by using Freedmann-Diaconis rule

        :param data_points: np.array
            Data points to plot

        :return: int
            Number of bins to compute
        """
        data: np.array = np.asarray(data_points, dtype=np.float_)
        iqr = stats.iqr(data, rng=(25, 75), scale='raw', nan_policy='omit')
        rng = max(data) - min(data)
        return int((rng / ((2 * iqr) / np.power(data.size, 1 / 3))) + 1)

    @staticmethod
    def freedman_diaconis_width(data_points: np.array) -> float:
        """
        Calculate the width of each bin by using Freedmann-Diaconis rule

        :param data_points: np.array
            Data points to plot

        :return: float
            Width of each bin
        """
        data: np.ndarray = np.asarray(data_points, dtype=np.float_)
        iqr = stats.iqr(data, rng=(25, 75), scale='raw', nan_policy='omit')
        return (2 * iqr) / np.power(data.size, 1 / 3)

    @staticmethod
    def generate_git_ignore(file_path: str, exclude_files: List[str] = None, exclude_default: bool = True):
        """
        Generate .gitignore file

        :param file_path: str
            File path

        :param exclude_files: List[str]
            Names of files or objects to be ignored

        :param exclude_default: bool
            Exclude default files and objects
        """
        _gitignore: str = ''
        _default: str = '#########\n# misc: ##########\n.git\n.idea\n.DS_Store\n\n########### python: ###########\nvenv\n**/.cache\n**/__pycache__\n.pytest_cache\n\n###################### jupyter notebook: ######################\n.ipynb_checkpoints\n\n################## data sources: ##################\n*.db\n*.txt\n'
        if exclude_default:
            _gitignore = _default
        if exclude_files is not None:
            if len(exclude_files) > 0:
                for f in exclude_files:
                    _gitignore = _gitignore + f
        DataExporter(obj=_gitignore, file_path='{}.gitignore'.format(file_path), create_dir=False, overwrite=False).file()

    @staticmethod
    def generate_network(df: pd.DataFrame, node_feature: str, edge_feature: str, kind: str = 'undirected', **kwargs) -> nx:
        """
        Generate network graph

        :param df: pd.DataFrame
            Data set

        :param node_feature: str
            Name of the feature used to generate nodes from

        :param edge_feature: str
            Name of the feature used to generate edges from

        :param kind: str
            Network type
                -> directed: Bayes network
                -> undirected: Markov Network
                -> geometric: Geometric Network based on x, y scale

        :return nx
            Preconfigured Networkx network graph object
        """
        if kind == 'directed':
            _graph: nx.DiGraph = nx.DiGraph()
        elif kind == 'undirected':
            _graph: nx.Graph = nx.Graph()
        elif kind == 'geometric':
            _graph: nx = nx.random_geometric_graph(n=kwargs.get('n'),
                                                   radius=kwargs.get('radius'),
                                                   dim=2 if kwargs.get('dim') is None else kwargs.get('dim'),
                                                   pos=kwargs.get('pos'),
                                                   p=2 if kwargs.get('p') is None else kwargs.get('p'),
                                                   seed=None
                                                   )
        else:
            raise EasyExploreUtilsException('Network graph type ({}) not supported'.format(kind))
        if node_feature not in df.keys():
            raise EasyExploreUtilsException('Node feature ({}) not found'.format(node_feature))
        if edge_feature not in df.keys():
            raise EasyExploreUtilsException('Edge feature ({}) not found'.format(edge_feature))
        _group_by = df.groupby(by=edge_feature).aggregate({node_feature: 'count'})[1:]
        _nodes: list = df[node_feature].unique().tolist()
        _graph.add_nodes_from(nodes_for_adding=_nodes)
        _node_val: list = _group_by.index.values.tolist()
        _edge_val: list = _group_by[node_feature].values.tolist()
        for i in range(0, _group_by.shape[0], 1):
            if i in _nodes:
                _graph.add_edge(u_of_edge=_node_val[i], v_of_edge=_edge_val[i])
            else:
                _graph.add_edge(u_of_edge=i, v_of_edge=0)
        return _graph

    @staticmethod
    def get_duplicates(df, cases: bool = True, features: bool = True) -> Dict[str, list]:
        """
        Get duplicate cases and / or features

        :param df: Pandas or dask DataFrame
            Data set

        :param cases: bool
            Check whether cases are duplicated or not

        :param features: bool
            Check whether features are duplicated or not

        :return Dict[str, list]
            Duplicated cases and / or features
        """
        if isinstance(df, pd.DataFrame):
            _df: pd.DataFrame = df
        elif isinstance(df, dd.DataFrame):
            _df: pd.DataFrame = df.compute()
        else:
            raise EasyExploreUtilsException('Format of data set ({}) not supported. Use Pandas DataFrame or dask dataframe instead'.format(type(df)))
        _duplicates: dict = dict(cases=[], features=[])
        if cases:
            _duplicates['cases'] = _df.loc[_df.duplicated(), :].index.values.tolist()
        if features:
            _duplicates['features'] = _df.loc[:, _df.transpose().duplicated()].columns.tolist()
        return _duplicates

    def get_feature_types(self,
                          df: pd.DataFrame,
                          features: List[str],
                          dtypes: List[np.dtype],
                          continuous: List[str] = None,
                          categorical: List[str] = None,
                          ordinal: List[str] = None,
                          date: List[str] = None,
                          id_text: List[str] = None,
                          date_edges: Tuple[str, str] = None,
                          max_categories: int = 100,
                          multi_threading: bool = False,
                          print_msg: bool = False
                          ) -> Dict[str, List[str]]:
        """
        Get feature types

        :param df: Pandas DataFrame or dask DataFrame
            Data set

        :param features: List[str]
            Names of features

        :param dtypes: np.dtype
            Data type of each feature

        :param continuous: List[str]
            Names of pre-defined continuous features

        :param categorical: List[str]
            Names of pre-defined categorical features

        :param ordinal: List[str]
            Names of pre-defined ordinal features

        :param date: List[str]
            Names of pre-defined date features

        :param id_text: List[str]
            Names of pre-defined id_text features

        :param date_edges:
            Minimum and maximum time for identifying date features

        :param max_categories: int
            Maximum number of categories for identifying feature as categorical

        :param multi_threading: bool
            Run feature type interpretation on multiple threads

        :param print_msg: bool
            Print segmentation message

        :return: dict
            List of feature names for each feature type
        """
        if print_msg:
            Log(write=False, level='info').log(msg='Start feature type segmentation...')
        _feature_types: Dict[str, List[str]] = dict(id_text=[], categorical=[], ordinal=[], date=[], continuous=[])
        if date_edges is None:
            _date_edges = None
        else:
            try:
                assert pd.to_datetime(date_edges[0])
                assert pd.to_datetime(date_edges[1])
                _date_edges: tuple = tuple([pd.to_datetime(date_edges[0]), pd.to_datetime(date_edges[1])])
            except Exception as e:
                _date_edges = None
                Log(write=False, level='warning').log(msg='Date edges ({}) cannot be converted into datetime\nError: {}'.format(date_edges, e))
        _threads: dict = {}
        for i, feature in enumerate(features):
            if multi_threading:
                _delayed_analytical_type: dask.delayed = dask.delayed(self._get_analytical_type(df=df,
                                                                                                feature=feature,
                                                                                                dtype=dtypes[i],
                                                                                                continuous=continuous,
                                                                                                categorical=categorical,
                                                                                                ordinal=ordinal,
                                                                                                date=date,
                                                                                                id_text=id_text,
                                                                                                date_edges=_date_edges,
                                                                                                max_categories=max_categories
                                                                                                )
                                                                      )
                _threads.update({i: _delayed_analytical_type})
            else:
                _analytical_type: Dict[str, str] = self._get_analytical_type(df=df,
                                                                             feature=feature,
                                                                             dtype=dtypes[i],
                                                                             continuous=continuous,
                                                                             categorical=categorical,
                                                                             ordinal=ordinal,
                                                                             date=date,
                                                                             id_text=id_text,
                                                                             date_edges=_date_edges,
                                                                             max_categories=max_categories
                                                                             )
                _type: str = list(_analytical_type.keys())[0]
                _feature: str = _analytical_type[list(_analytical_type.keys())[0]]
                _feature_types[copy.deepcopy(_type)].append(copy.deepcopy(_feature))
        if multi_threading:
            for thread in _threads.keys():
                _analytical_type: Dict[str, str] = _threads.get(thread).compute()
                _type: str = list(_analytical_type.keys())[0]
                _feature: str = _analytical_type[list(_analytical_type.keys())[0]]
                _feature_types[copy.deepcopy(_type)].append(copy.deepcopy(_feature))
        if print_msg:
            Log(write=False, level='info').log(msg='Segmentation finished')
        return _feature_types

    @staticmethod
    def get_freq(data: list) -> collections.Counter:
        """
        Get frequency of elements in a list

        :param data: list:
            Data set

        :return: dict:
            Frequency of elements
        """
        return collections.Counter(data)

    @staticmethod
    def get_geojson(df: pd.DataFrame,
                    lat: np.array,
                    lon: np.array,
                    features: List[str],
                    fig: str = 'polygon',
                    save: bool = False
                    ) -> dict:
        """
        Generate geojson dictionary from pandas data frame

        :param df:
        :param lat:
        :param lon:
        :param features:
        :param fig:
        :param save:
        :return: dict: Geojson
        """
        features = []
        insert_features = lambda X: features.append(geojson.Feature(geometry=geojson.Polygon((X["long"],
                                                                                              X["lat"],
                                                      # X["elev"]
                                                      )
                                                     ),
                            properties=dict(name=X["name"],
                                            description=X["description"])))
        df.apply(insert_features, axis=1)
        if save:
            with open('/Users/balistrerig/PycharmProjects/data_science/src/data_science/test/data/test.geojson', 'w',
                      encoding='utf8') as fp:
                geojson.dump(geojson.FeatureCollection(features), fp, sort_keys=True, ensure_ascii=False)
        return {}

    @staticmethod
    def get_list_of_files(file_path: str) -> List[str]:
        """
        Get list of file in given directory or zip file

        :param file_path: str
            Complete file path

        :return: List[str]
            Name of detected files
        """
        if os.path.exists(file_path):
            if file_path.split('.')[-1] == 'zip':
                return zipfile.ZipFile(file_path).namelist()
            else:
                return [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
        else:
            raise EasyExploreUtilsException('File path not found ({})'.format(file_path))

    @staticmethod
    def get_list_of_objects(file_path: str) -> List[str]:
        """
        Get list of objects in given directory and subdirectories

        :param file_path: str
            Complete file path

        :return: List[str]
            List of object names detected in directory
        """
        return [obj for obj in os.listdir(file_path)]

    @staticmethod
    def get_invariant_features(df) -> List[str]:
        """
        Get invariant features of data set

        :param df: Pandas or dask DataFrame
            Data set

        :return List[str]
            Names of invariant features
        """
        if isinstance(df, pd.DataFrame):
            _df: dd.DataFrame = dd.from_pandas(data=df, npartitions=1)
        elif isinstance(df, dd.DataFrame):
            _df: dd.DataFrame = df
            if _df.npartitions > 1:
                _df = _df.repartition(npartitions=1)
        else:
            raise EasyExploreUtilsException('Format of data set ({}) not supported. Use Pandas or dask DataFrame instead'.format(type(df)))
        _invariant_features: List[str] = []
        for ft in _df.columns:
            _unique_values: np.ndarray = _df[ft].unique().values.compute()
            if len(_unique_values) <= 2:
                if len(_unique_values) == 1:
                    _invariant_features.append(ft)
                else:
                    if any(_df[ft].isnull().compute()):
                        _invariant_features.append(ft)
        return _invariant_features

    @staticmethod
    def get_pairs(features: List[str], max_features_each_pair: int = 2) -> List[tuple]:
        """
        Get pairs of feature list

        :param features: List[str]
            Features to pair

        :param max_features_each_pair: int
            Maximum number of features for each pair

        :return: List[tuple]
            Features pairs
        """
        _features: List[str] = []
        for feature in features:
            if feature not in _features:
                _features.append(feature)
        return [pair for pair in itertools.combinations(iterable=_features, r=max_features_each_pair)]

    @staticmethod
    def get_group_by_percentile(data: pd.DataFrame,
                                group_by: str,
                                aggregate_by: List[str],
                                aggregation: str = 'median',
                                percentiles: int = 10,
                                duplicates: str = 'drop',
                                include_group: bool = True
                                ) -> pd.DataFrame:
        """
        Generate percentile evaluation of of grouped features

        :param data: pd.DataFrame:
            Data set to calculate percentiles from

        :param group_by: str
            Grouping features

        :param aggregate_by: List[str]
            Features to aggregate

        :param aggregation: str
            Aggregation method

        :param percentiles: int
            Number of percentiles to generate

        :param duplicates: str
            Handle duplicates
                -> raise: Raise exception
                -> drop: Drop

        :param include_group: bool
            Include grouping feature inbto aggregation process

        :return: pd.DataFrame
            Aggregated data set
        """
        _aggre: List[str] = []
        _df: pd.DataFrame = pd.DataFrame()
        if group_by in data.keys():
            _df[group_by] = data[group_by]
            if len(aggregate_by) == 0:
                raise EasyExploreUtilsException('No features for aggregation found')
            for agg in aggregate_by:
                if agg in data.keys():
                    _aggre.append(agg)
                    _df[agg] = data[agg]
        else:
            raise EasyExploreUtilsException('No feature for grouping found')
        if include_group:
            _aggre.append(group_by)
        _q: tuple = pd.qcut(_df[group_by], q=percentiles, retbins=True, duplicates='drop')
        _perc_labels: np.array = _q[0].unique()
        _perc_edges: list = _q[1].tolist()
        _df['perc'] = pd.qcut(x=_df[group_by],
                              q=percentiles,
                              labels=_perc_labels,
                              retbins=False,
                              precision=4,
                              duplicates=duplicates
                              )
        _df_perc = _df.groupby(by=['perc']).agg({a: aggregation for a in _aggre})
        return _df_perc

    @staticmethod
    def get_perc_eval(pred: list, obs: list, aggregation: str = 'median', percentiles: int = 10) -> pd.DataFrame:
        """
        Generate percentile evaluation of predictions and observation

        :param pred: list
            Predictions

        :param obs: list
            Observations

        :param aggregation: str
            Aggregation method
                -> min: Minimum
                -> max: Maximum
                -> median: Median
                -> mean: Mean

        :param percentiles: int
            Number of percentiles to calculate

        :return: pd.DataFrame
            Percentile evaluation of prediction and observation
        """
        _df = pd.DataFrame({'obs': obs, 'preds': pred})
        _df['pred_perc'] = pd.qcut(x=_df['obs'], q=percentiles, labels=np.arange(1, percentiles + 1), retbins=False, precision=4, duplicates='drop')
        _df_perc = _df.groupby(by=['pred_perc']).agg({'preds': aggregation, 'obs': aggregation})
        _df_perc['abs_diff'] = _df_perc['preds'] - _df_perc['obs']
        _df_perc['rel_diff'] = (_df_perc['preds'] - _df_perc['obs']) / _df_perc['obs']
        return _df_perc

    @staticmethod
    def label_encoder(values: np.array) -> np.array:
        """
        Label encoder

        :param values: np.array
            Values to encode into numeric values
        """
        _unique: list = list(pd.unique(values))
        _enc_val: list = []
        for val in values:
            _enc_val.append(_unique.index(val))
        return np.array(_enc_val)

    @staticmethod
    def replace_dict_keys(d: dict, new_keys: List[str]) -> dict:
        """
        Replace keys of a dictionary by values of given list

        :param d: dict
            Dictionary in which keys should be replaced

        :param new_keys: List[str]
            New key names

        :return dict:
            Adjusted dictionary
        """
        _d = dict()
        for k in d.keys():
            _d[k] = new_keys[k]
        return _d

    @staticmethod
    def search_for_file(key_word: str, starting_dir: str) -> List[str]:
        """
        Search for files with similar name patterns as key word input

        :param key_word: str
            Searched key word

        :param starting_dir: str
            Directory to start searching in

        :return: List[str]
            Names of the files found under given key word
        """
        return glob.glob(pathname=key_word)

    @staticmethod
    def subset_dict(d: dict, threshold: float) -> dict:
        """
        Subset dictionary by given threshold value

        :param d: dict
            Dictionary containing data

        :param threshold: float
            Threshold for subsetting

        :return dict
            Subsetted dictionary
        """
        _d = {}
        for k in d.keys():
            if d.get(k) >= threshold:
                _d[k] = d.get(k)
        return _d
