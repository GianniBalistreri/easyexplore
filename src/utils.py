import geojson
import glob
import itertools
import json
import logging
import os
import numpy as np
import networkx as nx
import pandas as pd
import psutil
import random
import subprocess
import zipfile

from .data_import_export import DataExporter, DataImporter, FileUtils
from datetime import datetime
from ipywidgets import FloatProgress
from IPython.display import Image, display, HTML
from itertools import accumulate, islice
from operator import mul
from scipy.stats import anderson, chi2, chi2_contingency, f_oneway, friedmanchisquare, mannwhitneyu, normaltest, kendalltau,\
                        kruskal, kstest, pearsonr, powerlaw, shapiro, spearmanr, stats, ttest_ind, ttest_rel, wilcoxon
from statsmodels.stats.weightstats import ztest
from typing import Dict, List, Mapping, Tuple


MAX_INTEGER: int = 100000000000
MIN_MAX_FLOAT: Tuple[float, float] = (1e-320, 1e+300)
INVALID_VALUES: list = ['nan', 'NaN', 'NaT', np.nan, 'none', 'None', 'inf', '-inf', np.inf, -np.inf] # None
PERCENTILES: List[float] = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

# TODO:
#  Write unit test


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
        :param log_path: str: Path of the log file
        :param show_msg: bool: Enable and disable message display
        :param show_header: bool: Enable and disable header display
        :param show_progress_bar: bool: Enable and disable progress bar display
        :param show_ram_usage: bool: Enable and disable ram usage display
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

        :param str title:
        :param Tuple[str, str] data_shape:
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

        :param str msg: Message to display
        :return:
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
                print('\n{}: RAM usage: {}%: {}\n'.format(datetime.now().strftime(self.timestamp_format),
                                                          psutil.virtual_memory().percent,
                                                          msg
                                                          )
                      )
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
            self.ram: str = ' -> RAM {}%'.format(psutil.virtual_memory().percent)
        else:
            self.ram: str = ''
        if log_cpu_usage:
            self.cpu: str = ' -> CPU {}%'.format(psutil.cpu_percent(percpu=False))
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


class Logger:
    """

    Class for handling logging

    """
    def __init__(self,
                 write: bool = False,
                 logger_name: str = None,
                 logger_file_path: str = None,
                 logger_file_env: str = 'prod',
                 logger_level: str = 'debug',
                 log_ram_usage: bool = True,
                 log_cpu_usage: bool = False
                 ):
        """
        Parameters
        ----------
        write: bool
            Write message to log file or print it in console

        logger_name: str
            Name of the logger
                -> None: Name of the current python file is used

        logger_file_path: str
            Complete file path of the logger
                -> None: Path of current working directory is used

        logger_file_env: str
            Set logger file environment
                -> dev: Write logging file at INFO level
                -> stage: Write logging file at WARNING level
                -> prod: Write logging file at ERROR level

        logger_level: str
            Set logger level
                -> debug: Level 1
                -> info: Level 2
                -> warning: Level 3
                -> error: Level 4
                -> critical: level 5
        """
        self.write: bool = write
        if log_ram_usage:
            _ram: str = ' - RAM {}% - '.format(psutil.virtual_memory().percent)
        else:
            _ram: str = ' - '
        if log_cpu_usage:
            _cpu: str = ' - CPU {}% - '.format(psutil.cpu_percent(percpu=False))
        else:
            _cpu: str = ' - '
        self.formatter = {'dev': '%(asctime)s {} %(message)s'.format(_ram, _cpu),
                          'stage': '%(asctime)s {} %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'.format(_ram, _cpu),
                          'prod': '%(asctime)s {} %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'.format(_ram, _cpu)
                          }
        if write:
            if logger_file_path is None:
                self.log_file_path: str = os.path.join(os.getcwd(), 'log.txt')
            else:
                self.log_file_path: str = logger_file_path
            FileUtils(file_path=self.log_file_path, create_dir=True).make_dir()
        else:
            self.log_file_path: str = None
        if logger_file_env in self.formatter.keys():
            _env: str = logger_file_env
        else:
            _env: str = 'dev'
        if logger_level == 'debug':
            _level: logging = logging.DEBUG
        elif logger_level == 'info':
            _level: logging = logging.INFO
        elif logger_level == 'warning':
            _level: logging = logging.WARNING
        elif logger_level == 'error':
            _level: logging = logging.ERROR
        elif logger_level == 'critical':
            _level: logging = logging.CRITICAL
        else:
            _level: logging = logging.DEBUG
        logging.basicConfig(filename=self.log_file_path, filemode='a', format=self.formatter.get(_env), level=_level)
        _name: str = __name__ if logger_name is None else logger_name
        self.log: logging.getLogger = logging.getLogger(_name)

    def set(self) -> logging:
        """
        Set logger

        Returns
        -------
        logging: Preconfigured logger object
        """
        return self.log


class RequestUtilsException(Exception):
    """

    Class for handling exceptions for class RequestUtils

    """
    pass


class RequestUtils(FileUtils):
    """

    Class for handling requests

    """
    def __init__(self, url: List[str], parallel: bool = True, file_path: str = None):
        """
        Parameters
        ----------
        url: List[str]
            URL's for requesting payloads

        parallel: bool
            Request payloads in parallel or not

        file_path: str
            Complete file path either of imported files for sending or exported files for receiving payload
        """
        super().__init__(file_path=file_path, create_dir=True)
        self.url: List[str] = url
        self.parallel: bool = parallel
        self.file_path: str = file_path
        self.payload = None

    def get_payload(self):
        pass

    def send_payload(self):
        pass


class StatsUtils:
    """

    Class for calculating univariate and multivariate statistics

    """

    def __init__(self,
                 data: pd.DataFrame,
                 features: List[str]
                 ):
        """
        :param data:
        :param features:
        """
        self.data_set = data
        self.features = features
        self.nan_policy = 'omit'

    def _anderson_darling_test(self, feature: str, sig_level: float = 0.05) -> float:
        """

        Anderson-Darling test for normality tests

        :param feature:
        :param sig_level:
        :return: float: Probability describing significance level
        """
        _stat = anderson(x=self.data_set[feature], dist='norm')
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
        _n_cases, _n_features = self.data_set.shape
        _cor = self.data_set[self.features].corr('pearson')
        _cor_det = np.linalg.det(_cor.values)
        _statistic: np.ndarray = -np.log(_cor_det) * (_n_cases - 1 - (2 * _n_features + 5) / 6)
        _dof = _n_features * (_n_features - 1) / 2
        return dict(statistic=_statistic, p=chi2.pdf(_statistic, _dof))

    def _dagostino_k2_test(self, feature: str) -> float:
        """

        D'Agostino K² test for normality

        :param feature: String containing the name of the feature
        :return: Float indicating the statistical probability value (p-value)
        """
        stat, p = normaltest(a=self.data_set[feature], axis=0, nan_policy='propagate')
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

        :param feature: String containing the name of the feature
        :return: Float indicating the statistical probability value (p-value)
        """
        return shapiro(x=self.data_set[feature])

    def curtosis_test(self) -> List[str]:
        """

        Test whether a distribution is tailed or not

        :return: List of strings containing the names of the tailed features
        """
        raise UtilsException('Method not supported yet')

    def correlation(self, meth: str = 'pearson', min_obs: int = 1) -> pd.DataFrame:
        """

        Calculate correlation coefficients

        :param meth: String containing the method to be used as correlation coefficient
                        -> pearson: Marginal Correlation based on Pearson's r
                        -> kendall: Rank Correlation based on Kendall
                        -> spearman: Rank Correlation based on Spearman
                        -> partial: Partial Correlation
        :param min_obs: Integer indicating the minimum amount of valid observations
        :return: Pandas DataFrame containing the correlation matrix
        """
        if meth in ['pearson', 'kendall', 'spearman']:
            _cor: pd.DataFrame = self.data_set[self.features].corr(method=meth, min_periods=min_obs)
        elif meth == 'partial':
            if self.data_set.shape[0] - self.data_set.isnull().astype(dtype=int).sum().sum() > 0:
                _cov: np.ndarray = np.cov(m=self.data_set[self.features].dropna())
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
                                                  index=self.features)
            else:
                _cor: pd.DataFrame = pd.DataFrame()
                Log(write=False, level='info').log(msg='Can not calculate coefficients for partial correlation because of the high missing data rate')
        else:
            raise UtilsException('Method for calculating correlation coefficient ({}) not supported'.format(meth))
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
            _correlation_test = pearsonr(x=self.data_set[x], y=self.data_set[y])
        elif meth == 'spearman':
            _correlation_test = spearmanr(a=self.data_set[x], b=self.data_set[y], axis=0, nan_policy=self.nan_policy)
        elif meth == 'kendall':
            _correlation_test = kendalltau(x=self.data_set[x], y=self.data_set[y], nan_policy=self.nan_policy)
        elif meth == 'chi-squared':
            _correlation_test = chi2_contingency(observed=freq_table, correction=yates_correction, lambda_=power_divergence)
        else:
            raise UtilsException('Method for correlation test not supported')
        if _correlation_test[1] <= self.p:
            _reject = False
        else:
            _reject = True
        return {'feature': ''.join(self.data_set.keys()),
                'cases': len(self.data_set.values),
                'test_statistic': _correlation_test[0],
                'p_value': _correlation_test[1],
                'reject': _reject}

    def factoriability_test(self, meth: str = 'kmo') -> dict:
        """

        Test whether a data set contains unobserved features required for factor analysis

        :param meth: String containing the name of the used method
                        -> kmo: Kaiser-Meyer-Olkin Criterion
                        -> bartlette: Bartlette's test of sphericity
        """
        _fac: dict = {}
        if meth == 'kmo':
            pass
        elif meth == 'bartlette':
            pass
        else:
            raise UtilsException('Method for testing "factoriability" ({}) not supported'.format(meth))
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
            _non_parametric_test = mannwhitneyu(x=self.data_set[x],
                                                y=self.data_set[y],
                                                use_continuity=continuity_correction,
                                                alternative=alternative)
        elif meth == 'wilcoxon':
            _non_parametric_test = wilcoxon(x=self.data_set[x],
                                            y=self.data_set[y],
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
        return {'feature': ''.join(self.data_set.keys()),
                'cases': len(self.data_set.values),
                'test_statistic': _non_parametric_test[0],
                'p_value': _non_parametric_test[1],
                'reject': _reject}

    def normality_test(self, alpha: float = 0.05, meth: str = 'shapiro-wilk') -> dict:
        """

        Test whether a distribution is normal distributed or not

        :param alpha: Float indicating the threshold that indicates whether a hypothesis can be rejected or not
        :param meth: String containing the method to test normality
                        -> shapiro-wilk:
                        -> anderson-darling:
                        -> dagostino:
        :return: dict: Results of normality test (statistic, p-value, p > alpha)
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
                raise UtilsException('Method ({}) for testing normality not supported'.format(meth))
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
            _parametric_test = ttest_ind(a=self.data_set[x], b=self.data_set[y],
                                         axis=0, equal_var=not welch_t_test, nan_policy=self.nan_policy)
        elif meth == 't-test-paired':
            _parametric_test = ttest_rel(a=self.data_set[x], b=self.data_set[y], axis=0, nan_policy=self.nan_policy)
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
        return {'feature': ''.join(self.data_set.keys()),
                'cases': len(self.data_set.values),
                'test_statistic': _parametric_test[0],
                'p_value': _parametric_test[1],
                'reject': _reject}

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
        raise UtilsException('Method not implemented yet :-(')

    def skewness_test(self, axis: str = 'col', threshold_interval: Tuple[float, float] = (-0.5, 0.5)) -> dict:
        """

        Test whether a distribution is skewed or not

        :param axis: String containing the name of the axis of the data frame to use
                        -> col: Test skewness of feature
                        -> row: test skewness of cases
        :param threshold_interval: Tuple of floats indicating the threshold interval for testing
        :return: List of strings containing the name of the skewed features
        """
        if axis == 'col':
            _axis = 0
        elif axis == 'row':
            _axis = 1
        else:
            raise UtilsException('Axis ({}) not supported'.format(axis))
        return self.data_set[self.features].skew(axis=_axis).to_dict()


class PalladiumUtils:
    """

    Class for running machine learning models on palladium framework

    """
    def __init__(self, file_path: str):
        """
        :param file_path: String containing the complete file path either of the config.py or request.json
        """
        self.full_path: str = file_path.replace('\\', '/')
        self.file_name: str = file_path.replace('\\', '/').split('/')[-1]
        self.file_path: str = file_path.replace(self.file_name, '')

    def generate_config_py(self,
                           train_path: str = None,
                           test_path: str = None,
                           local: bool = True,
                           sep: str = ',',
                           sql_query: str = None,
                           use_numpy: bool = False
                           ):
        """

        Generate pre-configured config.py

        :param train_path: String containing the path of the train data set (either data base or file)
        :param test_path: String containing the path of the test data set (either data base or file)
        :param local: Boolean indicating whether to generate config.py file for local testing or not
        :param sep: String containing the separator value
        :param sql_query: String containing the SQL query to fetch data from data base
        :param use_numpy: Boolean indicating whether to use Numpy ndarray or Pandas DataFrame
        """
        _config_py = {'data_loader_train': dict(),
                      'data_loader_test': dict(),
                      'model': dict(),
                      'predict_service': dict(),
                      'model_persister': dict()
                      }
        if local:
            _config_py['data_loader_train'].update({'__factory__': 'palladium.dataset.Table'})
            if train_path is None:
                raise UtilsException('Path for the train data set is empty')
            _config_py['data_loader_train'].update({'path': train_path})
            _config_py['data_loader_train'].update({'sep': sep})
            _config_py['data_loader_train'].update({'ndarray': use_numpy})
            _config_py['data_loader_test'].update({'__factory__': 'palladium.dataset.Table'})
            if test_path is None:
                _config_py['data_loader_test']['path'] = ''
            else:
                _config_py['data_loader_test'].update({'path': train_path})
            _config_py['data_loader_test'].update({'sep': sep})
            _config_py['data_loader_test'].update({'ndarray': use_numpy})
        else:
            _config_py['data_loader_train'].update({'__factory__': 'palladium.dataset.SQL'})
        _config_py['model'].update({'__factory__': '{}.request_adapter.RequestAdapter'.format(__name__)})
        _config_py['predict_service'].update({'__factory__': '{}.request_adapter.RequestAdapter'.format(__name__),
                                              'mapping': [],
                                              'entry_point': '/{}'.format(__name__)
                                              })
        _config_py['model_persister'].update({'__factory__': 'palladium.persistence.CachedUpdatePersister',
                                              'impl': {'__factory__': 'palladium.persistence.Database',
                                                       'url': 'sqlite:///model.db'
                                                       }
                                              })
        DataExporter(obj=_config_py,
                     file_path=self.full_path,
                     overwrite=True,
                     create_dir=True
                     ).file()
        os.environ['PALLADIUM_CONFIG'] = '{}{}'.format(self.file_path, self.file_name)

    def generate_json(self, data_file_path: str, action: str, sample: int = None):
        """

        Generate json file for testing purposes

        :param data_file_path: String containing the complete path of the test data file path
        :param action: String containing the actual model name on the AnalyticsPlatform
        :param sample: Integer indicating the amount of cases to sample randomly
        """
        _df = DataImporter(file_path=data_file_path.replace('\\', '/'), create_dir=True).file()
        _data = {'actions': list()}
        if sample is None:
            _case = range(0, _df.shape[0], 1)
        else:
            _case = random.sample(population=[i for i in range(0, _df.shape[0], 1)], k=sample)
        for case in _case:
            _actions = [{'action': action, 'features': [dict()]}]
            for feature in _df.keys():
                _actions[0]['features'][0].update({feature: str(_df[feature][case])})
            _data['actions'].append(_actions)
        DataExporter(obj=_data,
                     file_path=self.full_path,
                     overwrite=True,
                     create_dir=True
                     ).file()

    def generate_request(self, payload: List[Mapping[str, str]] = None, end_point: str = 'questions') -> json:
        """

        Generate request for palladium dev-server

        :param end_point: String containing the name of the configured end point
        :param payload: Dictionary containing the test data for prediction
        :return:
        """
        if payload is None:
            _payload: List[Mapping[str, str]] = DataImporter(file_path=self.full_path).file().to_json()
        else:
            _payload: List[Mapping[str, str]] = payload
        try:
            _response = requests.post('http://localhost:5000/{}'.format(end_point), json=_payload)
        except ConnectionError as e:
            raise UtilsException('Error occured while reqesting prediction: {}'.format(e))
        return _response.json()

    def local_test(self):
        """

        Run local training and prediction tests

        :return:
        """
        raise UtilsException('Method not implemented yet :-(')


class UtilsException(Exception):
    """

    Class for setting up exceptions for class Utils

    """
    pass


class Utils:
    """

    Class for applying general utility methods

    """
    def __init__(self):
        """
        """
        pass

    @staticmethod
    def bayesian_blocks(df: pd.DataFrame, feature: str) -> dict:
        """
        Optimal univariate binning using Bayesian Blocks

        Returns
        -------
        dict: Binning edges (changepoints) and labels
        """
        _x: np.array = df[feature].sort_values(axis=0, ascending=True, inplace=False).values
        _edges: np.ndarray = np.concatenate([_x[:1], 0.5 * (_x[1:] + _x[:-1]), _x[-1:]])
        _block_length = _x[-1] - _edges
        _ones_arr: np.array = np.ones(_x.size)
        _best: list = []
        _last: list = []
        for k in range(0, _x.size, 1):
            # Compute the width and count of the final bin for all possible
            # locations of the K^th changepoint
            _width = _block_length[:k + 1] - _block_length[k + 1]
            _count_vec = np.cumsum(_ones_arr[:k + 1][::-1])[::-1]

            # evaluate fitness function for these possibilities
            _fit_vec: np.array = _count_vec * (np.log(_count_vec) - np.log(_width))
            _fit_vec -= 4  # 4 comes from the prior on the number of changepoints
            _fit_vec[1:] += _best[:k]

            # find the max of the fitness: this is the K^th changepoint
            _last.append(np.argmax(_fit_vec))
            _best.append(_fit_vec[np.argmax(_fit_vec)])

        # -----------------------------------------------------------------
        # Recover changepoints by iteratively peeling off the last block
        # -----------------------------------------------------------------
        _change_points = np.zeros(_x.size, dtype=int)
        _i: int = _x.size
        _idx: int = _x.size
        while True:
            _i -= 1
            _change_points[_i] = _idx
            if _idx == 0:
                break
            _idx = _last[_idx - 1]
        _bayesian_blocks: dict = dict(edges=_edges[_change_points[_idx:]], labels=[])
        _bayesian_blocks['labels'] = [np.argmin(np.abs(val - _bayesian_blocks.get('edges'))) for val in _x.tolist()]
        return _bayesian_blocks

    @staticmethod
    def cat_array(array_with_nan: np.array) -> np.array:
        """

        Convert categorical float typed Numpy array into integer typed array

        :param array_with_nan: Numpy array containing the categorical data with missing values (float typed)
        :return: Numpy array containing the categorical data as integer without missing values
        """
        return np.array(array_with_nan[~pd.isnull(array_with_nan)], dtype=np.int_)

    @staticmethod
    def check_dtypes(df: pd.DataFrame, date_edges: Tuple[str, str] = None) -> dict:
        """
        Check if data types of Pandas DataFrame match with the analytical measurement of data

        Parameters
        ----------
        df: pd.DataFrame
            Data set

        date_edges: Tuple[str, str]

        Returns
        -------
        dict: Data type conversion recommendation
        """
        _typing: dict = dict(meta={}, conversion={})
        _features: List[str] = list(df.keys())
        _dtypes: List[str] = [str(dt) for dt in df.dtypes.tolist()]
        _feature_types: Dict[str, List[str]] = Utils().get_feature_types(df=df,
                                                                         features=_features,
                                                                         dtypes=df.dtypes.tolist(),
                                                                         date_edges=date_edges
                                                                         )
        _table: Dict[str, List[str]] = {'feature_type': [], 'data_type': [], 'rec': []}
        for i in range(0, len(_features), 1):
            if any(df[_features[i]].isnull()):
                if len(df[_features[i]].unique()) == 1:
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
                    if any(df[_features[i]].isnull()):
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
                        if any(df[_features[i]].isnull()):
                            _typing['meta'][_features[i]].update({'rec': 'Handle missing data and convert to integer'})
                        else:
                            _typing['meta'][_features[i]].update({'rec': 'Convert to integer'})
            elif str(_dtypes[i]).find('int') >= 0:
                if _features[i] not in _feature_types.get('categorical'):
                    _typing['meta'][_features[i]].update({_features[i]: dict(feature_type='int')})
                    if _features[i] in _feature_types.get('ordinal'):
                        continue
                    elif _features[i] in _feature_types.get('date'):
                        _typing['meta'].update({_features[i]: dict(data_type='date',
                                                                   rec='Convert to datetime'
                                                                   )})
                        _typing['conversion'].update({_features[i]: 'date'})
                    elif _features[i] in _feature_types.get('text'):
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
                    if any(df[_features[i]].isnull()):
                        _typing['meta'][_features[i]].update({'rec': 'Handle missing data and convert to float'})
                    else:
                        _typing['meta'][_features[i]].update({'rec': 'Convert to float'})
                elif _features[i] in _feature_types.get('categorical') or _features[i] in _feature_types.get('ordinal'):
                    _typing['meta'].update({_features[i]: dict(data_type='categorical',
                                                               feature_type='text'
                                                               )
                                            })
                    _typing['conversion'].update({_features[i]: 'int'})
                    if any(df[_features[i]].isnull()):
                        _typing['meta'][_features[i]].update({'rec': 'Handle missing data and convert to integer by label encoding'})
                    else:
                        _typing['meta'][_features[i]].update({'rec': 'Convert to integer by label encoding'})
        return _typing

    @staticmethod
    def convert_jupyter(notebook_name: str, to: str = 'html'):
        """

        Convert Jupyter Notebook into several formats

        :param notebook_name: str: Name of the jupyter notebook
        :param to: str: Output format
                        -> html: HTML
                        -> pdf: PDF
                        -> latex: Latex
                        -> markdown: Markdown
                        -> rst: reStructuredText
                        -> script: Python / Julia / R script (depending on kernel settings of ipynb file)
        """
        subprocess.run('jupyter nbconvert "{}" --to {}'.format(notebook_name, to))

    @staticmethod
    def extract_tuple_el_in_list(list_of_tuples: List[tuple], tuple_pos: int) -> list:
        """

        Extract specific tuple elements from list of tuples

        :param list_of_tuples: List[tuple]: List of tuples
        :param tuple_pos: int: Position of element in tuples to extract
        :return: list: List of elements of tuple
        """
        if tuple_pos < 0:
            raise UtilsException('Position of element in tuple cannot be negative ({})'.format(tuple_pos))
        return next(islice(zip(*list_of_tuples), tuple_pos, None))

    @staticmethod
    def freedman_diaconis_bins(data_points: np.array) -> int:
        """

        Calculate the width of each bin by using Freedmann-Diaconis rule

        :param data_points: Numpy array containing the data point to plot
        :return: Integer indicating the amount of bins to compute
        """
        data: np.array = np.asarray(data_points, dtype=np.float_)
        iqr = stats.iqr(data, rng=(25, 75), scale='raw', nan_policy='omit')
        rng = data.max() - data.min()
        return int((rng / ((2 * iqr) / np.power(data.size, 1 / 3))) + 1)

    @staticmethod
    def freedman_diaconis_width(data_points: np.array) -> float:
        """

        Calculate the width of each bin by using Freedmann-Diaconis rule

        :param data_points: Numpy array containing the data point to plot
        :return: Float indicating the width of each bin
        """
        data: np.ndarray = np.asarray(data_points, dtype=np.float_)
        iqr = stats.iqr(data, rng=(25, 75), scale='raw', nan_policy='omit')
        return (2 * iqr) / np.power(data.size, 1 / 3)

    @staticmethod
    def generate_git_ignore(file_path: str, exclude_files: List[str] = None, exclude_default: bool = True):
        """

        Generate .gitignore file

        :param file_path:
        :param exclude_files: List[str]: Names of files or objects to be ignored
        :param exclude_default: bool: Exclude default files and objects
        """
        _gitignore: str = ''
        _default: str = '#########\n# misc: ##########\n.git\n.idea\n.DS_Store\n\n########### python: ###########\nvenv\n**/.cache\n**/__pycache__\n.pytest_cache\n\n###################### jupyter notebook: ######################\n.ipynb_checkpoints\n\n################## data sources: ##################\n*.db\n*.txt\n'
        if exclude_default:
            _gitignore = _default
        if exclude_files is not None:
            if len(exclude_files) > 0:
                for f in exclude_files:
                    _gitignore = _gitignore + f
        DataExporter(obj=_gitignore, file_path=file_path, create_dir=True, overwrite=False).file()

    @staticmethod
    def generate_network(df: pd.DataFrame, node_feature: str, edge_feature: str, kind: str = 'undirected', **kwargs) -> nx:
        """
        Generate network graph

        Parameters
        ----------
        df: pd.DataFrame
            Data set

        node_feature: str
            Name of the feature used to generate nodes from

        edge_feature: str
            Name of the feature used to generate edges from

        kind: str
            Network type
                -> directed: Bayes network
                -> undirected: Markov Network
                -> geometric: Geometric Network based on x, y scale

        Returns
        -------
        nx: Preconfigured Networkx network graph object
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
            raise UtilsException('Network graph type ({}) not supported'.format(kind))
        if node_feature not in df.keys():
            raise UtilsException('Node feature ({}) not found'.format(node_feature))
        if edge_feature not in df.keys():
            raise UtilsException('Edge feature ({}) not found'.format(edge_feature))
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
    def generate_time_series_arrays(data_set: np.ndarray, lag: int = 1, train: bool = True) -> dict:
        """

        Generate n-dimensional numpy arrays formated for LSTM and Convolutional neural networks especially

        :param data_set: np.ndarray: Data set as n-dimensional numpy array
        :param lag: int: Number of previous values to look back
        :param train: bool: Generate train data set or test data set
        :return: dict: Time series train and test data sets
        """
        if lag < 1:
            _lag: int = 1
        else:
            _lag: int = lag
        _x_train: list = []
        _y_train: list = []
        _x_test: list = []
        _y_test: list = []
        for x in range(_lag, len(data_set)):
            if train:
                _x_train.append(data_set[x - _lag:x, : -2])
                _y_train.append(data_set[x - 1, -2])
                _x_test.append(data_set[x - _lag:x, 1:-1])
                _y_test.append(data_set[x - 1, -1])
            else:
                _x_train.append(data_set[x - _lag:x, : -1])
                _y_train.append(data_set[x - 1, -1])
                _x_test.append(data_set[x - _lag:x, 1:])
        return dict(x_train=np.array(_x_train),
                    y_train=np.array(_y_train),
                    x_test=np.array(_x_test),
                    y_test=np.array(_y_test)
                    )

    @staticmethod
    def geometric_progression(n: int = 10, ratio: int = 2) -> List[int]:
        """
        Generate list of geometric progression values

        Parameters
        ----------
        n: int
            Amount of values of the geometric progression

        ratio: float
            Base ratio value of the geometric progression

        Returns
        -------
        List[int]: Geometric progression
        """
        return list(accumulate([ratio] * n, mul))

    @staticmethod
    def get_duplicates(df: pd.DataFrame, cases: bool = True, features: bool = True) -> Dict[str, list]:
        """
        Get duplicate cases and / or features

        Parameters
        ----------
        df: pd.DataFrame
            Data set

        cases: bool
            Check whether cases are duplicated or not

        features: bool
            Check whether features are duplicated or not

        Returns
        -------
        Dict[str, list]: List of duplicated cases and / or features
        """
        _duplicates: dict = dict(cases=[], features=[])
        if cases:
            _duplicates['cases'] = df.loc[df.duplicated(), :].index.values.tolist()
        if features:
            _duplicates['features'] = df.loc[:, df.transpose().duplicated()].keys().tolist()
        return _duplicates

    @staticmethod
    def get_feature_types(df: pd.DataFrame,
                          features: List[str],
                          dtypes: List[np.dtype],
                          continuous: List[str] = None,
                          categorical: List[str] = None,
                          ordinal: List[str] = None,
                          date: List[str] = None,
                          text: List[str] = None,
                          max_cats: int = 500,
                          date_edges: Tuple[str, str] = None
                          ) -> Dict[str, List[str]]:
        """

        Get feature types

        :param df:
        :param features:
        :param dtypes:
        :param continuous:
        :param categorical:
        :param ordinal:
        :param date:
        :param text: List[str]
        :param max_cats:
        :param date_edges:
        :return:
        """
        _num: list = []
        _cat: list = []
        _str: list = []
        _date: list = []
        _ordinal: list = []
        #TODO: Check date edges
        if date_edges is None:
            _date_edges = None
        else:
            try:
                assert pd.to_datetime(date_edges[0])
                assert pd.to_datetime(date_edges[1])
                _date_edges: tuple = tuple([pd.to_datetime(date_edges[0]), pd.to_datetime(date_edges[1])])
            except:
                _date_edges = None
        for i, feature in enumerate(features):
            if date is not None:
                if feature in date:
                    _date.append(feature)
                    continue
            if ordinal is not None:
                if feature in ordinal:
                    _ordinal.append(feature)
                    continue
            if continuous is not None:
                if feature in continuous:
                    _num.append(feature)
                    continue
            if categorical is not None:
                if feature in categorical:
                    _cat.append(feature)
                    continue
            if text is not None:
                if feature in text:
                    _str.append(feature)
                    continue
            if str(dtypes[i]).find('float') >= 0:
                _unique: np.array = df[feature].unique()
                if any(df[feature].isnull()):
                    if any(_unique[~pd.isnull(_unique)] % 1) != 0:
                        _num.append(feature)
                    else:
                        if len(str(df[feature].min()).split('.')[0]) >= 4:
                            try:
                                assert pd.to_datetime(df[feature])
                                if _date_edges is None:
                                    _date.append(feature)
                                else:
                                    if (_date_edges[0] < pd.to_datetime(_unique.min())) or (_date_edges[1] > pd.to_datetime(_unique.max())):
                                        _str.append(feature)
                                    else:
                                        _date.append(feature)
                            except (TypeError, ValueError):
                                _str.append(feature)
                        else:
                            _cat.append(feature)
                else:
                    if any(_unique % 1) != 0:
                        _num.append(feature)
                    else:
                        if len(str(df[feature].min()).split('.')[0]) >= 4:
                            try:
                                assert pd.to_datetime(df[feature])
                                if _date_edges is None:
                                    _date.append(feature)
                                else:
                                    if (_date_edges[0] < pd.to_datetime(_unique.min())) or (_date_edges[1] > pd.to_datetime(_unique.max())):
                                        _str.append(feature)
                                    else:
                                        _date.append(feature)
                            except (TypeError, ValueError):
                                _str.append(feature)
                        else:
                            _cat.append(feature)
            elif str(dtypes[i]).find('int') >= 0:
                if len(str(df[feature].min())) >= 4:
                    try:
                        assert pd.to_datetime(df[feature])
                        _date.append(feature)
                    except (TypeError, ValueError):
                        _cat.append(feature)
                else:
                    if df.shape[0] == len(df[feature].unique()):
                        _str.append(feature)
                    else:
                        _cat.append(feature)
            elif str(dtypes[i]).find('object') >= 0:
                try:
                    _ar: np.ndarray = df[feature].astype(float).values
                    if any(pd.isnull(_ar)):
                        _unique: np.array = df[feature].unique()
                        if any(_unique[~pd.isnull(_unique)] % 1) != 0:
                            _num.append(feature)
                        else:
                            if len(str(df[feature].min()).split('.')[0]) >= 4:
                                try:
                                    assert pd.to_datetime(df[feature])
                                    _date.append(feature)
                                except (TypeError, ValueError):
                                    _str.append(feature)
                            else:
                                _cat.append(feature)
                except (TypeError, ValueError):
                    # TODO: check regex for values like 120,00
                    if len(df[feature].unique()) <= max_cats:
                        if df.shape[0] == len(df[feature].unique()):
                            _str.append(feature)
                        else:
                            try:
                                assert int(df[feature].min())
                                _cat.append(feature)
                            except (AssertionError, TypeError, ValueError):
                                try:
                                    assert pd.to_datetime(df[feature])
                                    _date.append(feature)
                                except (TypeError, ValueError):
                                    _cat.append(feature)
                    else:
                        _str.append(feature)
            elif str(dtypes[i]).find('date') >= 0:
                _date.append(feature)
            elif str(dtypes[i]).find('bool') >= 0:
                _cat.append(feature)
        return dict(continuous=_num, categorical=_cat, ordinal=_ordinal, date=_date, text=_str)

    @staticmethod
    def get_float_threshold() -> Tuple[float, float]:
        """
        Get minimum and maximum threshold of float value
            -> after minimum threshold float value => 0.0
            -> after maximum threshold float value => inf / -inf

        :return:
        Tuple[float, float]: Minimum and maximum float value
        """
        _i: int = 1
        _threshold: Tuple[float, float] = (0.0, 0.0)
        while True:
            if 1e+1 + _i == 0:
                break
            else:
                _i += 1
        return _threshold

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
        return dict

    @staticmethod
    def get_ml_type(values: np.array) -> str:
        """
        Get supervised machine learning problem type from value of target feature

        Parameters
        ----------
        values: np.array
            Value of target feature

        Returns
        -------
        str: Supervised machine learning type
                -> reg: Regression
                -> clf_multi: Classification with multi class output
                -> clf_binary: Classification with binary class output
        """
        _unique: np.array = pd.unique(values=values)
        if any(_unique[~pd.isnull(_unique)] % 1) != 0:
            return 'reg'
        else:
            if len(_unique[~pd.isnull(_unique)]) == 2:
                return 'clf_binary'
            else:
                return 'clf_multi'

    @staticmethod
    def get_list_of_files(file_path: str) -> List[str]:
        """

        Get list of file in given directory or zip file

        :param file_path: str: Complete file path
        :return: List[str]: Name of detected files
        """
        if os.path.exists(file_path):
            if file_path.split('.')[-1] is 'zip':
                return zipfile.ZipFile(file_path).namelist()
            else:
                return [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
        else:
            raise UtilsException('File path not found ({})'.format(file_path))

    @staticmethod
    def get_list_of_objects(file_path: str) -> List[str]:
        """

        Get list of objects in given directory and subdirectories

        :param file_path: str: Complete file path
        :return: List[str]: List of object names detected in directory
        """
        return [obj for obj in os.listdir(file_path)]

    @staticmethod
    def get_invariant_features(df: pd.DataFrame) -> List[str]:
        """
        Get invariant features of data set

        Parameters
        ----------
        df: pd.DataFrame
            Data set

        Returns
        -------
        List[str]: Names of invariant features
        """
        _invariant_features: List[str] = []
        for ft in df.keys():
            _unique_values: np.ndarray = df[ft].unique()
            if len(_unique_values) <= 2:
                if len(_unique_values) == 1:
                    _invariant_features.append(ft)
                else:
                    if any(df[ft].isnull()):
                        _invariant_features.append(ft)
        return _invariant_features

    @staticmethod
    def get_pairs(features: List[str], max_features_each_pair: int = 2) -> List[tuple]:
        """

        Get pairs of feature list

        :param List[str] features: Features to pair
        :param int max_features_each_pair: Maximum number of features for each pair
        :return: List[tuple]: List of features pairs
        """
        return [pair for pair in itertools.combinations(iterable=list(set(features)), r=max_features_each_pair)]

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

        :param data: pd.DataFrame: Data set to calculate percentiles from
        :param group_by: str: Grouping features
        :param aggregate_by: List[str]: Features to aggregate
        :param aggregation: str: Aggregation method
        :param percentiles: int: Number of percentiles to generate
        :param duplicates: str: Handle duplicates
                                -> raise: Raise exception
                                -> drop: Drop
        :param include_group: bool: Include grouping feature inbto aggregation process
        :return: pd.DataFrame: Aggregated data set
        """
        _aggre: List[str] = []
        _df: pd.DataFrame = pd.DataFrame()
        if group_by in data.keys():
            _df[group_by] = data[group_by]
            if len(aggregate_by) == 0:
                raise UtilsException('No features for aggregation found')
            for agg in aggregate_by:
                if agg in data.keys():
                    _aggre.append(agg)
                    _df[agg] = data[agg]
        else:
            raise UtilsException('No feature for grouping found')
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

        :param pred: list: List of predictions
        :param obs: list: List of observations
        :param aggregation: str: Aggregation method
                                -> min: Minimum
                                -> max: Maximum
                                -> median: Median
                                -> mean: Mean
        :param percentiles: int: Number of percentiles to calculate
        :return: pd.DataFrame: Percentile evaluation of prediction and observation
        """
        _df = pd.DataFrame({'obs': obs, 'preds': pred})
        _df['pred_perc'] = pd.qcut(x=_df['obs'], q=percentiles, labels=np.arange(1, percentiles + 1), retbins=False, precision=4, duplicates='drop')
        _df_perc = _df.groupby(by=['pred_perc']).agg({'preds': aggregation, 'obs': aggregation})
        _df_perc['abs_diff'] = _df_perc['preds'] - _df_perc['obs']
        _df_perc['rel_diff'] = (_df_perc['preds'] - _df_perc['obs']) / _df_perc['obs']
        return _df_perc

    @staticmethod
    def get_random_perm(shape: int) -> np.random:
        """

        Get random permutation

        :param shape: int: Maximum threshold of range to permutate randomly
        :return: np.random: Randomly permutated range
        """
        np.random.seed(seed=1234)
        return np.random.permutation(x=shape)

    @staticmethod
    def index_to_label(idx: List[int], labels: List[str]) -> List[str]:
        """

        Get list of labels based on a list of indices

        :param idx: List of integers containing the indices
        :param labels: List of integers containing the labels
        :return: List of string containing the subset of labels
        """
        return [labels[i] for i in idx]

    @staticmethod
    def label_to_index(all_labels: List[str], labels: List[str]) -> List[int]:
        """

        Get list of indices based on a list if labels

        :param labels: List of integers containing the indices
        :param all_labels: List of integers containing the labels
        :return: List of integers indicating the subset of indices
        """
        return [all_labels.index(i) for i in labels]

    @staticmethod
    def replace_dict_keys(d: dict, new_keys: List[str]) -> dict:
        """

        Replace keys of a dictionary by values of given list

        :param d:
        :param new_keys:
        :return:
        """
        _d = dict()
        for k in d.keys():
            _d[k] = new_keys[k]
        return _d

    @staticmethod
    def search_for_file(key_word: str, starting_dir: str) -> List[str]:
        """

        Search for files with similar name patterns as key word input

        :param key_word: str: Searched key word
        :param starting_dir: str: Directory to start searching in
        :return: List[str]: Names of the files found under given key word
        """
        return glob.glob(pathname=key_word)

    @staticmethod
    def subset_array(arr: np.array, idx: List[int]) -> np.array:
        """

        Subset Numpy array

        :param arr: Numpy array containing the data set
        :param idx: List of integers containing the indices to remove
        :return: Numpy array containing the data subset
        """
        return np.array(list(itertools.compress(data=arr, selectors=[i not in idx for i in range(len(idx))])))

    @staticmethod
    def subset_dict(d: dict, threshold: float) -> dict:
        """

        Subset dictionary by given threshold value

        :param d:
        :param threshold:
        :return:
        """
        _d = {}
        for k in d.keys():
            if d.get(k) >= threshold:
                _d[k] = d.get(k)
        return _d

    @staticmethod
    def rename_dict_keys(d: dict, old_keys: List[str], new_keys: List[str]) -> dict:
        """

        Rename keys of a dictionary

        :param d: Dictionary containing data
        :param old_keys: List of strings containing the old (current) key names
        :param new_keys: List of strings containing the new key names
        :return: Dictionary containing the renamed dictionary
        """
        if len(old_keys) != len(new_keys):
            raise UtilsException('Length of the two lists are unequal (old={}, new={}'.format(len(old_keys),
                                                                                              len(new_keys))
                                 )
        _d = d
        for i, k in enumerate(old_keys):
            _d = json.loads(json.dumps(_d).replace(k, new_keys[i]))
        return _d
