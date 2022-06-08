import os
import pandas as pd
import unittest

from easyexplore.utils import EasyExploreUtils, StatsUtils
from typing import List

OUTPUT_PATH: str = os.getcwd()
DATA_SET: pd.DataFrame = pd.read_csv(filepath_or_buffer='test_data.csv', sep=',')


class EasyExploreUtilsTest(unittest.TestCase):
    """
    Unit test for class EasyExploreUtils
    """
    def test_check_dtypes(self):
        self.assertDictEqual(d1={'B': 'int', 'D': 'date', 'F': 'int', 'I': 'int', 'J': 'int', 'K': 'int'},
                             d2=EasyExploreUtils().check_dtypes(df=DATA_SET, date_edges=None).get('conversion')
                             )

    def test_convert_jupyter(self):
        EasyExploreUtils().convert_jupyter(notebook_name=os.path.join(OUTPUT_PATH, 'test_notebook.ipynb'), to='html')
        self.assertTrue(expr=os.path.isfile(os.path.join(OUTPUT_PATH, 'test_notebook.ipynb')))

    def test_friedmann_diaconis_bins(self):
        pass

    def test_friedmann_diaconis_width(self):
        pass

    #def test_generate_git_ignore(self):
    #    EasyExploreUtils().generate_git_ignore(file_path='{}.gitignore'.format(OUTPUT_PATH), exclude_files=None, exclude_default=True)
    #    self.assertTrue(expr=os.path.isfile('{}.gitignore'.format(OUTPUT_PATH)))

    def test_generate_network(self):
        pass

    def test_get_duplicates(self):
        self.assertDictEqual(d1=dict(cases=[], features=['K']),
                             d2=EasyExploreUtils().get_duplicates(df=DATA_SET, cases=True, features=True)
                             )

    def test_get_feature_types(self):
        self.assertDictEqual(d1={'continuous': ['C', 'G', 'H'],
                                 'categorical': ['A', 'B', 'F', 'I', 'J', 'K'],
                                 'ordinal': [],
                                 'date': ['D'],
                                 'text': ['E']
                                 },
                             d2=EasyExploreUtils().get_feature_types(df=DATA_SET,
                                                                     features=list(DATA_SET.keys()),
                                                                     dtypes=DATA_SET.dtypes.tolist()
                                                                     )
                             )

    def test_get_geojson(self):
        pass

    def test_get_list_of_files(self):
        pass

    def test_get_list_of_objects(self):
        pass

    def test_get_pairs(self):
        self.assertListEqual(list1=[tuple(['A', 'B']), tuple(['A', 'C']), tuple(['B', 'C'])],
                             list2=EasyExploreUtils().get_pairs(features=['A', 'B', 'C'], max_features_each_pair=2)
                             )


class StatsUtilsTest(unittest.TestCase):
    """
    Unit test for class StatsUtils
    """
    def test_chi_squared_independence_test(self):
        _stats_utils: StatsUtils = StatsUtils(data=DATA_SET)
        _results: dict = _stats_utils.chi_squared_independence_test(x='B', y='F')
        self.assertListEqual(list1=['features', 'cases', 'test_statistic', 'p_value', 'reject'],
                             list2=list(_results.keys())
                             )

    def test_correlation_pearson(self):
        _features: List[str] = ['C', 'H']
        _stats_utils: StatsUtils = StatsUtils(data=DATA_SET)
        _results: pd.DataFrame = _stats_utils.correlation(features=_features, meth='pearson')
        self.assertTrue(expr=_results['C'].values.tolist()[0] == _results['H'].values.tolist()[1] and
                             _results['C'].values.tolist()[1] == _results['H'].values.tolist()[0]
                        )

    def test_correlation_spearman(self):
        _features: List[str] = ['B', 'K']
        _stats_utils: StatsUtils = StatsUtils(data=DATA_SET)
        _results: pd.DataFrame = _stats_utils.correlation(features=_features, meth='spearman')
        self.assertTrue(expr=_results['B'].values.tolist()[0] == _results['K'].values.tolist()[1] and
                             _results['B'].values.tolist()[1] == _results['K'].values.tolist()[0]
                        )

    def test_correlation_kendall(self):
        _features: List[str] = ['B', 'K']
        _stats_utils: StatsUtils = StatsUtils(data=DATA_SET)
        _results: pd.DataFrame = _stats_utils.correlation(features=_features, meth='kendall')
        self.assertTrue(expr=_results['B'].values.tolist()[0] == _results['K'].values.tolist()[1] and
                             _results['B'].values.tolist()[1] == _results['K'].values.tolist()[0]
                        )

    def test_correlation_partial(self):
        _features: List[str] = ['C', 'G', 'H']
        _stats_utils: StatsUtils = StatsUtils(data=DATA_SET)
        _results: pd.DataFrame = _stats_utils.correlation(features=_features, meth='partial')
        self.assertTrue(expr=_results['C'].values.tolist()[0] == _results['G'].values.tolist()[1] and
                             _results['C'].values.tolist()[1] == _results['G'].values.tolist()[0]
                        )

    def test_correlation_test(self):
        _stats_utils: StatsUtils = StatsUtils(data=DATA_SET)
        _results: dict = _stats_utils.correlation_test(x='C', y='H', meth='pearson')
        self.assertListEqual(list1=['features', 'cases', 'test_statistic', 'p_value', 'reject'],
                             list2=list(_results.keys())
                             )

    def test_kurtosis(self):
        _features: List[str] = ['C', 'G', 'H']
        _stats_utils: StatsUtils = StatsUtils(data=DATA_SET)
        _results: dict = _stats_utils.kurtosis(features=_features,
                                               axis='col',
                                               skip_missing_values=True,
                                               use_numeric_features_only=True
                                               )
        _feature_names: List[str] = []
        _is_float: bool = False
        for feature, kurtosis in _results.items():
            _feature_names.append(feature)
            if isinstance(kurtosis, float):
                _is_float = True
            else:
                _is_float = False
        self.assertTrue(expr=_features == _feature_names and _is_float)

    def test_non_parametric_test_kruskal_wallis(self):
        _stats_utils: StatsUtils = StatsUtils(data=DATA_SET)
        _results: dict = _stats_utils.non_parametric_test(x='C', y='H', meth='kruskal-wallis')
        self.assertListEqual(list1=['features', 'cases', 'test_statistic', 'p_value', 'reject'],
                             list2=list(_results.keys())
                             )

    def test_non_parametric_test_mann_whitney_u(self):
        _stats_utils: StatsUtils = StatsUtils(data=DATA_SET)
        _results: dict = _stats_utils.non_parametric_test(x='C', y='H', meth='mann-whitney')
        self.assertListEqual(list1=['features', 'cases', 'test_statistic', 'p_value', 'reject'],
                             list2=list(_results.keys())
                             )

    def test_non_parametric_test_wilcoxon(self):
        _stats_utils: StatsUtils = StatsUtils(data=DATA_SET)
        _results: dict = _stats_utils.non_parametric_test(x='C', y='H', meth='wilcoxon')
        self.assertListEqual(list1=['features', 'cases', 'test_statistic', 'p_value', 'reject'],
                             list2=list(_results.keys())
                             )

    def test_non_parametric_test_friedman(self):
        _stats_utils: StatsUtils = StatsUtils(data=DATA_SET)
        _results: dict = _stats_utils.non_parametric_test(x='C', y='H', meth='friedman')
        self.assertListEqual(list1=['features', 'cases', 'test_statistic', 'p_value', 'reject'],
                             list2=list(_results.keys())
                             )

    def test_non_parametric_test_ks(self):
        _stats_utils: StatsUtils = StatsUtils(data=DATA_SET)
        _results: dict = _stats_utils.non_parametric_test(x='C', y='H', meth='ks')
        self.assertListEqual(list1=['features', 'cases', 'test_statistic', 'p_value', 'reject'],
                             list2=list(_results.keys())
                             )

    def test_normality_test(self):
        _features: List[str] = ['C', 'H']
        _stats_utils: StatsUtils = StatsUtils(data=DATA_SET)
        _results: dict = _stats_utils.normality_test(features=_features, alpha=0.5, meth='shapiro-wilk')
        _feature_names: List[str] = list(_results.keys())
        _keys: List[str] = list(_results[_feature_names[0]].keys())
        self.assertTrue(expr=_features == _feature_names and _keys == ['stat', 'p', 'normality'])

    def test_parametric_test_t_test(self):
        _stats_utils: StatsUtils = StatsUtils(data=DATA_SET)
        _results: dict = _stats_utils.parametric_test(x='C', y='H', meth='t-test', welch_t_test=True)
        self.assertListEqual(list1=['features', 'cases', 'test_statistic', 'p_value', 'reject'],
                             list2=list(_results.keys())
                             )

    def test_parametric_test_t_test_paired(self):
        _stats_utils: StatsUtils = StatsUtils(data=DATA_SET)
        _results: dict = _stats_utils.parametric_test(x='C', y='H', meth='t-test-paired', welch_t_test=True)
        self.assertListEqual(list1=['features', 'cases', 'test_statistic', 'p_value', 'reject'],
                             list2=list(_results.keys())
                             )

    def test_parametric_test_z_test(self):
        _stats_utils: StatsUtils = StatsUtils(data=DATA_SET)
        _results: dict = _stats_utils.parametric_test(x='C', y='H', meth='z-test', welch_t_test=True)
        self.assertListEqual(list1=['features', 'cases', 'test_statistic', 'p_value', 'reject'],
                             list2=list(_results.keys())
                             )

    def test_parametric_test_anova(self):
        _stats_utils: StatsUtils = StatsUtils(data=DATA_SET)
        _results: dict = _stats_utils.parametric_test(x='C', y='H', meth='anova', welch_t_test=True)
        self.assertListEqual(list1=['features', 'cases', 'test_statistic', 'p_value', 'reject'],
                             list2=list(_results.keys())
                             )

    def test_skewness(self):
        _features: List[str] = ['C', 'G', 'H']
        _stats_utils: StatsUtils = StatsUtils(data=DATA_SET)
        _results: dict = _stats_utils.skewness(features=_features,
                                               axis='col',
                                               skip_missing_values=True,
                                               use_numeric_features_only=True
                                               )
        _feature_names: List[str] = []
        _is_float: bool = False
        for feature, skewness in _results.items():
            _feature_names.append(feature)
            if isinstance(skewness, float):
                _is_float = True
            else:
                _is_float = False
        self.assertTrue(expr=_features == _feature_names and _is_float)


if __name__ == '__main__':
    unittest.main()
