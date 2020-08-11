import pandas as pd
import unittest

from easyexplore.data_explorer import DataExplorer

DATA_SET: pd.DataFrame = pd.read_csv(filepath_or_buffer='test_data.csv', sep=',')
TEXT_DATA: pd.DataFrame = pd.read_csv(filepath_or_buffer='amazon_musical_instruments_reviews.csv')


class DataExplorerTest(unittest.TestCase):
    """
    Unit test for class DataExplorer
    """
    def test_break_down(self):
        self.assertEqual(first=0.49000000000000005,
                         second=DataExplorer(df=DATA_SET, plot=False).break_down()['continuous']['J']['de']['C'].get('mean')
                         )

    def test_cor(self):
        pass

    def test_data_distribution(self):
        _data_distribution: dict = DataExplorer(df=DATA_SET, plot=False).data_distribution()
        _sample_results: dict = dict(F=_data_distribution['F'].get('Hamburg'), C=_data_distribution['C'].get('mean'))
        self.assertDictEqual(d1=dict(F=4, C=0.49000000000000005),
                             d2=_sample_results
                             )

    def test_data_health_check(self):
        self.assertDictEqual(d1={'cases': [], 'features': ['J', 'K']},
                             d2=DataExplorer(df=DATA_SET, plot=False).data_health_check()
                             )

    def test_data_typing(self):
        self.assertDictEqual(d1={'B': 'int', 'D': 'datetime', 'F': 'int', 'I': 'int', 'J': 'int', 'K': 'int'},
                             d2=DataExplorer(df=DATA_SET, plot=False).data_typing()
                             )

    def test_get_feature_types(self):
        self.assertDictEqual(d1={'continuous': ['C', 'G', 'H'],
                                 'categorical': ['A', 'B', 'F', 'I', 'J', 'K'],
                                 'ordinal': [],
                                 'date': ['D'],
                                 'text': ['E']
                                 },
                             d2=DataExplorer(df=DATA_SET, plot=False).get_feature_types()
                             )

    def test_geo_stats(self):
        pass

    def test_outlier_detector(self):
        pass

    def test_text_analyzer(self):
        self.assertTrue(expr=DataExplorer(df=TEXT_DATA).text_analyzer(lang='en').shape[1] > 0)


if __name__ == '__main__':
    unittest.main()
