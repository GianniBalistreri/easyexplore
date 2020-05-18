import os
import pandas as pd
import unittest

from easyexplore.utils import EasyExploreUtils

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


if __name__ == '__main__':
    unittest.main()
