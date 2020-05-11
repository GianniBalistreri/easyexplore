import os
import pandas as pd
import unittest

from easyexplore.data_import_export import DataExporter, DataImporter


class DataImporterTest(unittest.TestCase):
    """
    Unit test for class DataImporter
    """
    def test_file(self):
        _df: pd.DataFrame = DataImporter(file_path='test_data.csv',
                                         as_data_frame=True,
                                         sep=','
                                         ).file()
        self.assertDictEqual(d1=dict(rows=10, cols=11),
                             d2=dict(rows=_df.shape[0], cols=_df.shape[1])
                             )

    def test_zip(self):
        pass


class DataExporterTest(unittest.TestCase):
    """
    Unit test for class DataExporter
    """
    def test_file(self):
        _obj: str = 'easyexplore'
        DataExporter(obj=_obj,
                     file_path='test_data_exporter.pkl',
                     create_dir=False,
                     overwrite=True
                     ).file()
        self.assertTrue(expr=os.path.isfile('test_data_exporter.pkl'))


if __name__ == '__main__':
    unittest.main()
