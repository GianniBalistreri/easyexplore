import os
import pandas as pd
import unittest

from easyexplore.data_import_export import DataExporter, DataImporter, DBUtils


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
        #_zip: dict = DataImporter(file_path='test.zip',
        #                          as_data_frame=False,
        #                          create_dir=False
        #                          ).zip(files=['test_data_csv'])
        #self.assertTrue(expr=isinstance(_zip['test.zip'].get('A'), list))


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


class DBUtilsTest(unittest.TestCase):
    """
    Unit test for class DBUtils
    """
    def test_create_connection(self):
        _con = DBUtils(file_path='test_data.db').create_connection()
        if _con is None:
            _is_connected: bool = False
        else:
            _is_connected: bool = True
            _con.close()
        self.assertTrue(expr=_is_connected)

    def test_get_table(self):
        _con = DBUtils(file_path='test_data.db').create_connection()
        _table = DBUtils(con=_con, table_name='test_data').get_table()
        _con.close()
        self.assertTrue(expr=isinstance(_table, pd.DataFrame))

    def test_create_table(self):
        _con = DBUtils(file_path='test_data.db').create_connection()
        DBUtils(df=pd.DataFrame(), con=_con, table_name='new_table').create_table()
        _new_table = DBUtils(con=_con, table_name='new_table').get_table()
        _con.close()
        self.assertTrue(expr=isinstance(_new_table, pd.DataFrame))

    def test_update_table(self):
        _con = DBUtils(file_path='test_data.db').create_connection()
        _table = DBUtils(con=_con, table_name='test_data').get_table()
        DBUtils(df=_table, con=_con, table_name='new_table').update_table()
        _new_table = DBUtils(con=_con, table_name='new_table').get_table()
        _con.close()
        self.assertListEqual(list1=_table['A'].values.tolist(),
                             list2=_new_table['A'].values.tolist()
                             )

    def test_drop_table(self):
        _con = DBUtils(file_path='test_data.db').create_connection()
        DBUtils(con=_con, table_name='new_table').drop_table()
        try:
            _new_table = DBUtils(con=_con, table_name='new_table').get_table()
            _dropped: bool = False
        except Exception:
            _dropped: bool = True
        _con.close()
        self.assertTrue(expr=_dropped)


if __name__ == '__main__':
    unittest.main()
