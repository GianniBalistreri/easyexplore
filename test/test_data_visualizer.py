import os
import pandas as pd
import unittest

from easyexplore.data_visualizer import DataVisualizer

DATA_SET: pd.DataFrame = pd.read_csv(filepath_or_buffer='test_data.csv', sep=',')


class DataVisualizerTest(unittest.TestCase):
    """
    Unit test for class DataVisualizer
    """
    def test_load(self):
        pass

    def test_run(self):
        DataVisualizer(title='DataVisualizer Test',
                       df=DATA_SET,
                       features=['F'],
                       plot_type='pie',
                       render=False,
                       file_path='test_data_visualizer.html'
                       ).run()
        self.assertTrue(expr=os.path.isfile('test_data_visualizer.html'))


if __name__ == '__main__':
    unittest.main()
