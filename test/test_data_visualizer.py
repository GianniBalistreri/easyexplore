import os
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import unittest

from easyexplore.data_visualizer import DataVisualizer, load_plotly_from_json, load_static_plot
from easyexplore.unsupervised_machine_learning import Clustering, UnsupervisedML
from typing import List

DATA_SET: pd.DataFrame = pd.read_csv(filepath_or_buffer='test_data.csv', sep=',')


class DataVisualizerUtilsTest(unittest.TestCase):
    """
    Unit test for utility functions for class DataVisualizer
    """
    def test_load_plotly_from_json(self):
        _file_path_json: str = 'test_contour_hist.json'
        DataVisualizer(title='Load from json Test',
                       df=DATA_SET,
                       features=['C', 'H'],
                       group_by=None,
                       melt=False,
                       plot_type='contour_hist',
                       render=False,
                       file_path=_file_path_json,
                       use_auto_extensions=False
                       ).run()
        load_plotly_from_json(file_path=_file_path_json)

    def test_load_static_plot(self):
        _file_path: str = 'test_contour_hist.png'
        DataVisualizer(title='Load Static Plot Test',
                       df=DATA_SET,
                       features=['C', 'H'],
                       group_by=None,
                       melt=False,
                       plot_type='contour_hist',
                       render=False,
                       file_path=_file_path,
                       use_auto_extensions=False,
                       interactive=False
                       ).run()
        _static_plot: go.Figure = load_static_plot(file_path=_file_path)
        self.assertTrue(expr=isinstance(_static_plot, go.Figure))


class DataVisualizerTest(unittest.TestCase):
    """
    Unit test for class DataVisualizer
    """
    def test_get_plotly_figure(self):
        _features: List[str] = ['F']
        _fig = DataVisualizer(title='Get Plotly Figure Test',
                              df=DATA_SET,
                              features=_features,
                              plot_type='bar'
                              ).get_plotly_figure()
        self.assertTrue(expr=isinstance(_fig, list) and isinstance(_fig[0], go.Bar))

    def test_run_2d_histogram_contour_chart(self):
        _features: List[str] = ['C', 'H']
        _group_features: List[str] = ['A']
        _group_values: List[int] = DATA_SET[_group_features[0]].unique().tolist()
        # without group by and without melt and without auto extensions
        DataVisualizer(title='2D Histogram Contour Test',
                       df=DATA_SET,
                       features=_features,
                       group_by=None,
                       melt=False,
                       plot_type='contour_hist',
                       render=False,
                       file_path='test_contour_hist.html',
                       use_auto_extensions=False
                       ).run()
        _output_file_path_1: str = 'test_contour_hist.html'
        # without group by and without melt and with auto extensions
        DataVisualizer(title='2D Histogram Contour Test',
                       df=DATA_SET,
                       features=_features,
                       group_by=None,
                       melt=False,
                       plot_type='contour_hist',
                       render=False,
                       file_path='test_contour_hist.html',
                       use_auto_extensions=True
                       ).run()
        _output_file_path_2: str = 'test_contour_hist_C_H.html'
        # with group by and without melt
        DataVisualizer(title='2D Histogram Contour Test',
                       df=DATA_SET,
                       features=_features,
                       group_by=_group_features,
                       melt=False,
                       plot_type='contour_hist',
                       render=False,
                       file_path='test_contour_hist.html',
                       use_auto_extensions=False
                       ).run()
        _output_file_path_4: str = 'test_contour_hist_C_H_A_0.html'
        _output_file_path_5: str = 'test_contour_hist_C_H_A_1.html'
        _output_file_path_6: str = 'test_contour_hist_C_H_A_2.html'
        _output_file_path_7: str = 'test_contour_hist_C_H_A_4.html'
        _output_file_path_8: str = 'test_contour_hist_C_H_A_5.html'
        _output_file_path_9: str = 'test_contour_hist_C_H_A_8.html'
        _output_file_path_10: str = 'test_contour_hist_C_H_A_9.html'
        self.assertTrue(expr=os.path.isfile(_output_file_path_1) and
                             os.path.isfile(_output_file_path_2) and
                             os.path.isfile(_output_file_path_4) and
                             os.path.isfile(_output_file_path_5) and
                             os.path.isfile(_output_file_path_6) and
                             os.path.isfile(_output_file_path_7) and
                             os.path.isfile(_output_file_path_8) and
                             os.path.isfile(_output_file_path_9) and
                             os.path.isfile(_output_file_path_10)
                        )

    def test_run_bar_chart(self):
        _features: List[str] = ['F']
        _group_features: List[str] = ['A']
        _group_values: List[int] = DATA_SET[_group_features[0]].unique().tolist()
        # without group by and without melt and without auto extensions
        DataVisualizer(title='Bar Test',
                       df=DATA_SET,
                       features=_features,
                       group_by=None,
                       melt=False,
                       plot_type='bar',
                       render=False,
                       file_path='test_bar.html',
                       use_auto_extensions=False
                       ).run()
        _output_file_path_1: str = 'test_bar.html'
        # without group by and without melt and with auto extensions
        DataVisualizer(title='Bar Test',
                       df=DATA_SET,
                       features=_features,
                       group_by=None,
                       melt=False,
                       plot_type='bar',
                       render=False,
                       file_path='test_bar.html',
                       use_auto_extensions=True
                       ).run()
        _output_file_path_2: str = 'test_bar_F.html'
        # without group by and with melt and with auto extensions
        DataVisualizer(title='Bar Test',
                       df=DATA_SET,
                       features=_features,
                       group_by=None,
                       melt=True,
                       plot_type='bar',
                       render=False,
                       file_path='test_bar.html',
                       use_auto_extensions=True
                       ).run()
        _output_file_path_3: str = 'test_bar_melt.html'
        # with group by and without melt
        DataVisualizer(title='Bar Test',
                       df=DATA_SET,
                       features=_features,
                       group_by=_group_features,
                       melt=False,
                       plot_type='bar',
                       render=False,
                       file_path='test_bar.html',
                       use_auto_extensions=False
                       ).run()
        _output_file_path_4: str = 'test_bar_F_A_0.html'
        _output_file_path_5: str = 'test_bar_F_A_1.html'
        _output_file_path_6: str = 'test_bar_F_A_2.html'
        _output_file_path_7: str = 'test_bar_F_A_4.html'
        _output_file_path_8: str = 'test_bar_F_A_5.html'
        _output_file_path_9: str = 'test_bar_F_A_8.html'
        _output_file_path_10: str = 'test_bar_F_A_9.html'
        # with group by and with melt
        DataVisualizer(title='Bar Test',
                       df=DATA_SET,
                       features=_features,
                       group_by=_group_features,
                       melt=True,
                       plot_type='bar',
                       render=False,
                       file_path='test_bar.html',
                       use_auto_extensions=False
                       ).run()
        _output_file_path_11: str = 'test_bar_F_A.html'
        self.assertTrue(expr=os.path.isfile(_output_file_path_1) and
                             os.path.isfile(_output_file_path_2) and
                             os.path.isfile(_output_file_path_3) and
                             os.path.isfile(_output_file_path_4) and
                             os.path.isfile(_output_file_path_5) and
                             os.path.isfile(_output_file_path_6) and
                             os.path.isfile(_output_file_path_7) and
                             os.path.isfile(_output_file_path_8) and
                             os.path.isfile(_output_file_path_9) and
                             os.path.isfile(_output_file_path_10) and
                             os.path.isfile(_output_file_path_11)
                        )

    def test_run_box_whisker_chart(self):
        _features: List[str] = ['C']
        _group_features: List[str] = ['A']
        _group_values: List[int] = DATA_SET[_group_features[0]].unique().tolist()
        # without group by and without melt and without auto extensions
        DataVisualizer(title='Box-Whisker Test',
                       df=DATA_SET,
                       features=_features,
                       group_by=None,
                       melt=False,
                       plot_type='box',
                       render=False,
                       file_path='test_box_whisker.html',
                       use_auto_extensions=False
                       ).run()
        _output_file_path_1: str = 'test_box_whisker.html'
        # without group by and without melt and with auto extensions
        DataVisualizer(title='Box-Whisker Test',
                       df=DATA_SET,
                       features=_features,
                       group_by=None,
                       melt=False,
                       plot_type='box',
                       render=False,
                       file_path='test_box_whisker.html',
                       use_auto_extensions=True
                       ).run()
        _output_file_path_2: str = 'test_box_whisker_C.html'
        # without group by and with melt and with auto extensions
        DataVisualizer(title='Box-Whisker Test',
                       df=DATA_SET,
                       features=_features,
                       group_by=None,
                       melt=True,
                       plot_type='box',
                       render=False,
                       file_path='test_box_whisker.html',
                       use_auto_extensions=True
                       ).run()
        _output_file_path_3: str = 'test_box_whisker_melt.html'
        # with group by and without melt
        DataVisualizer(title='Box-Whisker Test',
                       df=DATA_SET,
                       features=_features,
                       group_by=_group_features,
                       melt=False,
                       plot_type='box',
                       render=False,
                       file_path='test_box_whisker.html',
                       use_auto_extensions=False
                       ).run()
        _output_file_path_4: str = 'test_box_whisker_C_A_0.html'
        _output_file_path_5: str = 'test_box_whisker_C_A_1.html'
        _output_file_path_6: str = 'test_box_whisker_C_A_2.html'
        _output_file_path_7: str = 'test_box_whisker_C_A_4.html'
        _output_file_path_8: str = 'test_box_whisker_C_A_5.html'
        _output_file_path_9: str = 'test_box_whisker_C_A_8.html'
        _output_file_path_10: str = 'test_box_whisker_C_A_9.html'
        # with group by and with melt
        DataVisualizer(title='Box-Whisker Test',
                       df=DATA_SET,
                       features=_features,
                       group_by=_group_features,
                       melt=True,
                       plot_type='box',
                       render=False,
                       file_path='test_box_whisker.html',
                       use_auto_extensions=False
                       ).run()
        _output_file_path_11: str = 'test_box_whisker_C_A.html'
        self.assertTrue(expr=os.path.isfile(_output_file_path_1) and
                             os.path.isfile(_output_file_path_2) and
                             os.path.isfile(_output_file_path_3) and
                             os.path.isfile(_output_file_path_4) and
                             os.path.isfile(_output_file_path_5) and
                             os.path.isfile(_output_file_path_6) and
                             os.path.isfile(_output_file_path_7) and
                             os.path.isfile(_output_file_path_8) and
                             os.path.isfile(_output_file_path_9) and
                             os.path.isfile(_output_file_path_10) and
                             os.path.isfile(_output_file_path_11)
                        )

    def test_run_violin_chart(self):
        _features: List[str] = ['C']
        _group_features: List[str] = ['A']
        _group_values: List[int] = DATA_SET[_group_features[0]].unique().tolist()
        # without group by and without melt and without auto extensions
        DataVisualizer(title='Violin Test',
                       df=DATA_SET,
                       features=_features,
                       group_by=None,
                       melt=False,
                       plot_type='violin',
                       render=False,
                       file_path='test_violin.html',
                       use_auto_extensions=False
                       ).run()
        _output_file_path_1: str = 'test_violin.html'
        # without group by and without melt and with auto extensions
        DataVisualizer(title='Violin Test',
                       df=DATA_SET,
                       features=_features,
                       group_by=None,
                       melt=False,
                       plot_type='violin',
                       render=False,
                       file_path='test_violin.html',
                       use_auto_extensions=True
                       ).run()
        _output_file_path_2: str = 'test_violin_C.html'
        # without group by and with melt and with auto extensions
        DataVisualizer(title='Violin Test',
                       df=DATA_SET,
                       features=_features,
                       group_by=None,
                       melt=True,
                       plot_type='violin',
                       render=False,
                       file_path='test_violin.html',
                       use_auto_extensions=True
                       ).run()
        _output_file_path_3: str = 'test_violin_melt.html'
        # with group by and without melt
        DataVisualizer(title='Violin Test',
                       df=DATA_SET,
                       features=_features,
                       group_by=_group_features,
                       melt=False,
                       plot_type='violin',
                       render=False,
                       file_path='test_violin.html',
                       use_auto_extensions=False
                       ).run()
        _output_file_path_4: str = 'test_violin_C_A_0.html'
        _output_file_path_5: str = 'test_violin_C_A_1.html'
        _output_file_path_6: str = 'test_violin_C_A_2.html'
        _output_file_path_7: str = 'test_violin_C_A_4.html'
        _output_file_path_8: str = 'test_violin_C_A_5.html'
        _output_file_path_9: str = 'test_violin_C_A_8.html'
        _output_file_path_10: str = 'test_violin_C_A_9.html'
        # with group by and with melt
        DataVisualizer(title='Violin Test',
                       df=DATA_SET,
                       features=_features,
                       group_by=_group_features,
                       melt=True,
                       plot_type='violin',
                       render=False,
                       file_path='test_violin.html',
                       use_auto_extensions=False
                       ).run()
        _output_file_path_11: str = 'test_violin_C_A.html'
        self.assertTrue(expr=os.path.isfile(_output_file_path_1) and
                             os.path.isfile(_output_file_path_2) and
                             os.path.isfile(_output_file_path_3) and
                             os.path.isfile(_output_file_path_4) and
                             os.path.isfile(_output_file_path_5) and
                             os.path.isfile(_output_file_path_6) and
                             os.path.isfile(_output_file_path_7) and
                             os.path.isfile(_output_file_path_8) and
                             os.path.isfile(_output_file_path_9) and
                             os.path.isfile(_output_file_path_10) and
                             os.path.isfile(_output_file_path_11)
                        )

    def test_run_candlestick_chart(self):
        _df: pd.DataFrame = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')
        _time_features: List[str] = ['Date']
        _group_features: List[str] = ['direction']
        _group_values: List[int] = _df[_group_features[0]].unique().tolist()
        # without group by and without melt and without auto extensions
        DataVisualizer(title='Candlestick Test',
                       df=_df,
                       time_features=_time_features,
                       group_by=None,
                       melt=False,
                       plot_type='candlestick',
                       render=False,
                       file_path='test_candlestick.html',
                       use_auto_extensions=False,
                       **dict(open=['AAPL.Open'], low=['AAPL.Low'], high=['AAPL.High'], close=['AAPL.Close'])
                       ).run()
        _output_file_path_1: str = 'test_candlestick.html'
        # without group by and without melt and with auto extensions
        DataVisualizer(title='Candlestick Test',
                       df=_df,
                       time_features=_time_features,
                       group_by=None,
                       melt=False,
                       plot_type='candlestick',
                       render=False,
                       file_path='test_candlestick.html',
                       use_auto_extensions=True,
                       **dict(open=['AAPL.Open'], low=['AAPL.Low'], high=['AAPL.High'], close=['AAPL.Close'])
                       ).run()
        _output_file_path_2: str = 'test_candlestick_Date_AAPL..html'
        # without group by and with melt and with auto extensions
        DataVisualizer(title='Candlestick Test',
                       df=_df,
                       time_features=_time_features,
                       group_by=None,
                       melt=True,
                       plot_type='candlestick',
                       render=False,
                       file_path='test_candlestick.html',
                       use_auto_extensions=True,
                       **dict(open=['AAPL.Open'], low=['AAPL.Low'], high=['AAPL.High'], close=['AAPL.Close'])
                       ).run()
        _output_file_path_3: str = 'test_candlestick_melt.html'
        # with group by and without melt
        DataVisualizer(title='Candlestick Test',
                       df=_df,
                       time_features=_time_features,
                       group_by=_group_features,
                       melt=False,
                       plot_type='candlestick',
                       render=False,
                       file_path='test_candlestick.html',
                       use_auto_extensions=False,
                       **dict(open=['AAPL.Open'], low=['AAPL.Low'], high=['AAPL.High'], close=['AAPL.Close'])
                       ).run()
        _output_file_path_4: str = 'test_candlestick_Date_AAPL._direction_Decreasing.html'
        _output_file_path_5: str = 'test_candlestick_Date_AAPL._direction_Increasing.html'
        self.assertTrue(expr=os.path.isfile(_output_file_path_1) and
                             os.path.isfile(_output_file_path_2) and
                             os.path.isfile(_output_file_path_3) and
                             os.path.isfile(_output_file_path_4) and
                             os.path.isfile(_output_file_path_5)
                        )

    def test_run_choroleth_map_chart(self):
        pass

    def test_run_contour_chart(self):
        _features: List[str] = ['C', 'H']
        DataVisualizer(title='Contour Test',
                       df=DATA_SET,
                       features=_features,
                       group_by=None,
                       melt=False,
                       plot_type='contour',
                       render=False,
                       file_path='test_contour.html',
                       use_auto_extensions=False
                       ).run()
        self.assertTrue(expr=os.path.isfile('test_contour.html'))

    def test_run_dendrogram_chart(self):
        _features: List[str] = ['C', 'H']
        DataVisualizer(title='Dendrogram Test',
                       df=DATA_SET,
                       features=_features,
                       group_by=None,
                       melt=False,
                       plot_type='dendro',
                       render=False,
                       file_path='test_dendrogram.html',
                       use_auto_extensions=False
                       ).run()
        self.assertTrue(expr=os.path.isfile('test_dendrogram.html'))

    def test_run_density_map_chart(self):
        _df: pd.DataFrame = pd.DataFrame()
        _df['lat'] = [30, 6, 6, 30, 30, None, 20, 30, 30, 20, 20, None, 40, 50, 50, 40, 40]
        _df['lon'] = [-10, -10, 8, 8, -10, None, 30, 30, 50, 50, 30, None, 100, 100, 80, 80, 100]
        _df['val'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        DataVisualizer(title='Density Map Test',
                       df=_df,
                       features=['val'],
                       plot_type='density',
                       render=False,
                       file_path='test_density_map.html',
                       **dict(lat='lat', lon='lon')
                       ).run()
        self.assertTrue(expr=os.path.isfile('test_density_map.html'))

    def test_run_dist_chart(self):
        _df: pd.DataFrame = pd.read_csv('wine.csv')
        _features: List[str] = ['alcohol']
        _group_features: List[str] = ['class']
        _group_values: List[int] = _df[_group_features[0]].unique().tolist()
        # without group by and without melt and without auto extensions
        DataVisualizer(title='Dist Test',
                       df=_df,
                       features=_features,
                       group_by=None,
                       melt=False,
                       plot_type='dist',
                       render=False,
                       file_path='test_dist.html',
                       use_auto_extensions=False
                       ).run()
        _output_file_path_1: str = 'test_dist.html'
        # without group by and without melt and with auto extensions
        DataVisualizer(title='Dist Test',
                       df=_df,
                       features=_features,
                       group_by=None,
                       melt=False,
                       plot_type='dist',
                       render=False,
                       file_path='test_dist.html',
                       use_auto_extensions=True
                       ).run()
        _output_file_path_2: str = 'test_dist_alcohol.html'
        # without group by and with melt and with auto extensions
        DataVisualizer(title='Dist Test',
                       df=_df,
                       features=_features,
                       group_by=None,
                       melt=True,
                       plot_type='dist',
                       render=False,
                       file_path='test_dist.html',
                       use_auto_extensions=True
                       ).run()
        _output_file_path_3: str = 'test_dist_melt.html'
        # with group by and without melt
        DataVisualizer(title='Dist Test',
                       df=_df,
                       features=_features,
                       group_by=_group_features,
                       melt=False,
                       plot_type='dist',
                       render=False,
                       file_path='test_dist.html',
                       use_auto_extensions=False
                       ).run()
        _output_file_path_4: str = 'test_dist_alcohol_class_1.html'
        _output_file_path_5: str = 'test_dist_alcohol_class_2.html'
        _output_file_path_6: str = 'test_dist_alcohol_class_3.html'
        # with group by and with melt
        DataVisualizer(title='Dist Test',
                       df=_df,
                       features=_features,
                       group_by=_group_features,
                       melt=True,
                       plot_type='dist',
                       render=False,
                       file_path='test_dist.html',
                       use_auto_extensions=False
                       ).run()
        _output_file_path_7: str = 'test_dist_alcohol_class.html'
        self.assertTrue(expr=os.path.isfile(_output_file_path_1) and
                             os.path.isfile(_output_file_path_2) and
                             os.path.isfile(_output_file_path_3) and
                             os.path.isfile(_output_file_path_4) and
                             os.path.isfile(_output_file_path_5) and
                             os.path.isfile(_output_file_path_6) and
                             os.path.isfile(_output_file_path_7)
                        )

    def test_run_funnel_chart(self):
        pass

    def test_run_geo_map_chart(self):
        _df: pd.DataFrame = pd.DataFrame()
        _df['lat'] = [30, 6, 6, 30, 30, None, 20, 30, 30, 20, 20, None, 40, 50, 50, 40, 40]
        _df['lon'] = [-10, -10, 8, 8, -10, None, 30, 30, 50, 50, 30, None, 100, 100, 80, 80, 100]
        _df['val'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        DataVisualizer(title='Geomap Test',
                       df=_df,
                       features=['val'],
                       plot_type='geo',
                       render=False,
                       file_path='test_geomap.html',
                       **dict(lat='lat', lon='lon')
                       ).run()
        self.assertTrue(expr=os.path.isfile('test_geomap.html'))

    def test_run_heat_map_chart(self):
        _confusion_matrix: np.array = np.array([[23, 5], [3, 30]])
        _df: pd.DataFrame = pd.DataFrame(data=_confusion_matrix,
                                         columns=['obs', 'pred'],
                                         index=['obs', 'pred']
                                         )
        DataVisualizer(title='Heat Map Test',
                       df=_df,
                       features=['obs', 'pred'],
                       group_by=None,
                       melt=False,
                       plot_type='heat',
                       render=False,
                       file_path='test_heatmap.html',
                       use_auto_extensions=False
                       ).run()
        self.assertTrue(expr=os.path.isfile('test_heatmap.html'))

    def test_run_histogramm_chart(self):
        _features: List[str] = ['H']
        _group_features: List[str] = ['A']
        _group_values: List[int] = DATA_SET[_group_features[0]].unique().tolist()
        # without group by and without melt and without auto extensions
        DataVisualizer(title='Histogram Test',
                       df=DATA_SET,
                       features=_features,
                       group_by=None,
                       melt=False,
                       plot_type='hist',
                       render=False,
                       file_path='test_histogram.html',
                       use_auto_extensions=False
                       ).run()
        _output_file_path_1: str = 'test_histogram.html'
        # without group by and without melt and with auto extensions
        DataVisualizer(title='Histogram Test',
                       df=DATA_SET,
                       features=_features,
                       group_by=None,
                       melt=False,
                       plot_type='hist',
                       render=False,
                       file_path='test_histogram.html',
                       use_auto_extensions=True
                       ).run()
        _output_file_path_2: str = 'test_histogram_H.html'
        # without group by and with melt and with auto extensions
        DataVisualizer(title='Histogram Test',
                       df=DATA_SET,
                       features=_features,
                       group_by=None,
                       melt=True,
                       plot_type='hist',
                       render=False,
                       file_path='test_histogram.html',
                       use_auto_extensions=True
                       ).run()
        _output_file_path_3: str = 'test_histogram_melt.html'
        # with group by and without melt
        DataVisualizer(title='Histogram Test',
                       df=DATA_SET,
                       features=_features,
                       group_by=_group_features,
                       melt=False,
                       plot_type='hist',
                       render=False,
                       file_path='test_histogram.html',
                       use_auto_extensions=False
                       ).run()
        _output_file_path_4: str = 'test_histogram_H_A_0.html'
        _output_file_path_5: str = 'test_histogram_H_A_1.html'
        _output_file_path_6: str = 'test_histogram_H_A_2.html'
        _output_file_path_7: str = 'test_histogram_H_A_4.html'
        _output_file_path_8: str = 'test_histogram_H_A_5.html'
        _output_file_path_9: str = 'test_histogram_H_A_8.html'
        _output_file_path_10: str = 'test_histogram_H_A_9.html'
        # with group by and with melt
        DataVisualizer(title='Histogram Test',
                       df=DATA_SET,
                       features=_features,
                       group_by=_group_features,
                       melt=True,
                       plot_type='hist',
                       render=False,
                       file_path='test_histogram.html',
                       use_auto_extensions=False
                       ).run()
        _output_file_path_11: str = 'test_histogram_H_A.html'
        self.assertTrue(expr=os.path.isfile(_output_file_path_1) and
                             os.path.isfile(_output_file_path_2) and
                             os.path.isfile(_output_file_path_3) and
                             os.path.isfile(_output_file_path_4) and
                             os.path.isfile(_output_file_path_5) and
                             os.path.isfile(_output_file_path_6) and
                             os.path.isfile(_output_file_path_7) and
                             os.path.isfile(_output_file_path_8) and
                             os.path.isfile(_output_file_path_9) and
                             os.path.isfile(_output_file_path_10) and
                             os.path.isfile(_output_file_path_11)
                        )

    def test_run_histogram_decile_chart(self):
        _df: pd.DataFrame = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2016-weather-data-seattle.csv')
        _df['month'] = pd.to_datetime(_df['Date']).dt.month
        _df['week'] = pd.to_datetime(_df['Date']).dt.week
        DataVisualizer(title='Histogram-Decile Test',
                       df=_df,
                       features=['Max_TemperatureC'],
                       group_by=['month'],
                       time_features=['week'],
                       melt=False,
                       plot_type='hist_decile',
                       render=False,
                       file_path='test_histogram_decile.html',
                       use_auto_extensions=False
                       ).run()
        self.assertTrue(expr=os.path.isfile('test_histogram_decile.html'))

    def test_run_joint_distribution_chart(self):
        _features: List[str] = ['C', 'H']
        _group_features: List[str] = ['A']
        _group_values: List[int] = DATA_SET[_group_features[0]].unique().tolist()
        # without group by and without melt and without auto extensions
        DataVisualizer(title='Joint Distribution Test',
                       df=DATA_SET,
                       features=_features,
                       group_by=None,
                       melt=False,
                       plot_type='joint',
                       render=False,
                       file_path='test_joint.html',
                       use_auto_extensions=False
                       ).run()
        _output_file_path_1: str = 'test_joint.html'
        # without group by and without melt and with auto extensions
        DataVisualizer(title='Joint Distribution Test',
                       df=DATA_SET,
                       features=_features,
                       group_by=None,
                       melt=False,
                       plot_type='joint',
                       render=False,
                       file_path='test_joint.html',
                       use_auto_extensions=True
                       ).run()
        _output_file_path_2: str = 'test_joint_C_H.html'
        # with group by and without melt
        DataVisualizer(title='Joint Distribution Test',
                       df=DATA_SET,
                       features=_features,
                       group_by=_group_features,
                       melt=False,
                       plot_type='joint',
                       render=False,
                       file_path='test_joint.html',
                       use_auto_extensions=False
                       ).run()
        _output_file_path_4: str = 'test_joint_C_H_A_0.html'
        _output_file_path_5: str = 'test_joint_C_H_A_1.html'
        _output_file_path_6: str = 'test_joint_C_H_A_2.html'
        _output_file_path_7: str = 'test_joint_C_H_A_4.html'
        _output_file_path_8: str = 'test_joint_C_H_A_5.html'
        _output_file_path_9: str = 'test_joint_C_H_A_8.html'
        _output_file_path_10: str = 'test_joint_C_H_A_9.html'
        self.assertTrue(expr=os.path.isfile(_output_file_path_1) and
                             os.path.isfile(_output_file_path_2) and
                             os.path.isfile(_output_file_path_4) and
                             os.path.isfile(_output_file_path_5) and
                             os.path.isfile(_output_file_path_6) and
                             os.path.isfile(_output_file_path_7) and
                             os.path.isfile(_output_file_path_8) and
                             os.path.isfile(_output_file_path_9) and
                             os.path.isfile(_output_file_path_10)
                        )

    def test_run_line_chart(self):
        _features: List[str] = ['F']
        _time_features: List[str] = ['D']
        _group_features: List[str] = ['A']
        _group_values: List[int] = DATA_SET[_group_features[0]].unique().tolist()
        # without group by and without melt and without auto extensions
        DataVisualizer(title='Line Test',
                       df=DATA_SET,
                       features=_features,
                       time_features=_time_features,
                       group_by=None,
                       melt=False,
                       plot_type='line',
                       render=False,
                       file_path='test_line.html',
                       use_auto_extensions=False
                       ).run()
        _output_file_path_1: str = 'test_line.html'
        # without group by and without melt and with auto extensions
        DataVisualizer(title='Line Test',
                       df=DATA_SET,
                       features=_features,
                       time_features=_time_features,
                       group_by=None,
                       melt=False,
                       plot_type='line',
                       render=False,
                       file_path='test_line.html',
                       use_auto_extensions=True
                       ).run()
        _output_file_path_2: str = 'test_line_D_F.html'
        # without group by and with melt and with auto extensions
        DataVisualizer(title='Line Test',
                       df=DATA_SET,
                       features=_features,
                       time_features=_time_features,
                       group_by=None,
                       melt=True,
                       plot_type='line',
                       render=False,
                       file_path='test_line.html',
                       use_auto_extensions=True
                       ).run()
        _output_file_path_3: str = 'test_line_melt.html'
        # with group by and without melt
        DataVisualizer(title='Line Test',
                       df=DATA_SET,
                       features=_features,
                       time_features=_time_features,
                       group_by=_group_features,
                       melt=False,
                       plot_type='line',
                       render=False,
                       file_path='test_line.html',
                       use_auto_extensions=False
                       ).run()
        _output_file_path_4: str = 'test_line_D_F_A_0.html'
        _output_file_path_5: str = 'test_line_D_F_A_1.html'
        _output_file_path_6: str = 'test_line_D_F_A_2.html'
        _output_file_path_7: str = 'test_line_D_F_A_4.html'
        _output_file_path_8: str = 'test_line_D_F_A_5.html'
        _output_file_path_9: str = 'test_line_D_F_A_8.html'
        _output_file_path_10: str = 'test_line_D_F_A_9.html'
        # with group by and with melt
        DataVisualizer(title='Line Test',
                       df=DATA_SET,
                       features=_features,
                       time_features=_time_features,
                       group_by=_group_features,
                       melt=True,
                       plot_type='line',
                       render=False,
                       file_path='test_line.html',
                       use_auto_extensions=False
                       ).run()
        _output_file_path_11: str = 'test_line_D_F_A.html'
        self.assertTrue(expr=os.path.isfile(_output_file_path_1) and
                             os.path.isfile(_output_file_path_2) and
                             os.path.isfile(_output_file_path_3) and
                             os.path.isfile(_output_file_path_4) and
                             os.path.isfile(_output_file_path_5) and
                             os.path.isfile(_output_file_path_6) and
                             os.path.isfile(_output_file_path_7) and
                             os.path.isfile(_output_file_path_8) and
                             os.path.isfile(_output_file_path_9) and
                             os.path.isfile(_output_file_path_10) and
                             os.path.isfile(_output_file_path_11)
                        )

    def test_run_line_bar_chart(self):
        _features: List[str] = ['F']
        _time_features: List[str] = ['D']
        _group_features: List[str] = ['A']
        _group_values: List[int] = DATA_SET[_group_features[0]].unique().tolist()
        # without group by and without melt and without auto extensions
        DataVisualizer(title='Line-Bar Test',
                       df=DATA_SET,
                       features=_features,
                       time_features=_time_features,
                       group_by=None,
                       melt=False,
                       plot_type='line_bar',
                       render=False,
                       file_path='test_line_bar.html',
                       use_auto_extensions=False
                       ).run()
        _output_file_path_1: str = 'test_line_bar.html'
        # without group by and without melt and with auto extensions
        DataVisualizer(title='Line-Bar Test',
                       df=DATA_SET,
                       features=_features,
                       time_features=_time_features,
                       group_by=None,
                       melt=False,
                       plot_type='line_bar',
                       render=False,
                       file_path='test_line_bar.html',
                       use_auto_extensions=True
                       ).run()
        _output_file_path_2: str = 'test_line_bar_D_F.html'
        # without group by and with melt and with auto extensions
        DataVisualizer(title='Line-Bar Test',
                       df=DATA_SET,
                       features=_features,
                       time_features=_time_features,
                       group_by=None,
                       melt=True,
                       plot_type='line_bar',
                       render=False,
                       file_path='test_line_bar.html',
                       use_auto_extensions=True
                       ).run()
        _output_file_path_3: str = 'test_line_bar_melt.html'
        # with group by and without melt
        DataVisualizer(title='Line-Bar Test',
                       df=DATA_SET,
                       features=_features,
                       time_features=_time_features,
                       group_by=_group_features,
                       melt=False,
                       plot_type='line_bar',
                       render=False,
                       file_path='test_line_bar.html',
                       use_auto_extensions=False
                       ).run()
        _output_file_path_4: str = 'test_line_bar_D_F_A_0.html'
        _output_file_path_5: str = 'test_line_bar_D_F_A_1.html'
        _output_file_path_6: str = 'test_line_bar_D_F_A_2.html'
        _output_file_path_7: str = 'test_line_bar_D_F_A_4.html'
        _output_file_path_8: str = 'test_line_bar_D_F_A_5.html'
        _output_file_path_9: str = 'test_line_bar_D_F_A_8.html'
        _output_file_path_10: str = 'test_line_bar_D_F_A_9.html'
        # with group by and with melt
        DataVisualizer(title='Line-Bar Test',
                       df=DATA_SET,
                       features=_features,
                       time_features=_time_features,
                       group_by=_group_features,
                       melt=True,
                       plot_type='line_bar',
                       render=False,
                       file_path='test_line_bar.html',
                       use_auto_extensions=False
                       ).run()
        _output_file_path_11: str = 'test_line_bar_D_F_A.html'
        self.assertTrue(expr=os.path.isfile(_output_file_path_1) and
                             os.path.isfile(_output_file_path_2) and
                             os.path.isfile(_output_file_path_3) and
                             os.path.isfile(_output_file_path_4) and
                             os.path.isfile(_output_file_path_5) and
                             os.path.isfile(_output_file_path_6) and
                             os.path.isfile(_output_file_path_7) and
                             os.path.isfile(_output_file_path_8) and
                             os.path.isfile(_output_file_path_9) and
                             os.path.isfile(_output_file_path_10) and
                             os.path.isfile(_output_file_path_11)
                        )

    def test_run_network_graph_chart(self):
        pass

    def test_run_multi_chart(self):
        pass

    def test_run_parallel_category_chart(self):
        pass

    def test_run_parallel_coordinate_chart(self):
        _features: List[str] = ['A', 'B', 'C', 'D', 'E']
        _color_feature: str = 'F'
        DataVisualizer(title='Parallel Coordinate Test',
                       df=DATA_SET,
                       features=_features,
                       group_by=None,
                       melt=False,
                       color_feature=_color_feature,
                       plot_type='parcoords',
                       render=False,
                       file_path='test_parcoords.html',
                       use_auto_extensions=False
                       ).run()
        self.assertTrue(expr=os.path.isfile('test_parcoords.html'))

    def test_run_pie_chart(self):
        _features: List[str] = ['F']
        _group_features: List[str] = ['A']
        _group_values: List[int] = DATA_SET[_group_features[0]].unique().tolist()
        # without group by and without melt and without auto extensions
        DataVisualizer(title='Pie Test',
                       df=DATA_SET,
                       features=_features,
                       group_by=None,
                       melt=False,
                       plot_type='pie',
                       render=False,
                       file_path='test_pie.html',
                       use_auto_extensions=False
                       ).run()
        _output_file_path_1: str = 'test_pie.html'
        # without group by and without melt and with auto extensions
        DataVisualizer(title='Pie Test',
                       df=DATA_SET,
                       features=_features,
                       group_by=None,
                       melt=False,
                       plot_type='pie',
                       render=False,
                       file_path='test_pie.html',
                       use_auto_extensions=True
                       ).run()
        _output_file_path_2: str = 'test_pie_F.html'
        # with group by and without melt
        DataVisualizer(title='Pie Test',
                       df=DATA_SET,
                       features=_features,
                       group_by=_group_features,
                       melt=False,
                       plot_type='pie',
                       render=False,
                       file_path='test_pie.html',
                       use_auto_extensions=False
                       ).run()
        _output_file_path_4: str = 'test_pie_F_A_0.html'
        _output_file_path_5: str = 'test_pie_F_A_1.html'
        _output_file_path_6: str = 'test_pie_F_A_2.html'
        _output_file_path_7: str = 'test_pie_F_A_4.html'
        _output_file_path_8: str = 'test_pie_F_A_5.html'
        _output_file_path_9: str = 'test_pie_F_A_8.html'
        _output_file_path_10: str = 'test_pie_F_A_9.html'
        self.assertTrue(expr=os.path.isfile(_output_file_path_1) and
                             os.path.isfile(_output_file_path_2) and
                             os.path.isfile(_output_file_path_4) and
                             os.path.isfile(_output_file_path_5) and
                             os.path.isfile(_output_file_path_6) and
                             os.path.isfile(_output_file_path_7) and
                             os.path.isfile(_output_file_path_8) and
                             os.path.isfile(_output_file_path_9) and
                             os.path.isfile(_output_file_path_10)
                        )

    def test_run_radar_chart(self):
        pass

    def test_run_ridgeline_chart(self):
        _df: pd.DataFrame = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2016-weather-data-seattle.csv')
        _df['month'] = pd.to_datetime(_df['Date']).dt.month
        _df['week'] = pd.to_datetime(_df['Date']).dt.week
        _features: List[str] = ['Max_TemperatureC']
        _time_features: List[str] = ['week']
        _group_features: List[str] = ['month']
        _group_values: List[int] = _df[_group_features[0]].unique().tolist()
        # without group by and without auto extensions
        DataVisualizer(title='Ridgeline Test',
                       df=_df,
                       features=_features,
                       time_features=_time_features,
                       group_by=None,
                       melt=False,
                       plot_type='ridgeline',
                       render=False,
                       file_path='test_ridgeline.html',
                       use_auto_extensions=False
                       ).run()
        _output_file_path_1: str = 'test_ridgeline.html'
        # without group by and with auto extensions
        DataVisualizer(title='Ridgeline Test',
                       df=_df,
                       features=_features,
                       time_features=_time_features,
                       group_by=None,
                       melt=False,
                       plot_type='ridgeline',
                       render=False,
                       file_path='test_ridgeline.html',
                       use_auto_extensions=True
                       ).run()
        _output_file_path_2: str = 'test_ridgeline_week_Max_TemperatureC.html'
        # with group by
        DataVisualizer(title='Ridgeline Test',
                       df=_df,
                       features=_features,
                       time_features=_time_features,
                       group_by=_group_features,
                       melt=False,
                       plot_type='ridgeline',
                       render=False,
                       file_path='test_ridgeline.html',
                       use_auto_extensions=False
                       ).run()
        _output_file_path_3: str = 'test_ridgeline_week_Max_TemperatureC_month_1.html'
        _output_file_path_4: str = 'test_ridgeline_week_Max_TemperatureC_month_2.html'
        _output_file_path_5: str = 'test_ridgeline_week_Max_TemperatureC_month_3.html'
        _output_file_path_6: str = 'test_ridgeline_week_Max_TemperatureC_month_4.html'
        _output_file_path_7: str = 'test_ridgeline_week_Max_TemperatureC_month_5.html'
        _output_file_path_8: str = 'test_ridgeline_week_Max_TemperatureC_month_6.html'
        _output_file_path_9: str = 'test_ridgeline_week_Max_TemperatureC_month_7.html'
        _output_file_path_10: str = 'test_ridgeline_week_Max_TemperatureC_month_8.html'
        _output_file_path_11: str = 'test_ridgeline_week_Max_TemperatureC_month_9.html'
        _output_file_path_12: str = 'test_ridgeline_week_Max_TemperatureC_month_10.html'
        _output_file_path_13: str = 'test_ridgeline_week_Max_TemperatureC_month_11.html'
        _output_file_path_14: str = 'test_ridgeline_week_Max_TemperatureC_month_12.html'
        self.assertTrue(expr=os.path.isfile(_output_file_path_1) and
                             os.path.isfile(_output_file_path_2) and
                             os.path.isfile(_output_file_path_3) and
                             os.path.isfile(_output_file_path_4) and
                             os.path.isfile(_output_file_path_5) and
                             os.path.isfile(_output_file_path_6) and
                             os.path.isfile(_output_file_path_7) and
                             os.path.isfile(_output_file_path_8) and
                             os.path.isfile(_output_file_path_9) and
                             os.path.isfile(_output_file_path_10) and
                             os.path.isfile(_output_file_path_11) and
                             os.path.isfile(_output_file_path_12) and
                             os.path.isfile(_output_file_path_13) and
                             os.path.isfile(_output_file_path_14)
                        )

    def test_run_scatter_chart(self):
        _features: List[str] = ['C', 'H']
        _group_features: List[str] = ['A']
        _group_values: List[int] = DATA_SET[_group_features[0]].unique().tolist()
        # without group by and without melt and without auto extensions
        DataVisualizer(title='Scatter Test',
                       df=DATA_SET,
                       features=_features,
                       group_by=None,
                       melt=False,
                       plot_type='scatter',
                       render=False,
                       file_path='test_scatter.html',
                       use_auto_extensions=False
                       ).run()
        _output_file_path_1: str = 'test_scatter.html'
        # without group by and without melt and with auto extensions
        DataVisualizer(title='Scatter Test',
                       df=DATA_SET,
                       features=_features,
                       group_by=None,
                       melt=False,
                       plot_type='scatter',
                       render=False,
                       file_path='test_scatter.html',
                       use_auto_extensions=True
                       ).run()
        _output_file_path_2: str = 'test_scatter_C_H.html'
        # without group by and with melt and with auto extensions
        DataVisualizer(title='Scatter Test',
                       df=DATA_SET,
                       features=_features,
                       group_by=None,
                       melt=True,
                       plot_type='scatter',
                       render=False,
                       file_path='test_scatter.html',
                       use_auto_extensions=True
                       ).run()
        _output_file_path_3: str = 'test_scatter_melt.html'
        # with group by and without melt
        DataVisualizer(title='Scatter Test',
                       df=DATA_SET,
                       features=_features,
                       group_by=_group_features,
                       melt=False,
                       plot_type='scatter',
                       render=False,
                       file_path='test_scatter.html',
                       use_auto_extensions=False
                       ).run()
        _output_file_path_4: str = 'test_scatter_C_H_A_0.html'
        _output_file_path_5: str = 'test_scatter_C_H_A_1.html'
        _output_file_path_6: str = 'test_scatter_C_H_A_2.html'
        _output_file_path_7: str = 'test_scatter_C_H_A_4.html'
        _output_file_path_8: str = 'test_scatter_C_H_A_5.html'
        _output_file_path_9: str = 'test_scatter_C_H_A_8.html'
        _output_file_path_10: str = 'test_scatter_C_H_A_9.html'
        # with group by and with melt
        DataVisualizer(title='Scatter Test',
                       df=DATA_SET,
                       features=_features,
                       group_by=_group_features,
                       melt=True,
                       plot_type='scatter',
                       render=False,
                       file_path='test_scatter.html',
                       use_auto_extensions=False
                       ).run()
        _output_file_path_11: str = 'test_scatter_H_A.html'
        self.assertTrue(expr=os.path.isfile(_output_file_path_1) and
                             os.path.isfile(_output_file_path_2) and
                             os.path.isfile(_output_file_path_3) and
                             os.path.isfile(_output_file_path_4) and
                             os.path.isfile(_output_file_path_5) and
                             os.path.isfile(_output_file_path_6) and
                             os.path.isfile(_output_file_path_7) and
                             os.path.isfile(_output_file_path_8) and
                             os.path.isfile(_output_file_path_9) and
                             os.path.isfile(_output_file_path_10) and
                             os.path.isfile(_output_file_path_11)
                        )

    def test_run_scatter_3d_chart(self):
        _features: List[str] = ['C', 'H', 'K']
        _group_features: List[str] = ['A']
        _group_values: List[int] = DATA_SET[_group_features[0]].unique().tolist()
        # without group by and without melt and without auto extensions
        DataVisualizer(title='Scatter 3D Test',
                       df=DATA_SET,
                       features=_features,
                       group_by=None,
                       melt=False,
                       plot_type='scatter3d',
                       render=False,
                       file_path='test_scatter3d.html',
                       use_auto_extensions=False
                       ).run()
        _output_file_path_1: str = 'test_scatter3d.html'
        # without group by and without melt and with auto extensions
        DataVisualizer(title='Scatter 3D Test',
                       df=DATA_SET,
                       features=_features,
                       group_by=None,
                       melt=False,
                       plot_type='scatter3d',
                       render=False,
                       file_path='test_scatter3d.html',
                       use_auto_extensions=True
                       ).run()
        _output_file_path_2: str = 'test_scatter3d_C_H_K.html'
        # without group by and with melt and with auto extensions
        DataVisualizer(title='Scatter 3D Test',
                       df=DATA_SET,
                       features=_features,
                       group_by=None,
                       melt=True,
                       plot_type='scatter3d',
                       render=False,
                       file_path='test_scatter3d.html',
                       use_auto_extensions=True
                       ).run()
        _output_file_path_3: str = 'test_scatter3d_melt.html'
        # with group by and without melt
        DataVisualizer(title='Scatter 3D Test',
                       df=DATA_SET,
                       features=_features,
                       group_by=_group_features,
                       melt=False,
                       plot_type='scatter3d',
                       render=False,
                       file_path='test_scatter3d.html',
                       use_auto_extensions=False
                       ).run()
        _output_file_path_4: str = 'test_scatter3d_C_H_K_A_0.html'
        _output_file_path_5: str = 'test_scatter3d_C_H_K_A_1.html'
        _output_file_path_6: str = 'test_scatter3d_C_H_K_A_2.html'
        _output_file_path_7: str = 'test_scatter3d_C_H_K_A_4.html'
        _output_file_path_8: str = 'test_scatter3d_C_H_K_A_5.html'
        _output_file_path_9: str = 'test_scatter3d_C_H_K_A_8.html'
        _output_file_path_10: str = 'test_scatter3d_C_H_K_A_9.html'
        # with group by and with melt
        DataVisualizer(title='Scatter 3D Test',
                       df=DATA_SET,
                       features=_features,
                       group_by=_group_features,
                       melt=True,
                       plot_type='scatter3d',
                       render=False,
                       file_path='test_scatter3d.html',
                       use_auto_extensions=False
                       ).run()
        _output_file_path_11: str = 'test_scatter3d_C_H_K_A.html'
        self.assertTrue(expr=os.path.isfile(_output_file_path_1) and
                             os.path.isfile(_output_file_path_2) and
                             os.path.isfile(_output_file_path_3) and
                             os.path.isfile(_output_file_path_4) and
                             os.path.isfile(_output_file_path_5) and
                             os.path.isfile(_output_file_path_6) and
                             os.path.isfile(_output_file_path_7) and
                             os.path.isfile(_output_file_path_8) and
                             os.path.isfile(_output_file_path_9) and
                             os.path.isfile(_output_file_path_10) and
                             os.path.isfile(_output_file_path_11)
                        )

    def test_run_silhouette_chart(self):
        _df: pd.DataFrame = pd.read_csv('wine.csv')
        _features: List[str] = ['alcohol', 'malic_acid', 'ash']
        _clustering = Clustering(cl_params=dict(n_clusters=3)).kmeans()
        _clustering.fit(X=_df[_features])
        _silhouette: dict = UnsupervisedML(df=_df,
                                           n_cluster_components=3
                                           ).silhouette_analysis(labels=_clustering.predict(_df[_features]))
        DataVisualizer(title='Silhouette Test',
                       df=_df,
                       features=None,
                       plot_type='silhouette',
                       file_path='test_silhouette.html',
                       **dict(layout={},
                              n_clusters=3,
                              silhouette=_silhouette
                              )
                       ).run()
        self.assertTrue(expr=os.path.isfile('test_silhouette.html'))

    def test_run_sunburst_chart(self):
        pass

    def test_run_tree_map_chart(self):
        pass

    def test_run_table_chart(self):
        _features: List[str] = ['A', 'B', 'C', 'D', 'E']
        DataVisualizer(title='Table Test',
                       df=DATA_SET,
                       features=_features,
                       group_by=None,
                       melt=False,
                       plot_type='table',
                       render=False,
                       file_path='test_table.html',
                       use_auto_extensions=False
                       ).run()
        self.assertTrue(expr=os.path.isfile('test_table.html'))


if __name__ == '__main__':
    unittest.main()
