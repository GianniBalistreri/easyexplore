import copy
import dask.dataframe as dd
import ipywidgets as widgets
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import re
import string

from .data_import_export import DataImporter, FileUtils
from .interactive_visualizer import PlotlyAdapter
from .utils import EasyExploreUtils, INVALID_VALUES, Log
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import Image, display, HTML
from plotly.colors import n_colors
from plotly.offline import init_notebook_mode, iplot
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List

# Show all code cells outputs
InteractiveShell.ast_node_interactivity = 'all'
# Supported visualization frameworks
framework: Dict[str, List[str]] = dict(interactive=['plotly'], static=[])
# Supported visualization methods
plots: List[str] = ['bar',
                    'box',
                    'candlestick',
                    #'choro',
                    'contour',
                    'contour_hist',
                    'dendro',
                    'density',
                    'dist',
                    #'funnel',
                    #'funnel_area',
                    'geo',
                    'heat',
                    'hist',
                    'hist_decile',
                    'joint',
                    'line',
                    'multi',
                    'network',
                    #'radar',
                    'ridgeline',
                    'parcats',
                    'parcoords',
                    'pie',
                    'scatter',
                    'scatter3d',
                    'silhouette',
                    #'sunburst',
                    'table',
                    #'tree',
                    'violin'
                    ]

# TODO:
#  Geo Stats Hovertemplate
#  Chorolethmap: handle geojson input output
#  Check Missings in each categorical distribution chart
#  Fix bugs in Parcats brushing feature


class DataVisualizerException(Exception):
    """
    Class for handling exception for class DataVisualizer
    """
    pass


class DataVisualizer:
    """
    Class for visualizing data in jupyter notebooks
    """
    def __init__(self,
                 df=None,
                 title: str = '',
                 features: List[str] = None,
                 time_features: List[str] = None,
                 graph_features: Dict[str, str] = None,
                 group_by: List[str] = None,
                 feature_types: Dict[str, List[str]] = None,
                 plot_type: str = None,
                 subplots: dict = None,
                 melt: bool = False,
                 brushing: bool = True,
                 xaxis_label: List[str] = None,
                 yaxis_label: List[str] = None,
                 zaxis_label: List[str] = None,
                 annotations: List[dict] = None,
                 width: int = 500,
                 height: int = 500,
                 unit: str = 'px',
                 interactive: bool = True,
                 file_path: str = None,
                 render: bool = True,
                 color_scale: List[str] = None,
                 color_edges: List[str] = None,
                 color_feature: str = None,
                 max_row: int = 50,
                 max_col: int = 20,
                 rows_sub: int = None,
                 cols_sub: int = None,
                 **kwargs
                 ):
        """
        :param plot_type: str
            Name of the plot type
                -> bar: Bar plot
                -> bi: Bivariate plot (categorical - continuous)
                -> box: Box-Whisker plot
                -> geo: Geo map
                -> heat: Heat map
                -> hist: Histogram
                -> line: Line plot
                -> pie: Pie plot
                -> scatter: Scatter plot
                -> table: Table plot

        :param title: str
            Name of the plot title

        :param df: Pandas DataFrame oder dask dataframe
            Data set

        :param features: List[str]
            Name of the features

        :param time_features: List[str]
            Name of the time regarding features in line plots

        :param feature_types: Dict[str, List[str]]
            Pre-defined feature type segmentation

        :param melt: bool
            Melt subplots into one main plot

        :param brushing: bool
            Generate additional scatter chart for case-based exploration of feature connections

        :param xaxis_label: List[str]
            User based labeling of the xaxis for all subplots

        :param yaxis_label: List[str]
            User based labeling of the yaxis for all subplots

        :param annotations: List[dict]
            Annotation configuration for each subplot

        :param subplots: dict
            Subplot configuration

        :param rows_sub: int
            Number of rows to use in subplot

        :param cols_sub: int
            Number of columns to use in subplot

        :param width: int
            Width size for each subplot

        :param height: int
            Height size for each subplot

        :param unit: str
            Measurement unit
                -> px, pixel: Pixel
                -> in, inch: Inch
                -> cm, centimeter: Centimeter

        :param interactive: bool
            Use interactive plot with plotly or classic seaborn / matplotlib

        :param file_path: str
            File path of the plot to save

        :param render: bool
            Render plotly chart or not

        :param colors: dict
            Set user-based color palette

        :param max_row: int
            Maximum number of rows of visualized Pandas DataFrames

        :param max_col: int
            Maximum number of columns of visualized Pandas DataFrames

        :param log_path: str
            Path of the log file

        :param kwargs: dict
            Key word arguments for handling plotly
        """
        self.interactive: bool = interactive
        _features: List[str] = [] if features is None else features
        _group_by_features: List[str] = [] if group_by is None else group_by
        _time_features: List[str] = [] if time_features is None else time_features
        _graph_features: List[str] = [] if graph_features is None else graph_features
        _color_features: List[str] = [] if color_feature is None else [color_feature]
        _geo_features: List[str] = []
        if kwargs.get('lat') is not None:
            _geo_features.append(kwargs.get('lat'))
        if kwargs.get('lon') is not None:
            _geo_features.append(kwargs.get('lon'))
        _all_features: List[str] = _features + _group_by_features + _time_features + _graph_features + _color_features + _geo_features
        _all_features = list(set(_all_features))
        self.grouping: bool = False
        if isinstance(df, dd.DataFrame):
            try:
                if len(_all_features) > 0:
                    self.df: pd.DataFrame = pd.DataFrame(data=df[_all_features].compute(), columns=_all_features)
                else:
                    self.df: pd.DataFrame = pd.DataFrame(data=df.compute(), columns=list(df.columns))
            except MemoryError:
                raise DataVisualizerException('Data set is too big (cases={} | features={}) to visualize using Plot.ly (offline)'.format(len(df), len(df.columns)))
        elif isinstance(df, pd.DataFrame):
            self.df: pd.DataFrame = df
        if df is None:
            self.df = df
        else:
            if len(df) == 0 or len(df.columns) == 0:
                raise DataVisualizerException('Data set is empty')
        if subplots is None and plot_type is None:
            raise DataVisualizerException('Neither plot type nor subplots found')
        if plot_type is not None:
            if plot_type not in plots:
                raise DataVisualizerException('Plot type "{}" not supported'.format(plot_type))
        if subplots is not None:
            for title, sub in subplots.items():
                if sub.get('plot_type') not in plots:
                    raise DataVisualizerException('Subplot type "{}" not supported'.format(sub))
        pd.options.display.max_rows = max_row if max_row > 0 else 50
        pd.options.display.max_columns = max_col if max_col > 0 else 20
        self.subplots: dict = subplots
        self.title: str = title
        if self.df is None:
            self.features: List[str] = features
            self.n_features: int = 0
            self.index: list = None
        else:
            self.features: List[str] = list(df.columns) if features is None else features
            self.n_features: int = len(self.features)
            self.index: list = list(self.df.index.values)
        self.time_features: List[str] = time_features
        self.graph_features: Dict[str, str] = graph_features
        self.group_by: List[str] = group_by
        if feature_types is None:
            if self.df is None:
                if kwargs.get('df') is None and kwargs.get('data') is None:
                    self.feature_types: Dict[str, List[str]] = {}
                else:
                    self.feature_types: Dict[str, List[str]] = EasyExploreUtils().get_feature_types(df=self.kwargs.get('df') if self.kwargs.get('data') is None else self.kwargs.get('data'),
                                                                                                    features=self.kwargs.get('features'),
                                                                                                    dtypes=list(self.kwargs.get('df')[self.kwargs.get('features')].dtypes) if self.kwargs.get('data') is None else list(self.kwargs.get('data')[self.kwargs.get('features')].dtypes),
                                                                                                    continuous=None,
                                                                                                    categorical=None,
                                                                                                    ordinal=None,
                                                                                                    date=None,
                                                                                                    id_text=None,
                                                                                                    date_edges=None,
                                                                                                    print_msg=False
                                                                                                    )
            else:
                self.feature_types: Dict[str, List[str]] = EasyExploreUtils().get_feature_types(df=self.df,
                                                                                                features=self.features,
                                                                                                dtypes=list(self.df[self.features].dtypes),
                                                                                                continuous=None,
                                                                                                categorical=None,
                                                                                                ordinal=None,
                                                                                                date=None,
                                                                                                id_text=None,
                                                                                                date_edges=None,
                                                                                                print_msg=False
                                                                                                )
        else:
            self.feature_types: Dict[str, List[str]] = feature_types
        self.plot: dict = {}
        self.plot_type: str = plot_type
        self.melt: bool = melt
        self.brushing: bool = brushing
        self.xaxis_label: List[str] = xaxis_label
        self.yaxis_label: List[str] = yaxis_label
        self.zaxis_label: List[str] = zaxis_label
        self.annotations: List[dict] = annotations
        self.layout = None
        self.rows: int = rows_sub
        self.cols: int = cols_sub
        self.width: int = width
        self.height: int = height
        self.unit: str = unit
        self.pair: List[tuple] = [()]
        self.ax = None
        self.fig = None
        self.render: bool = render
        self.plt_fig_size: dict = {'size': tuple([20, 3]), 'dpi': 100}
        self.color_scale: List[str] = color_scale
        self.color_edges: List[str] = color_edges
        self.color_feature: str = color_feature
        self.color_type: str = ''
        self.table_color: dict = dict(line='#7D7F80', fill='#a1c3d1')
        self.path: str = ''
        self.file_path: str = file_path
        self.file_path_extension: str = ''
        self.max_str_length: int = 40
        self.seed: int = 1234
        if self.file_path is not None:
            if len(self.file_path) > 0:
                if self.file_path.replace('\\', '/').find('/') >= 0:
                    FileUtils(file_path=file_path).make_dir()
            else:
                Log(write=False).log('Invalid file path ({})'.format(self.file_path))
                self.file_path = None
        self.kwargs: dict = kwargs

    def _config_plotly_offline(self):
        """
        Config plotly offline visualization options
        """
        init_notebook_mode(connected=False)
        if self.subplots is None:
            self.subplots = dict()
            if self.features is None and self.plot_type not in ['heat']:
                raise DataVisualizerException('No features found')
            if self.plot_type in ['geo', 'heat']:
                _subplot_type: str = 'domain'
            elif self.plot_type in ['hist', 'line', 'scatter']:
                _subplot_type: str = 'xy'
            elif self.plot_type in ['radar']:
                _subplot_type: str = 'polar'
            elif self.plot_type in ['mesh3d', 'scatter3d']:
                _subplot_type: str = 'scene'
            else:
                _subplot_type: str = self.plot_type
            self.subplots.update({self.title: {'data': self.df,
                                               'features': self.features,
                                               'plot_type': self.plot_type,
                                               'type': _subplot_type,
                                               'melt': self.melt,
                                               'group_by': self.group_by,
                                               'time_features': self.time_features,
                                               'graph_features': self.graph_features,
                                               'brushing': self.brushing,
                                               'color_scale': self.color_scale,
                                               'color_edges': self.color_edges,
                                               'color_feature': self.color_feature,
                                               'xaxis_label': self.xaxis_label,
                                               'yaxis_label': self.yaxis_label,
                                               'annotations': self.annotations,
                                               'file_path': self.file_path,
                                               'kwargs': {} if self.kwargs is None or not isinstance(self.kwargs, dict) else self.kwargs
                                               }
                                  })
            if self.subplots[self.title].get('color_feature') is not None:
                if not self.subplots[self.title].get('color_feature') in self.subplots[self.title].get('data'):
                    self.subplots[self.title].update({'color_feature': self.color_feature})
            if self.subplots[self.title].get('color_scale') is not None:
                if len(self.subplots[self.title].get('color_scale')) > 0:
                    _hex: int = 0
                    _color_scale: List[str] = []
                    for cs in self.subplots[self.title].get('color_scale'):
                        if all(c in string.hexdigits for c in cs):
                            if len(_color_scale) is _hex:
                                _hex += 1
                                _color_scale.append(cs)
                            else:
                                self.subplots[self.title].update({'color_scale': None})
                                break
                        else:
                            _color: List[str] = cs.split(',')
                            if len(_color) == 3 or len(_color) == 4:
                                _rgb: str = ', '.join([re.findall(r'\d+', rgb)[0] for rgb in _color])
                                if len(_color) == 3:
                                    _color_scale.append('rgb({})'.format(_rgb))
                                else:
                                    _color_scale.append('rgba({})'.format(_rgb))
                            else:
                                self.subplots[self.title].update({'color_scale': None})
                                break
                else:
                    self.subplots.update({'color_scale': None})
            if self.subplots[self.title].get('color_edges') is not None:
                if all(c in string.hexdigits for c in self.subplots[self.title].get('color_edges')[0]):
                    self.color_type = 'hexa'
                else:
                    _color_edge: List[str] = self.subplots[self.title].get('color_edges')[0].split(',')
                    if len(_color_edge) == 3 or len(_color_edge) == 4:
                        _rgb: str = ', '.join([re.findall(r'\d+', rgb)[0] for rgb in _color_edge])
                        if len(_color_edge) == 3:
                            self.color_type = 'rgb'
                            self.subplots[self.title]['color_edges'][0] = 'rgb({})'.format(_rgb)
                        else:
                            self.color_type = 'rgba'
                            self.subplots[self.title]['color_edges'][0] = 'rgba({})'.format(_rgb)
                    else:
                        self.subplots.update({'color_edges': self.color_edges})
                if all(c in string.hexdigits for c in self.subplots[self.title].get('color_edges')[1]):
                    self.color_type = 'hexa'
                else:
                    _color_edge: List[str] = self.subplots[self.title].get('color_edges')[1].split(',')
                    if len(_color_edge) == 3 or len(_color_edge) == 4:
                        _rgb: str = ', '.join([re.findall(r'\d+', rgb)[0] for rgb in _color_edge])
                        if len(_color_edge) == 3:
                            self.color_type = 'rgb'
                            self.subplots['color_edges'][self.title][1] = 'rgb({})'.format(_rgb)
                        else:
                            self.color_type = 'rgba'
                            self.subplots['color_edges'][self.title][1] = 'rgba({})'.format(_rgb)
                    else:
                        self.subplots.update({'color_edges': self.color_edges})
            if self.subplots[self.title]['kwargs'].get('layout') is None:
                self.subplots[self.title]['kwargs'].update({'layout': {}})
        else:
            if isinstance(self.subplots, dict):
                for plot in self.subplots.keys():
                    if isinstance(self.subplots.get(plot), dict):
                        if 'data' in self.subplots.get(plot).keys():
                            if not isinstance(self.subplots[plot].get('data'), pd.DataFrame):
                                if isinstance(self.subplots[plot].get('data'), dd.DataFrame):
                                    self.subplots[plot].update({'data': pd.DataFrame(data=self.subplots[plot].get('data').values,
                                                                                     columns=self.subplots[plot].get('data').columns
                                                                                     )
                                                                })
                                else:
                                    self.subplots[plot].update({'data': None})
                        else:
                            if self.df is None:
                                self.subplots[plot].update({'data': None})
                            else:
                                self.subplots[plot].update({'data': self.df})
                        if 'df' in self.subplots.get(plot).keys():
                            if not isinstance(self.subplots[plot].get('df'), pd.DataFrame):
                                if isinstance(self.subplots[plot].get('data'), dd.DataFrame):
                                    self.subplots[plot].update(
                                        {'data': pd.DataFrame(data=self.subplots[plot].get('data').values.compute(),
                                                              columns=self.subplots[plot].get('data').columns
                                                              )
                                         })
                                else:
                                    self.subplots[plot].update({'data': None})
                                if 'data' in self.subplots.get(plot).keys():
                                    if not isinstance(self.subplots[plot].get('data'), pd.DataFrame):
                                        if isinstance(self.subplots[plot].get('data'), dd.DataFrame):
                                            self.subplots[plot].update({'data': pd.DataFrame(
                                                data=self.subplots[plot].get('data').values.compute(),
                                                columns=self.subplots[plot].get('data').columns
                                                )
                                                                        })
                                        else:
                                            self.subplots[plot].update({'data': None})
                                else:
                                    self.subplots[plot].update({'data': None})
                            else:
                                if 'data' not in self.subplots.get(plot).keys():
                                    self.subplots[plot].update({'data': self.subplots[plot].get('df')})
                                    self.subplots[plot].update({'df': None})
                                else:
                                    if not isinstance(self.subplots[plot].get('data'), pd.DataFrame):
                                        self.subplots[plot].update({'data': self.subplots[plot].get('df')})
                                        self.subplots[plot].update({'df': None})
                        else:
                            if self.df is None:
                                if 'data' in self.subplots.get(plot).keys():
                                    if not isinstance(self.subplots[plot].get('data'), pd.DataFrame):
                                        self.subplots[plot].update({'data': None})
                            else:
                                if 'data' in self.subplots.get(plot).keys():
                                    if not isinstance(self.subplots[plot].get('data'), pd.DataFrame):
                                        self.subplots[plot].update({'data': self.df})
                        if 'features' in self.subplots.get(plot).keys():
                            if isinstance(self.subplots[plot].get('features'), str):
                                self.subplots[plot].update({'features': [self.subplots[plot]['features']]})
                            else:
                                if not isinstance(self.subplots[plot].get('features'), list):
                                    if self.features is None:
                                        self.subplots[plot].update({'features': None})
                                        #raise DataVisualizerException('No features found in subplot config ({})'.format(plot))
                                    else:
                                        if len(self.features) == 0:
                                            self.subplots[plot].update({'features': None})
                                else:
                                    if len(self.subplots[plot].get('features')) == 0:
                                        self.subplots[plot].update({'features': None})
                        else:
                            if self.features is None:
                                self.subplots[plot].update({'features': None})
                                #raise DataVisualizerException('No data set found')
                            else:
                                if len(self.features) == 0:
                                    self.subplots[plot].update({'features': None})
                        if 'plot_type' in self.subplots.get(plot).keys():
                            if not isinstance(self.subplots[plot].get('plot_type'), str):
                                if self.plot_type is None:
                                    raise DataVisualizerException('No plot type found in subplot config ({})'.format(plot))
                                else:
                                    self.subplots[plot].update({'plot_type': self.plot_type})
                        else:
                            if self.plot_type is None:
                                raise DataVisualizerException('No plot type found in subplot config ({})'.format(plot))
                            else:
                                self.subplots[plot].update({'plot_type': self.plot_type})
                        if 'melt' in self.subplots.get(plot).keys():
                            if not isinstance(self.subplots[plot].get('melt'), bool):
                                self.subplots[plot].update({'melt': self.melt})
                        else:
                            self.subplots[plot].update({'melt': self.melt})
                        if 'group_by' in self.subplots.get(plot).keys():
                            if isinstance(self.subplots[plot].get('group_by'), str):
                                self.subplots[plot].update({'group_by': [self.subplots[plot]['group_by']]})
                            else:
                                if not isinstance(self.subplots[plot].get('group_by'), list):
                                    if self.group_by is None:
                                        raise DataVisualizerException('No features for grouping found in subplot config ({})'.format(plot))
                                    else:
                                        self.subplots[plot].update({'group_by': [self.subplots[plot]['group_by']]})
                                else:
                                    if len(self.subplots[plot].get('group_by')) == 0:
                                        raise DataVisualizerException('No features for grouping found in subplot config ({})'.format(plot))
                        else:
                            self.subplots[plot].update({'group_by': self.group_by})
                        if 'time_features' in self.subplots.get(plot).keys():
                            if isinstance(self.subplots[plot].get('time_features'), str):
                                self.subplots[plot].update({'time_features': [self.subplots[plot]['time_features']]})
                            else:
                                if not isinstance(self.subplots[plot].get('time_features'), list):
                                    if self.time_features is None:
                                        raise DataVisualizerException('No time features found in subplot config ({})'.format(plot))
                                    else:
                                        self.subplots[plot].update({'time_features': [self.subplots[plot]['time_features']]})
                                else:
                                    if len(self.subplots[plot].get('time_features')) == 0:
                                        raise DataVisualizerException('No time features found in subplot config ({})'.format(plot))
                        else:
                            self.subplots[plot].update({'time_features': self.time_features})
                        if 'graph_features' in self.subplots.get(plot).keys():
                            if not isinstance(self.subplots[plot].get('graph_features'), dict):
                                self.subplots[plot].update({'graph_features': self.graph_features})
                            else:
                                if self.subplots[plot]['graph_features'].get('node'):
                                    if self.subplots[plot]['graph_features'].get('node') is None:
                                        raise DataVisualizerException('No node feature found in subplot config ({})'.format(plot))
                                    else:
                                        if len(self.subplots[plot]['graph_features'].get('node')) == 0:
                                            raise DataVisualizerException('No node feature found in subplot config ({})'.format(plot))
                                if self.subplots[plot]['graph_features'].get('edge'):
                                    if self.subplots[plot]['graph_features'].get('edge') is None:
                                        raise DataVisualizerException('No edge feature found in subplot config ({})'.format(plot))
                                    else:
                                        if len(self.subplots[plot]['graph_features'].get('edge')) == 0:
                                            raise DataVisualizerException('No edge feature found in subplot config ({})'.format(plot))
                        else:
                            self.subplots[plot].update({'graph_features': self.graph_features})
                        self.feature_types: Dict[str, List[str]] = EasyExploreUtils().get_feature_types(df=self.subplots[plot]['data'],
                                                                                                        features=list(self.subplots[plot]['data'].columns),
                                                                                                        dtypes=list(self.subplots[plot]['data'].dtypes),
                                                                                                        continuous=None,
                                                                                                        categorical=None,
                                                                                                        ordinal=None,
                                                                                                        date=None,
                                                                                                        id_text=None,
                                                                                                        date_edges=None,
                                                                                                        print_msg=False
                                                                                                        )
                        if 'brushing' in self.subplots.get(plot).keys():
                            if not isinstance(self.subplots[plot].get('brushing'), bool):
                                self.subplots[plot].update({'brushing': self.brushing})
                        else:
                            self.subplots[plot].update({'brushing': self.brushing})
                        if 'color_scale' in self.subplots.get(plot).keys():
                            if isinstance(self.subplots[plot].get('color_scale'), list):
                                if len(self.subplots[plot].get('color_scale')) > 0:
                                    _hex: int = 0
                                    _color_scale: List[str] = []
                                    for cs in self.subplots[plot].get('color_scale'):
                                        if all(c in string.hexdigits for c in cs):
                                            if len(_color_scale) is _hex:
                                                _hex += 1
                                                _color_scale.append(cs)
                                            else:
                                                self.subplots[plot].update({'color_scale': self.color_scale})
                                                break
                                        else:
                                            _color: List[str] = cs.split(',')
                                            if len(_color) == 3 or len(_color) == 4:
                                                _rgb: str = ', '.join([re.findall(r'\d+', rgb)[0] for rgb in _color])
                                                if len(_color) == 3:
                                                    _color_scale.append('rgb({})'.format(_rgb))
                                                else:
                                                    _color_scale.append('rgba({})'.format(_rgb))
                                            else:
                                                self.subplots[plot].update({'color_scale': self.color_scale})
                                                break
                                else:
                                    self.subplots[plot].update({'color_scale': self.color_scale})
                            else:
                                self.subplots[plot].update({'color_scale': self.color_scale})
                        else:
                            self.subplots[plot].update({'color_scale': self.color_scale})
                        if 'color_edges' in self.subplots.get(plot).keys():
                            if isinstance(self.subplots[plot].get('color_edges'), tuple):
                                if isinstance(self.subplots[plot].get('color_edges')[0], str):
                                    if all(c in string.hexdigits for c in self.subplots[plot].get('color_edges')[0]):
                                        self.color_type = 'hexa'
                                    else:
                                        _color_edge: List[str] = self.subplots[plot].get('color_edges')[0].split(',')
                                        if len(_color_edge) == 3 or len(_color_edge) == 4:
                                            _rgb: str = ', '.join([re.findall(r'\d+', rgb)[0] for rgb in _color_edge])
                                            if len(_color_edge) == 3:
                                                self.color_type = 'rgb'
                                                self.subplots[plot]['color_edges'][0] = 'rgb({})'.format(_rgb)
                                            else:
                                                self.color_type = 'rgba'
                                                self.subplots[plot]['color_edges'][0] = 'rgba({})'.format(_rgb)
                                        else:
                                            self.subplots[plot].update({'color_edges': self.color_edges})
                                else:
                                    self.subplots[plot].update({'color_edges': self.color_edges})
                                if isinstance(self.subplots[plot].get('color_edges')[1], str):
                                    if all(c in string.hexdigits for c in self.subplots[plot].get('color_edges')[1]):
                                        self.color_type = 'hexa'
                                    else:
                                        _color_edge: List[str] = self.subplots[plot].get('color_edges')[1].split(',')
                                        if len(_color_edge) == 3 or len(_color_edge) == 4:
                                            _rgb: str = ', '.join([re.findall(r'\d+', rgb)[0] for rgb in _color_edge])
                                            if len(_color_edge) == 3:
                                                self.color_type = 'rgb'
                                                self.subplots[plot]['color_edges'][1] = 'rgb({})'.format(_rgb)
                                            else:
                                                self.color_type = 'rgba'
                                                self.subplots[plot]['color_edges'][1] = 'rgba({})'.format(_rgb)
                                        else:
                                            self.subplots[plot].update({'color_edges': self.color_edges})
                                else:
                                    self.subplots[plot].update({'color_edges': self.color_edges})
                            else:
                                self.subplots[plot].update({'color_edges': self.color_edges})
                        else:
                            self.subplots[plot].update({'color_edges': self.color_edges})
                        if 'color_feature' in self.subplots.get(plot).keys():
                            if isinstance(self.subplots[plot].get('color_feature'), str):
                                if not self.subplots[plot].get('color_feature') in self.subplots[plot].get('data').keys():
                                    self.subplots[plot].update({'color_feature': None})
                            else:
                                self.subplots[plot].update({'color_feature': self.color_feature})
                        else:
                            self.subplots[plot].update({'color_feature': self.color_feature})
                        if 'xaxis_label' in self.subplots.get(plot).keys():
                            if isinstance(self.subplots[plot].get('xaxis_label'), str):
                                self.subplots[plot].update({'xaxis_label': [self.subplots[plot]['xaxis_label']]})
                            else:
                                if not isinstance(self.subplots[plot].get('xaxis_label'), list):
                                    self.subplots[plot].update({'xaxis_label': [self.subplots[plot]['xaxis_label']]})
                                else:
                                    if len(self.subplots[plot].get('xaxis_label')) == 0:
                                        self.subplots[plot].update({'xaxis_label': self.xaxis_label})
                        else:
                            self.subplots[plot].update({'xaxis_label': self.xaxis_label})
                        if 'yaxis_label' in self.subplots.get(plot).keys():
                            if isinstance(self.subplots[plot].get('yaxis_label'), str):
                                self.subplots[plot].update({'yaxis_label': [self.subplots[plot]['yaxis_label']]})
                            else:
                                if not isinstance(self.subplots[plot].get('yaxis_label'), list):
                                    self.subplots[plot].update({'yaxis_label': [self.subplots[plot]['yaxis_label']]})
                                else:
                                    if len(self.subplots[plot].get('yaxis_label')) == 0:
                                        self.subplots[plot].update({'yaxis_label': self.yaxis_label})
                        else:
                            self.subplots[plot].update({'yaxis_label': self.yaxis_label})
                        if 'annotations' in self.subplots.get(plot).keys():
                            if not isinstance(self.subplots[plot].get('annotations'), dict):
                                self.subplots[plot].update({'annotations': self.annotations})
                        else:
                            self.subplots[plot].update({'annotations': self.annotations})
                        if 'file_path' in self.subplots.get(plot).keys():
                            if not isinstance(self.subplots[plot].get('file_path'), str):
                                self.subplots[plot].update({'file_path': self.file_path})
                        else:
                            self.subplots[plot].update({'file_path': self.file_path})
                        if 'kwargs' in self.subplots.get(plot).keys():
                            if isinstance(self.subplots[plot]['kwargs'], dict):
                                if 'layout' in self.subplots[plot].get('kwargs'):
                                    if not isinstance(self.subplots[plot]['kwargs'].get('layout'), dict):
                                        self.subplots[plot]['kwargs'].update({'layout': {}})
                                else:
                                    self.subplots[plot]['kwargs'].update({'layout': {}})
                                    #self.subplots[plot].update({'kwargs': dict(layout={})})
                            else:
                                self.subplots[plot]['kwargs'].update({'layout': {}})
                        else:
                            self.subplots[plot].update({'kwargs': dict(layout={})})
                    else:
                        raise DataVisualizerException('Subplots dictionary should contain dictionaries'.format(self.subplots))
            else:
                raise DataVisualizerException('Subplots parameter should be a dictionary containing dictionaries -> Dict[str, dict]'.format(self.subplots))

    def _hierarchical_data_set(self, value_feature: str, color_features: List[str] = None) -> pd.DataFrame:
        """
        Restructure data set hierarchically

        :param value_feature: str
            Feature name

        :param color_features: List[str]
            Names of the features used for coloring

        :return: pd.DataFrame
            Hierarchical data set
        """
        _all_trees: dict = {'id': [], 'parent': [], value_feature: []}
        for i, level in enumerate(self.group_by):
            _tree = pd.DataFrame(columns=['id', 'parent', value_feature])
            _group = self.df.groupby(self.group_by[i:]).sum(numerical_only=True).reset_index()
            _tree['id'] = _group[level]
            if i < len(self.group_by) - 1:
                _tree['parent'] = _group[self.group_by[i + 1]]
            else:
                _tree['parent'] = 'total'
            _tree[value_feature] = _group[value_feature]
            #_tree['color'] = _group[value_column]
            for key in _all_trees.keys():
                _all_trees[key] = _all_trees[key] + _tree[key].values.tolist()
        _all_trees['id'] = _all_trees['id'] + ['total']
        _all_trees['parent'] = _all_trees['parent'] + ['']
        _all_trees[value_feature] = _all_trees[value_feature] + [self.df[value_feature].sum()]
        #_all_trees['color'] = _all_trees['color'] + [df[value_column].sum()]
        return pd.DataFrame(_all_trees)

    def _run_plotly_offline(self):
        """
        Run visualization using plotly offline
        """
        for t, title in enumerate(self.subplots.keys()):
            _data: List[go] = []
            self.title = title
            self.plot = self.subplots.get(title)
            self.df = self.plot.get('data')
            if self.plot.get('group_by') is not None and len(self.plot.get('group_by')) > 0:
                self.grouping = True
            if self.plot.get('xaxis_label') is not None:
                self.plot['kwargs']['layout'].update({'xaxis': dict(title=dict(text=self.plot.get('xaxis_label')[t]))})
            if self.plot.get('yaxis_label') is not None:
                self.plot['kwargs']['layout'].update({'yaxis': dict(title=dict(text=self.plot.get('yaxis_label')[t]))})
            if self.plot.get('color_feature') is not None:
                if str(self.df[self.plot.get('color_feature')].dtype).find('object') >= 0:
                    _color_feature = LabelEncoder().fit_transform(y=self.df[self.plot.get('color_feature')])
                    _color_bar = dict(tickvals=np.unique(_color_feature).tolist(),
                                      ticktext=self.df[self.plot.get('color_feature')].unique().tolist()
                                      )
                else:
                    _color_bar = None
                    _color_feature = self.df[self.plot.get('color_feature')].values
            else:
                _color_bar = None
                _color_feature = None
            if self.plot.get('color_scale') is not None:
                _color_scale = self.plot.get('color_scale')
            else:
                if self.plot.get('color_range') is not None:
                    _color_scale = n_colors(lowcolor=self.plot.get('color_range')[0],
                                            highcolor=self.plot.get('color_range')[1],
                                            n_colors=10,
                                            colortype=self.color_type
                                            )
                else:
                    _color_scale = self.plot['kwargs'].get('colorscale')
            ################
            # Table Chart: #
            ################
            if self.plot.get('plot_type') == 'table':
                self.df.insert(loc=0,
                               column='-' if self.plot['kwargs'].get('index_title') is None else self.plot[
                                   'kwargs'].get('index_title'),
                               value=self.df.index.values.tolist(),
                               allow_duplicates=False
                               )
                _header_align: List[str] = ['left' if col == 0 else 'center' for col in range(0, self.df.shape[1], 1)]
                _cells_align: List[str] = ['left' if cell == 0 else 'center' for cell in range(0, self.df.shape[1], 1)]
                _cells_fill: List[str] = ['cornflowerblue' if col == 0 else 'white' for col in
                                          range(0, self.df.shape[1], 1)]
                self.plot['kwargs'].update({'header': dict(
                    values=self.df.columns if self.plot['kwargs'].get('header_values') is None else self.plot[
                        'kwargs'].get('header_values'),
                    line=dict(color='darkslategray') if self.plot['kwargs'].get('header_line') is None else self.plot[
                        'kwargs'].get('header_line'),
                    fill=dict(color='darkslateblue') if self.plot['kwargs'].get('header_fill') is None else self.plot[
                        'kwargs'].get('header_fill'),
                    align=_header_align if self.plot['kwargs'].get('header_align') is None else self.plot['kwargs'].get(
                        'header_align'),
                    font=dict(color='white', size=12) if self.plot['kwargs'].get('header_font') is None else self.plot[
                        'kwargs'].get('header_font'),
                    height=40 if self.plot['kwargs'].get('header_height') is None else self.plot['kwargs'].get(
                        'header_height'),
                    ) if self.plot['kwargs'].get('header') is None else self.plot['kwargs'].get('header'),
                                            'cells': dict(
                                                values=self.df.transpose().to_numpy() if self.plot['kwargs'].get(
                                                    'cells_values') is None else self.plot['kwargs'].get(
                                                    'cells_values'),
                                                line=dict(color='darkslategray') if self.plot['kwargs'].get(
                                                    'cells_line') is None else self.plot['kwargs'].get('cells_line'),
                                                fill=dict(color=_cells_fill) if self.plot['kwargs'].get(
                                                    'cells_fill') is None else self.plot['kwargs'].get('cells_fill'),
                                                align=_cells_align if self.plot['kwargs'].get(
                                                    'cells_align') is None else self.plot['kwargs'].get('cells_align'),
                                                font=dict(color='black', size=12) if self.plot['kwargs'].get(
                                                    'cells_font') is None else self.plot['kwargs'].get('cells_font'),
                                                height=30 if self.plot['kwargs'].get('cells_height') is None else
                                                self.plot['kwargs'].get('cells_height'),
                                                ) if self.plot['kwargs'].get('cells') is None else self.plot[
                                                'kwargs'].get('cells'),
                                            'columnorder': [i for i in range(0, self.df.shape[1], 1)] if self.plot[
                                                                                                             'kwargs'].get(
                                                'columnorder') is None else self.plot['kwargs'].get('columnorder'),
                                            'hoverinfo': 'text' if self.plot['kwargs'].get('hoverinfo') is None else
                                            self.plot['kwargs'].get('hoverinfo'),
                                            'visible': True if self.plot['kwargs'].get('visible') is None else
                                            self.plot['kwargs'].get('visible')
                                            })
                self.fig = PlotlyAdapter(plot=self.plot, offline=True).table()
                self._show_plotly_offline()
            #################
            # Funnel Chart: #
            #################
            elif self.plot.get('plot_type') in ['funnel', 'funnel_area']:
                if self.plot.get('group_by') is None:
                    pass
                else:
                    pass
            #############################
            # Sunburst & Treemap Chart: #
            #############################
            elif self.plot.get('plot_type') in ['sunburst', 'tree']:
                self.features = []
                self.group_by = [] if self.plot.get('group_by') is None else self.plot.get('group_by')
                for ft in self.plot.get('features'):
                    if ft in self.feature_types.get('continuous'):
                        self.features.append(ft)
                    else:
                        self.group_by.append(ft)
                self.group_by = list(set(self.group_by))
                for conti in self.features:
                    _hierarchical_df: pd.DataFrame = self._hierarchical_data_set(value_feature=conti,
                                                                                 color_features=[conti]
                                                                                 )
                    _elements = _hierarchical_df[['id', 'parent']].transpose().apply(
                        lambda x: str(x[0]) + '_' + str(x[1]))
                    if len(_elements) >= 100:
                        if self.plot.get('plot_type') == 'sunburst':
                            self.plot['kwargs'].update({'labels': _hierarchical_df['id'].values,
                                                        'parents': _hierarchical_df['parent'].values,
                                                        'values': _hierarchical_df[conti].values,
                                                        'branchvalues': 'total',
                                                        'marker': dict(colors=_hierarchical_df[conti].values,
                                                                       colorscale='RdBu',
                                                                       ),
                                                        'hovertemplate': '<b>%{label} </b> <br> %{conti}: %{value}<br> value: %{color:.2f}',
                                                        'name': 'Non-Factorial Breakdown'
                                                        })
                            self.fig = PlotlyAdapter(plot=self.plot, offline=True).sunburst()
                        elif self.plot.get('plot_type') == 'tree':
                            self.plot['kwargs'].update({'labels': _hierarchical_df['id'].values,
                                                        'parents': _hierarchical_df['parent'].values,
                                                        'values': _hierarchical_df[conti].values,
                                                        'branchvalues': 'total',
                                                        'marker': dict(colors=_hierarchical_df[conti].values,
                                                                       colorscale='RdBu',
                                                                       ),
                                                        'hovertemplate': '<b>%{label} </b> <br> %{conti}: %{value}<br> value: %{color:.2f}',
                                                        'name': 'Non-Factorial Breakdown'
                                                        })
                            self.fig = PlotlyAdapter(plot=self.plot, offline=True).treemap()
                        self.plot['kwargs']['layout'].update({'margin': dict(
                            t=10 if self.plot['kwargs']['layout'].get('t') is None else self.plot['kwargs'][
                                'layout'].get('t'),
                            b=10 if self.plot['kwargs']['layout'].get('b') is None else self.plot['kwargs'][
                                'layout'].get('b'),
                            r=10 if self.plot['kwargs']['layout'].get('r') is None else self.plot['kwargs'][
                                'layout'].get('r'),
                            l=10 if self.plot['kwargs']['layout'].get('l') is None else self.plot['kwargs'][
                                'layout'].get('l')
                            )
                                                              })
                        self._show_plotly_offline()
                    else:
                        Log(write=False, level='warn').log('Parents and ids are not unique')
                        _pairs: List[tuple] = EasyExploreUtils().get_pairs(features=self.group_by, max_features_each_pair=2)
                        for pair in _pairs:
                            self.df = self.df[[pair[0], pair[1]]]
                            _hierarchical_df = self._hierarchical_data_set(value_feature=conti,
                                                                           color_features=[conti]
                                                                           )
                            if self.plot.get('plot_type') == 'sunburst':
                                self.plot['kwargs'].update({'labels': _hierarchical_df['id'].values,
                                                            'parents': _hierarchical_df['parent'].values,
                                                            'values': _hierarchical_df[conti].values,
                                                            'branchvalues': 'total',
                                                            'marker': dict(
                                                                # colors=_hierarchical_df['color'].values.tolist(),
                                                                colorscale='RdBu',
                                                                # cmid=''
                                                            ),
                                                            'hovertemplate': '<b>%{label} </b> <br> %{conti}: %{value}<br> value: %{color:.2f}',
                                                            'name': self._trim(
                                                                input_str='{} x {}'.format(pair[0], pair[1]))
                                                            })
                                self.fig = PlotlyAdapter(plot=self.plot, offline=True).sunburst()
                            elif self.plot.get('plot_type') == 'tree':
                                self.plot['kwargs'].update({'labels': _hierarchical_df['id'].values,
                                                            'parents': _hierarchical_df['parent'].values,
                                                            'values': _hierarchical_df[conti].values,
                                                            'branchvalues': 'total',
                                                            'marker': dict(  # colors=_hierarchical_df['color'],
                                                                colorscale='RdBu',
                                                                # cmid=''
                                                            ),
                                                            'hovertemplate': '<b>%{label} </b> <br> %{conti}: %{value}<br> value: %{color:.2f}',
                                                            'name': 'Non-Factorial Breakdown'
                                                            })
                                self.fig = PlotlyAdapter(plot=self.plot, offline=True).treemap()
                            self.plot['kwargs']['layout'].update({'margin': dict(
                                t=10 if self.plot['kwargs']['layout'].get('t') is None else self.plot['kwargs'][
                                    'layout'].get('t'),
                                b=10 if self.plot['kwargs']['layout'].get('b') is None else self.plot['kwargs'][
                                    'layout'].get('b'),
                                r=10 if self.plot['kwargs']['layout'].get('r') is None else self.plot['kwargs'][
                                    'layout'].get('r'),
                                l=10 if self.plot['kwargs']['layout'].get('l') is None else self.plot['kwargs'][
                                    'layout'].get('l')
                            )
                            })
                            self._show_plotly_offline()
            #####################
            # Dendrogram Chart: #
            #####################
            elif self.plot.get('plot_type') == 'dendro':
                self.df = self.df[self.plot.get('features')]
                if len(self.plot.get('features')) > 0:
                    self.df = self.df[self.plot.get('features')]
                self.plot['kwargs'].update({'X': np.transpose(self.df.values),
                                            'orientation': 'bottom' if self.plot['kwargs'].get(
                                                'orientation') is None else self.plot['kwargs'].get('orientation'),
                                            'labels': self.df.columns if self.plot['kwargs'].get(
                                                'labels') is None else self.plot['kwargs'].get('labels'),
                                            'colorscale': _color_scale,
                                            'linkagefun': linkage,
                                            # 'hovertext': self.df.columns if self.plot['kwargs'].get('hovertext') is None else self.plot['kwargs'].get('hovertext'),
                                            'color_threshold': 1.5 if self.plot['kwargs'].get(
                                                'color_threshold') is None else self.plot['kwargs'].get(
                                                'color_threshold')
                                            })
                self.fig = PlotlyAdapter(plot=self.plot, offline=True).dendrogram()
                self._show_plotly_offline()
            ####################
            # Silhouette Chart #
            ####################
            elif self.plot.get('plot_type') == 'silhouette':
                if self.plot['kwargs'].get('silhouette') is None:
                    raise DataVisualizerException('No results of silhouette analysis found')
                if isinstance(self.plot['kwargs'].get('silhouette'), dict):
                    _silhouette: dict = self.plot['kwargs'].get('silhouette')
                    for pl in range(2, self.plot['kwargs'].get('n_clusters') + 1, 1):
                        for cl in range(0, pl, 1):
                            self.plot['kwargs'].update({'x': _silhouette['cluster_{}_samples'.format(cl)].get('scores'),
                                                        'y': _silhouette['cluster_{}_samples'.format(cl)].get('y'),
                                                        'mode': 'lines',
                                                        'line': dict(width=0.5),
                                                        'fill': 'tozerox',
                                                        'name': 'Cluster {}'.format(cl + 1),
                                                        'xaxis': 'x',
                                                        'yaxis': 'y',
                                                        'showlegend': True
                                                        })
                            _data.append(PlotlyAdapter(plot=self.plot, offline=True).scatter_gl())
                        self.plot['kwargs']['layout'].update({'xaxis': dict(anchor='y',
                                                                            domain=[0, 1],
                                                                            showticklabels=True
                                                                            ),
                                                              'yaxis': dict(anchor='x',
                                                                            domain=[0, 1],
                                                                            showticklabels=True
                                                                            )
                                                              })
                        self.fig = go.Figure(_data)
                        self._show_plotly_offline()
                        _data = []
                else:
                    raise DataVisualizerException('Results of silhouette analysis should be a dictionary')
            ##################
            # Contour Chart: #
            ##################
            elif self.plot.get('plot_type') == 'contour':
                if len(self.plot.get('features')) > 0:
                    self.df = self.df[self.plot.get('features')]
                self.plot['kwargs'].update(
                    {'z': self.df.values if self.plot['kwargs'].get('z') is None else self.plot['kwargs'].get('z'),
                     'x': self.df.values if self.plot['kwargs'].get('x') is None else self.plot['kwargs'].get('x'),
                     'y': self.df.values if self.plot['kwargs'].get('y') is None else self.plot['kwargs'].get('y'),
                     'colorscale': 'Hot' if _color_scale is None else _color_scale
                     })
                self.fig = PlotlyAdapter(plot=self.plot, offline=True).contour()
                self._show_plotly_offline()
            ###############################
            # 2D Histogram Contour Chart: #
            ###############################
            elif self.plot.get('plot_type') == 'contour_hist':
                _features: List[str] = self.plot.get('features')
                if len(_features) < 2:
                    raise DataVisualizerException(
                        'Not enough features ({}) for generating scatter plot'.format(len(_features)))
                _pairs: List[tuple] = EasyExploreUtils().get_pairs(features=_features, max_features_each_pair=2)
                if self.plot.get('group_by') is None:
                    for j, pair in enumerate(_pairs, start=1):
                        _fig: go.Figure = go.Figure()
                        self.file_path_extension = self._trim(input_str='{}_{}'.format(pair[0], pair[1]))
                        self.plot['kwargs'].update({'x': self.df[pair[0]].values,
                                                    'y': self.df[pair[1]].values,
                                                    'mode': 'markers' if self.plot['kwargs'].get('mode') is None else self.plot['kwargs'].get('mode'),
                                                    'name': self._trim(input_str='{} - {}'.format(pair[0], pair[1])),
                                                    'reversescale': True
                                                    })
                        if self.plot.get('xaxis_label') is None:
                            self.plot['kwargs']['layout'].update({'xaxis': dict(title=dict(text=pair[0]))})
                        if self.plot.get('yaxis_label') is None:
                            self.plot['kwargs']['layout'].update({'yaxis': dict(title=dict(text=pair[1]))})
                        _fig.add_trace(PlotlyAdapter(plot=self.plot, offline=True).histogram_2d_contour())
                    self.fig = _fig
                    self._show_plotly_offline()
                else:
                    for pair in _pairs:
                        _data: List[go] = []
                        for group in self.plot.get('group_by'):
                            _group_val: List[str] = self.df[group].unique().tolist()
                            for ft in self.plot.get('features'):
                                for ext, val in enumerate(_group_val):
                                    _fig: go.Figure = go.Figure()
                                    self.file_path_extension = self._trim(input_str='{}_{}_{}'.format(ft, group, ext))
                                    self.plot['kwargs'].update({'mode': 'markers' if self.plot['kwargs'].get(
                                        'mode') is None else self.plot['kwargs'].get('mode'),
                                                                'name': self._trim(input_str='{} ({}={})'.format(ft, group, val)),
                                                                'xaxis': 'x',
                                                                'yaxis': 'y',
                                                                'reversescale': True
                                                                })
                                    if val in INVALID_VALUES:
                                        self.plot['kwargs'].update(
                                            {'x': self.df.loc[self.df[group].isnull(), pair[0]].values,
                                             'y': self.df.loc[self.df[group].isnull(), pair[1]].values
                                             })
                                    else:
                                        self.plot['kwargs'].update(
                                            {'x': self.df.loc[self.df[group] == val, pair[0]].values,
                                             'y': self.df.loc[self.df[group] == val, pair[1]].values
                                             })
                                    _fig.add_trace(PlotlyAdapter(plot=self.plot, offline=True).histogram_2d_contour())
                                    if self.plot.get('xaxis_label') is None:
                                        self.plot['kwargs']['layout'].update({'xaxis': dict(title=dict(text=pair[0]))})
                                    if self.plot.get('yaxis_label') is None:
                                        self.plot['kwargs']['layout'].update({'yaxis': dict(title=dict(text=pair[1]))})
                                    self.plot['kwargs']['layout'].update({'autosize': False,
                                                                          'bargap': 0,
                                                                          'hovermode': 'closest',
                                                                          'showlegend': True
                                                                          })
                                    self.fig = _fig
                                    self._show_plotly_offline()
            ##################
            # Heatmap Chart: #
            ##################
            elif self.plot.get('plot_type') == 'heat':
                if self.plot.get('features') is not None:
                    if len(self.plot.get('features')) > 0:
                        self.df = self.df[self.plot.get('features')]
                if self.plot.get('group_by') is None:
                    self.plot['kwargs'].update(
                        {'z': self.df.values if self.plot['kwargs'].get('z') is None else self.plot['kwargs'].get('z'),
                         'x': self.plot.get('features') if self.plot['kwargs'].get('x') is None else self.plot['kwargs'].get('x'),
                         'y': self.df.index.values.tolist() if self.plot['kwargs'].get('y') is None else self.plot['kwargs'].get('y'),
                         'colorbar': self.plot['kwargs'].get('colorbar'),
                         'colorscale': _color_scale
                         })
                    if self.plot.get('annotation') is not None:
                        self.plot['kwargs'].update({'annotation_text': self.df.value.tolist() if self.plot.get(
                            'annotation') == '' else self.plot.get('annotation')})
                        self.fig = PlotlyAdapter(plot=self.plot, offline=True).heat_map_annotated()
                    else:
                        self.fig = PlotlyAdapter(plot=self.plot, offline=True).heat_map()
                    self._show_plotly_offline()
                else:
                    for group in self.plot.get('group_by'):
                        _group_val: List[str] = self.df[group].unique().tolist()
                        for ext, val in enumerate(_group_val):
                            self.file_path_extension = self._trim(input_str='{}_{}'.format(group, ext))
                            self.plot['kwargs'].update({'colorbar': self.plot['kwargs'].get('colorbar'),
                                                        'colorscale': _color_scale,
                                                        'name': self._trim(input_str='{}={}'.format(group, val))
                                                        })
                            if val in INVALID_VALUES:
                                self.df[group] = self.df[group].replace(INVALID_VALUES, np.nan)
                                self.plot['kwargs'].update({'x': self.plot.get('features'),
                                                            'y': self.df.index.values,
                                                            'z': self.df.loc[self.df[group].isnull(), :].values
                                                            })
                            else:
                                self.plot['kwargs'].update({'x': self.plot.get('features'),
                                                            'y': self.df.index.values,
                                                            'z': self.df.loc[self.df[group] == val, :].values
                                                            })
                            if self.plot.get('annotation') is not None:
                                self.plot['kwargs'].update({'annotation_text': self.df.value.tolist() if self.plot.get('annotation') == '' else self.plot.get('annotation')})
                                self.fig = PlotlyAdapter(plot=self.plot, offline=True).heat_map_annotated()
                            else:
                                self.fig = PlotlyAdapter(plot=self.plot, offline=True).heat_map()
                            self._show_plotly_offline()
            ################
            # Radar Chart: #
            ################
            elif self.plot.get('plot_type') == 'radar':
                self.plot['kwargs'].update(
                    {'fill': 'toself' if self.plot['kwargs'].get('fill') is None else self.plot['kwargs'].get('fill'),
                     'hoverinfo': 'all' if self.plot['kwargs'].get('hoverinfo') is None else self.plot['kwargs'].get(
                         'hoverinfo'),
                     'mode': 'lines' if self.plot['kwargs'].get('mode') is None else self.plot['kwargs'].get('mode'),
                     'layout': dict(polar=dict(radialaxis=dict(visible=True, type='-'))) if self.plot['kwargs'][
                                                                                                'layout'].get(
                         'polar') is None else self.plot['kwargs']['layout'].get('polar')
                     })
                _nums: List[str] = [] if self.plot['kwargs'].get('r') is None else self.plot['kwargs'].get('r')
                _cats: List[str] = [] if self.plot['kwargs'].get('theta') is None else self.plot['kwargs'].get('theta')
                _ordinal: List[str] = []
                if len(self.plot.get('features')) == 0 and len(_cats) == 0 and len(_nums) == 0:
                    raise DataVisualizerException('No feature found')
                for ft in self.plot.get('features'):
                    if ft in self.feature_types.get('continuous'):
                        _nums.append(ft)
                    else:
                        if ft in self.feature_types.get('categorical'):
                            if len(self.df.loc[~self.df[ft].isnull(), ft].unique()) <= 2:
                                _cats.append(ft)
                                self.df[ft] = self.df[ft].astype(bool)
                            else:
                                _ordinal.append(ft)
                        else:
                            _cats.append(ft)
                if len(_nums) == 0:
                    if len(_ordinal) == 0:
                        raise DataVisualizerException('No continuous or semi-continuous features found')
                    _nums = _nums + _ordinal
                else:
                    _cats = _cats + _ordinal
                for i, cat in enumerate(_cats, start=1):
                    if self.plot.get('color_edges') is not None:
                        _color_scale = n_colors(lowcolor=self.plot.get('color_edges')[0],
                                                highcolor=self.plot.get('color_edges')[1],
                                                n_colors=len(_nums),
                                                colortype=self.color_type)
                    for j, cont in enumerate(_nums, start=1):
                        if self.plot.get('group_by') is None:
                            self.plot['kwargs'].update({'r': self.df[cont].values,
                                                        'theta': self.df[cat].values,
                                                        'name': self._trim(input_str=cont),
                                                        'marker': dict(color=_color_scale[j]) if self.plot['kwargs'].get('marker') is None else self.plot['kwargs'].get('marker'),
                                                        })
                            if self.plot.get('melt'):
                                _data.append(PlotlyAdapter(plot=self.plot, offline=True).scatterpolar())
                                print('j', len(_nums))
                                if j == len(_nums):
                                    self.fig = _data
                                    self._show_plotly_offline()
                            else:
                                print('nothing')
                                self.fig = PlotlyAdapter(plot=self.plot, offline=True).scatterpolar()
                                self._show_plotly_offline()
                        else:
                            for k, group in enumerate(self.plot.get('group_by'), start=1):
                                if cat == group:
                                    if self.plot.get('melt'):
                                        if k == len(self.plot.get('group_by')) and j == len(_nums):
                                            self.fig = _data
                                            self._show_plotly_offline()
                                    else:
                                        if k == len(self.plot.get('group_by')):
                                            self.fig = _data
                                            self._show_plotly_offline()
                                            _data = []
                                else:
                                    _group_val: np.array = self.df[group].unique()
                                    for l, val in enumerate(_group_val, start=1):
                                        self.plot['kwargs'].update({'theta': self.df[cat].values,
                                                                    'name': self._trim(
                                                                        input_str='{} ({}={})'.format(cont, group,
                                                                                                      val)),
                                                                    'marker': dict(color=_color_scale[j]) if self.plot[
                                                                                                                 'kwargs'].get(
                                                                        'marker') is None else self.plot['kwargs'].get(
                                                                        'marker'),
                                                                    })
                                        if val in INVALID_VALUES:
                                            self.plot['kwargs'].update({'r': self.df.loc[self.df[group].isnull(), cont].values})
                                        else:
                                            self.plot['kwargs'].update({'r': self.df.loc[self.df[group] == val, cont].values})
                                        _data.append(PlotlyAdapter(plot=self.plot, offline=True).scatterpolar())
                                    if self.plot.get('melt'):
                                        print('k', k)
                                        print('group', len(self.plot.get('group_by')))
                                        print('j', j)
                                        print('nums', len(_nums))
                                        if k == len(self.plot.get('group_by')) and j == len(_nums):
                                            self.fig = _data
                                            self._show_plotly_offline()
                                    else:
                                        print('k', k)
                                        print('group', len(self.plot.get('group_by')))
                                        if k == len(self.plot.get('group_by')):
                                            self.fig = _data
                                            self._show_plotly_offline()
                                            _data = []
            ############################
            # Parallel Category Chart: #
            ############################
            elif self.plot.get('plot_type') == 'parcats':
                _dimensions: List[dict] = []
                _brushing: List[str] = []
                for ft in self.plot.get('features'):
                    if ft in self.feature_types.get('continuous'):
                        _brushing.append(ft)
                    else:
                        _dimensions.append(dict(label=ft, values=self.df[ft].values))
                if len(_dimensions) == 0:
                    Log(write=False).log(msg='No categorical features found. You cannot use Parallel Category chart without categorical features')
                self.plot['kwargs'].update({'dimensions': _dimensions})
                # if _color_feature is None:
                #    self.plot['kwargs'].update({'marker': dict(color='gray') if self.plot['kwargs'].get('marker') is None else self.plot['kwargs'].get('marker'),
                #                                'selected': dict(marker=dict(color='firebrick')) if self.plot['kwargs'].get('selected') is None else self.plot['kwargs'].get('selected'),
                #                                'unselected': dict(marker=dict(opacity=0.3)) if self.plot['kwargs'].get('unselected') is None else self.plot['kwargs'].get('unselected')
                #                                })
                # else:
                #    self.plot['kwargs'].update({'marker': dict(color=_color_feature,
                #                                               colorscale='Hot' if _color_scale is None else _color_scale
                #                                               ) if self.plot['kwargs'].get('marker') is None else self.plot['kwargs'].get('marker'),
                #                                'selected': dict(marker=dict(color='firebrick')) if self.plot['kwargs'].get('selected') is None else self.plot['kwargs'].get('selected'),
                #                                'unselected': dict(marker=dict(color=_color_feature,
                #                                                               colorscale='Hot' if _color_scale is None else _color_scale,
                #                                                               opacity=0.3
                #                                                               )
                #                                                   ) if self.plot['kwargs'].get('unselected') is None else self.plot['kwargs'].get('unselected')
                #                                })
                if self.brushing and len(_brushing) > 0:
                    if len(_brushing) == 1:
                        self.plot['kwargs'].update({'x': self.df[_brushing[0]].values,
                                                    'y': self.df[_brushing[0]].values,
                                                    'mode': 'markers' if self.plot['kwargs'].get('mode') is None else
                                                    self.plot['kwargs'].get('mode'),
                                                    'name': self._trim(input_str=_brushing[0])
                                                    })
                        if self.plot['kwargs'].get('color_toggle_buttons') is None:
                            self.plot['kwargs'].update({'marker': dict(color='gray') if self.plot['kwargs'].get(
                                'marker') is None else self.plot['kwargs'].get('marker'),
                                                        'selected': dict(marker=dict(color='firebrick')) if self.plot[
                                                                                                                'kwargs'].get(
                                                            'selected') is None else self.plot['kwargs'].get(
                                                            'selected'),
                                                        'unselected': dict(marker=dict(opacity=0.3)) if self.plot[
                                                                                                            'kwargs'].get(
                                                            'unselected') is None else self.plot['kwargs'].get(
                                                            'unselected')
                                                        })
                            _data.append(PlotlyAdapter(plot=self.plot, offline=True).scatter_gl())
                            self.plot['kwargs'].update({'domain': dict(y=[0, 0.4]),
                                                        'line': dict(color=np.zeros(self.df.shape[0], dtype='uint8') if
                                                        self.plot['kwargs'].get('color') is None else self.plot[
                                                            'kwargs'].get('color'),
                                                                     cmin=0 if self.plot['kwargs'].get(
                                                                         'cmin') is None else self.plot['kwargs'].get(
                                                                         'cmin'),
                                                                     cmax=1 if self.plot['kwargs'].get(
                                                                         'cmax') is None else self.plot['kwargs'].get(
                                                                         'cmax'),
                                                                     colorscale=[[0, 'gray'], [1, 'firebrick']] if
                                                                     self.plot['kwargs'].get('colorscale') is None else
                                                                     self.plot['kwargs'].get('colorscale'),
                                                                     shape='hspline' if self.plot['kwargs'].get(
                                                                         'shape') is None else self.plot['kwargs'].get(
                                                                         'shape')
                                                                     )
                                                        })
                            _data.append(PlotlyAdapter(plot=self.plot, offline=True).parallel_category())
                        else:
                            self.plot['kwargs'].update({'marker': dict(
                                color=np.zeros(len(self.df), dtype='uint8') if self.plot['kwargs'].get(
                                    'color') is None else self.plot['kwargs'].get('color'),
                                cmin=-0.5 if self.plot['kwargs'].get('cmin') is None else self.plot['kwargs'].get(
                                    'cmin'),
                                cmax=2.5 if self.plot['kwargs'].get('cmax') is None else self.plot['kwargs'].get(
                                    'cmax'),
                                colorscale=[[0, 'gray'], [0.33, 'gray'], [0.33, 'firebrick'], [0.66, 'firebrick'],
                                            [0.66, 'blue'], [1.0, 'blue']] if self.plot['kwargs'].get(
                                    'colorscale') is None else self.plot['kwargs'].get('colorscale'),
                                showscale=True if self.plot['kwargs'].get('showscale') is None else self.plot[
                                    'kwargs'].get('showscale'),
                                colorbar=dict(tickvals=[0, 1, 2],
                                              ticktext=['None', 'Red', 'Blue']
                                              ) if self.plot['kwargs'].get('colorbar') is None else self.plot[
                                    'kwargs'].get('colorbar'),
                                )
                                                        })
                            _data.append(PlotlyAdapter(plot=self.plot, offline=True).scatter_gl())
                            self.plot['kwargs'].update({'domain': dict(y=[0, 0.4]),
                                                        'line': dict(color=np.zeros(self.df.shape[0], dtype='uint8') if
                                                        self.plot['kwargs'].get('color') is None else self.plot[
                                                            'kwargs'].get('color'),
                                                                     cmin=0 if self.plot['kwargs'].get(
                                                                         'cmin') is None else self.plot['kwargs'].get(
                                                                         'cmin'),
                                                                     cmax=1 if self.plot['kwargs'].get(
                                                                         'cmax') is None else self.plot['kwargs'].get(
                                                                         'cmax'),
                                                                     colorscale=[[0, 'gray'], [0.33, 'gray'],
                                                                                 [0.33, 'firebrick'],
                                                                                 [0.66, 'firebrick'], [0.66, 'blue'],
                                                                                 [1.0, 'blue']] if self.plot[
                                                                                                       'kwargs'].get(
                                                                         'colorscale') is None else self.plot[
                                                                         'kwargs'].get('colorscale'),
                                                                     shape='hspline' if self.plot['kwargs'].get(
                                                                         'shape') is None else self.plot['kwargs'].get(
                                                                         'shape')
                                                                     )
                                                        })
                            _data.append(PlotlyAdapter(plot=self.plot, offline=True).parallel_category())
                        self.fig = go.FigureWidget(data=_data, layout=go.Layout(xaxis=dict(title=_brushing[0]),
                                                                                yaxis=dict(title=_brushing[0],
                                                                                           domain=[0.6, 1]),
                                                                                height=self.height,
                                                                                dragmode='lasso',
                                                                                hovermode='closest'
                                                                                )
                                                   )
                        _fig = self.fig
                        if self.plot['kwargs'].get('color_toggle_buttons') is None:
                            # Register callback on scatter selection...
                            _fig.data[0].on_selection(self.brushing_update_color)
                            # and parcats click
                            _fig.data[1].on_click(self.brushing_update_color)
                            if self.file_path is not None:
                                if len(self.file_path) >= 0:
                                    PlotlyAdapter(plot=self.plot, offline=True, fig=_fig).save()
                            if self.render:
                                display(_fig)
                        else:
                            if isinstance(self.plot['kwargs'].get('color_toggle_buttons'), dict):
                                self.plot['kwargs']['color_toggle_buttons'].update({'options': ['None', 'Red',
                                                                                                'Blue'] if self.plot[
                                                                                                               'kwargs'].get(
                                    'options') is None else self.plot['kwargs'].get('options'),
                                                                                    'index': 1 if self.plot[
                                                                                                      'kwargs'].get(
                                                                                        'index') is None else self.plot[
                                                                                        'kwargs'].get('index'),
                                                                                    'description': 'Brush Color:' if
                                                                                    self.plot['kwargs'].get(
                                                                                        'description') is None else
                                                                                    self.plot['kwargs'].get(
                                                                                        'description'),
                                                                                    'disabled': False if self.plot[
                                                                                                             'kwargs'].get(
                                                                                        'disabled') is None else
                                                                                    self.plot['kwargs'].get('disabled')
                                                                                    })
                            else:
                                self.plot['kwargs']['color_toggle_buttons'].update({'options': ['None', 'Red', 'Blue'],
                                                                                    'index': 1,
                                                                                    'description': 'Brush Color:',
                                                                                    'disabled': False
                                                                                    })
                            # Build color selection widget
                            self.color_toggle = widgets.ToggleButtons(
                                options=self.plot['kwargs']['color_toggle_buttons'].get('options'),
                                index=self.plot['kwargs']['color_toggle_buttons'].get('index'),
                                description=self.plot['kwargs']['color_toggle_buttons'].get('description'),
                                disabled=self.plot['kwargs']['color_toggle_buttons'].get('disabled')
                                )
                            # Register callback on scatter selection...
                            _fig.data[0].on_selection(self.brushing_update_color_toogle)
                            # and parcats click
                            _fig.data[1].on_click(self.brushing_update_color_toogle)
                            if self.file_path is not None:
                                if len(self.file_path) >= 0:
                                    PlotlyAdapter(plot=self.plot, offline=True, fig=_fig).save()
                            if self.render:
                                # Display figure
                                display(widgets.VBox([self.color_toggle, _fig]))
                    else:
                        _pairs: List[tuple] = EasyExploreUtils().get_pairs(features=_brushing, max_features_each_pair=2)
                        for pair in _pairs:
                            self.file_path_extension = self._trim(input_str='{}_{}'.format(pair[0], pair[1]))
                            self.plot['kwargs'].update({'x': self.df[pair[0]].values,
                                                        'y': self.df[pair[1]].values,
                                                        'mode': 'markers' if self.plot['kwargs'].get(
                                                            'mode') is None else self.plot['kwargs'].get('mode'),
                                                        'name': self._trim(input_str='{} x {}'.format(pair[0], pair[1]))
                                                        })
                            if self.plot['kwargs'].get('color_toggle_buttons') is None:
                                self.plot['kwargs'].update({'marker': dict(color='gray') if self.plot['kwargs'].get(
                                    'marker') is None else self.plot['kwargs'].get('marker'),
                                                            'selected': dict(marker=dict(color='firebrick')) if
                                                            self.plot['kwargs'].get('selected') is None else self.plot[
                                                                'kwargs'].get('selected'),
                                                            'unselected': dict(marker=dict(opacity=0.3)) if self.plot[
                                                                                                                'kwargs'].get(
                                                                'unselected') is None else self.plot['kwargs'].get(
                                                                'unselected')
                                                            })
                                _data.append(PlotlyAdapter(plot=self.plot, offline=True).scatter_gl())
                                self.plot['kwargs'].update({'domain': dict(y=[0, 0.3]),
                                                            'line': dict(
                                                                color=np.zeros(self.df.shape[0], dtype='uint8') if
                                                                self.plot['kwargs'].get('color') is None else self.plot[
                                                                    'kwargs'].get('color'),
                                                                cmin=0 if self.plot['kwargs'].get('cmin') is None else
                                                                self.plot['kwargs'].get('cmin'),
                                                                cmax=1 if self.plot['kwargs'].get('cmax') is None else
                                                                self.plot['kwargs'].get('cmax'),
                                                                colorscale=[[0, 'gray'], [1, 'firebrick']] if self.plot[
                                                                                                                  'kwargs'].get(
                                                                    'colorscale') is None else self.plot['kwargs'].get(
                                                                    'colorscale'),
                                                                shape='hspline' if self.plot['kwargs'].get(
                                                                    'shape') is None else self.plot['kwargs'].get(
                                                                    'shape')
                                                                )
                                                            })
                                _data.append(PlotlyAdapter(plot=self.plot, offline=True).parallel_category())
                            else:
                                self.plot['kwargs'].update({'marker': dict(
                                    color=np.zeros(self.df.shape[0], dtype='uint8') if self.plot['kwargs'].get(
                                        'color') is None else self.plot['kwargs'].get('color'),
                                    cmin=-0.5 if self.plot['kwargs'].get('cmin') is None else self.plot['kwargs'].get(
                                        'cmin'),
                                    cmax=2.5 if self.plot['kwargs'].get('cmax') is None else self.plot['kwargs'].get(
                                        'cmax'),
                                    colorscale=[[0, 'gray'], [0.33, 'gray'], [0.33, 'firebrick'], [0.66, 'firebrick'],
                                                [0.66, 'blue'], [1.0, 'blue']] if self.plot['kwargs'].get(
                                        'colorscale') is None else self.plot['kwargs'].get('colorscale'),
                                    showscale=True if self.plot['kwargs'].get('showscale') is None else self.plot[
                                        'kwargs'].get('showscale'),
                                    colorbar=dict(tickvals=[0, 1, 2],
                                                  ticktext=['None', 'Red', 'Blue']) if self.plot['kwargs'].get(
                                        'colorbar') is None else self.plot['kwargs'].get('colorbar'),
                                    )
                                                            })
                                _data.append(PlotlyAdapter(plot=self.plot, offline=True).scatter_gl())
                                self.plot['kwargs'].update({'domain': dict(y=[0, 0.3]),
                                                            'line': dict(
                                                                color=np.zeros(self.df.shape[0], dtype='uint8') if
                                                                self.plot['kwargs'].get('color') is None else self.plot[
                                                                    'kwargs'].get('color'),
                                                                cmin=-0.5 if self.plot['kwargs'].get(
                                                                    'cmin') is None else self.plot['kwargs'].get(
                                                                    'cmin'),
                                                                cmax=2.5 if self.plot['kwargs'].get('cmax') is None else
                                                                self.plot['kwargs'].get('cmax'),
                                                                colorscale=[[0, 'gray'], [0.33, 'gray'],
                                                                            [0.33, 'firebrick'], [0.66, 'firebrick'],
                                                                            [0.66, 'blue'], [1.0, 'blue']] if self.plot[
                                                                                                                  'kwargs'].get(
                                                                    'colorscale') is None else self.plot['kwargs'].get(
                                                                    'colorscale'),
                                                                showscale=True if self.plot['kwargs'].get(
                                                                    'showscale') is None else self.plot['kwargs'].get(
                                                                    'showscale'),
                                                                )
                                                            })
                                _data.append(PlotlyAdapter(plot=self.plot, offline=True).parallel_category())
                            self.fig = go.FigureWidget(data=_data, layout=go.Layout(xaxis=dict(title=pair[0]),
                                                                                    yaxis=dict(title=pair[1],
                                                                                               domain=[0.6, 1]),
                                                                                    dragmode='lasso',
                                                                                    hovermode='closest'
                                                                                    )
                                                       )
                            _fig = self.fig
                            if self.plot['kwargs'].get('color_toggle_buttons') is None:
                                # Register callback on scatter selection...
                                _fig.data[0].on_selection(self.brushing_update_color)
                                # and parcats click
                                _fig.data[1].on_click(self.brushing_update_color)
                                if self.file_path is not None:
                                    if len(self.file_path) >= 0:
                                        PlotlyAdapter(plot=self.plot, offline=True, fig=_fig).save()
                                if self.render:
                                    # Display figure
                                    display(_fig)
                            else:
                                if isinstance(self.plot['kwargs'].get('color_toggle_buttons'), dict):
                                    self.plot['kwargs']['color_toggle_buttons'].update({'options': ['None', 'Red',
                                                                                                    'Blue'] if
                                    self.plot['kwargs'].get('options') is None else self.plot['kwargs'].get('options'),
                                                                                        'index': 1 if self.plot[
                                                                                                          'kwargs'].get(
                                                                                            'index') is None else
                                                                                        self.plot['kwargs'].get(
                                                                                            'index'),
                                                                                        'description': 'Brush Color:' if
                                                                                        self.plot['kwargs'].get(
                                                                                            'description') is None else
                                                                                        self.plot['kwargs'].get(
                                                                                            'description'),
                                                                                        'disabled': False if self.plot[
                                                                                                                 'kwargs'].get(
                                                                                            'disabled') is None else
                                                                                        self.plot['kwargs'].get(
                                                                                            'disabled')
                                                                                        })
                                else:
                                    self.plot['kwargs']['color_toggle_buttons'].update(
                                        {'options': ['None', 'Red', 'Blue'],
                                         'index': 1,
                                         'description': 'Brush Color:',
                                         'disabled': False
                                         })
                                # Build color selection widget
                                self.color_toggle = widgets.ToggleButtons(
                                    options=self.plot['kwargs']['color_toggle_buttons'].get('options'),
                                    index=self.plot['kwargs']['color_toggle_buttons'].get('index'),
                                    description=self.plot['kwargs']['color_toggle_buttons'].get('description'),
                                    disabled=self.plot['kwargs']['color_toggle_buttons'].get('disabled')
                                    )
                                # Register callback on scatter selection...
                                _fig.data[0].on_selection(self.brushing_update_color_toogle)
                                # and parcats click
                                _fig.data[1].on_click(self.brushing_update_color_toogle)
                                if self.file_path is not None:
                                    if len(self.file_path) >= 0:
                                        PlotlyAdapter(plot=self.plot, offline=True, fig=_fig).save()
                                if self.render:
                                    # Display figure
                                    display(widgets.VBox([self.color_toggle, _fig]))
                else:
                    self.fig = PlotlyAdapter(plot=self.plot, offline=True).parallel_category()
                    self._show_plotly_offline()
            ###############################
            # Parallel Coordinates Chart: #
            ###############################
            elif self.plot.get('plot_type') == 'parcoords':
                _dimensions: List[dict] = []
                _brushing: List[str] = []
                _perc: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                if self.plot.get('group_by') is None:
                    _desc: dict = self.df[self.plot.get('features')].describe(percentiles=_perc).to_dict()
                    for i, ft in enumerate(self.plot.get('features')):
                        if _desc.get(ft) is None:
                            self.plot['kwargs'].update({'ticktext': self.df[ft].unique()})
                            self.df[ft] = EasyExploreUtils().label_encoder(values=self.df[ft].values)
                            self.plot['kwargs'].update({'tickvals': sorted(self.df[ft].unique())})
                            _range: list = [self.df[ft].min(), self.df[ft].max()]
                        else:
                            self.plot['kwargs'].update({'tickvals': None})
                            self.plot['kwargs'].update({'ticktext': None})
                            _range: list = [_desc[ft].get('min'), _desc[ft].get('max')]
                        _dimensions.append(dict(label=ft if self.plot['kwargs'].get('dimensions_label') is None else
                        self.plot['kwargs'].get('dimensions_label')[i],
                                                values=self.df[ft].values if self.plot['kwargs'].get(
                                                    'dimensions_values') is None else
                                                self.plot['kwargs'].get('dimensions_values')[i],
                                                range=_range if self.plot['kwargs'].get('dimensions_range') is None else
                                                self.plot['kwargs'].get('dimensions_range')[i],
                                                visible=True if self.plot['kwargs'].get(
                                                    'dimensions_visible') is None else
                                                self.plot['kwargs'].get('dimensions_visible')[i]
                                                )
                                           )
                        if self.plot['kwargs'].get('constraintrange') is not None:
                            _dimensions[i].update({'constraintrange': self.plot['kwargs'].get('constraintrange')})
                        if self.plot['kwargs'].get('tickvals') is not None:
                            _dimensions[i].update({'tickvals': self.plot['kwargs'].get('tickvals')})
                        if self.plot['kwargs'].get('ticktext') is not None:
                            _dimensions[i].update({'ticktext': self.plot['kwargs'].get('ticktext')})
                    if len(_dimensions) == 0:
                        raise DataVisualizer('No continuous or semi-continuous feature found')
                    self.plot['kwargs'].update({'dimensions': _dimensions,
                                                'line': dict(color=_color_feature,
                                                             colorscale='Jet' if _color_scale is None else _color_scale,
                                                             colorbar=_color_bar,
                                                             showscale=True,
                                                             reversescale=False,
                                                             # cmin=np.array([_desc[f].get('min') for f in _desc.keys()]).min(),
                                                             # cmax=np.array([_desc[f].get('max') for f in _desc.keys()]).max(),
                                                             ) if self.plot['kwargs'].get('line') is None else
                                                self.plot['kwargs'].get('line')
                                                })
                    self.fig = PlotlyAdapter(plot=self.plot, offline=True).parallel_coordinates()
                    self._show_plotly_offline()
                else:
                    for i, group in enumerate(self.plot.get('group_by'), start=1):
                        for val in self.df[group].unique():
                            if val in INVALID_VALUES:
                                _val: np.array = self.df.loc[self.df[group].isnull(), self.plot.get('features')].values
                                _desc: dict = self.df.loc[self.df[group].isnull(), self.plot.get('features')].describe(
                                    percentiles=_perc).to_dict()
                            else:
                                _val: np.array = self.df.loc[self.df[group] == val, self.plot.get('features')].values
                                _desc: dict = self.df.loc[self.df[group] == val, self.plot.get('features')].describe(
                                    percentiles=_perc).to_dict()
                            #print(_val, '\n\n', _desc)
                            for j, ft in enumerate(self.plot.get('features')):
                                if _desc.get(ft) is None:
                                    self.plot['kwargs'].update({'ticktext': self.df[ft].unique()})
                                    #self.df[ft] = EasyExploreUtils().label_encoder(values=self.df[ft].values)
                                    self.plot['kwargs'].update({'tickvals': sorted(self.df[ft].unique())})
                                    _range: list = [self.df[ft].min(), self.df[ft].max()]
                                else:
                                    self.plot['kwargs'].update({'tickvals': None})
                                    self.plot['kwargs'].update({'ticktext': None})
                                    _range: list = [_desc[ft].get('min'), _desc[ft].get('max')]
                                _dimensions.append(dict(
                                    label=ft if self.plot['kwargs'].get('dimensions_label') is None else
                                    self.plot['kwargs'].get('dimensions_label')[j],
                                    values=_val if self.plot['kwargs'].get('dimensions_values') is None else
                                    self.plot['kwargs'].get('dimensions_values')[j],
                                    range=_range if self.plot['kwargs'].get('dimensions_range') is None else
                                    self.plot['kwargs'].get('dimensions_range')[j],
                                    visible=True if self.plot['kwargs'].get('dimensions_visible') is None else
                                    self.plot['kwargs'].get('dimensions_visible')[j]
                                    )
                                                   )
                                if self.plot['kwargs'].get('constraintrange') is not None:
                                    _dimensions[i].update(
                                        {'constraintrange': self.plot['kwargs'].get('constraintrange')})
                                if self.plot['kwargs'].get('tickvals') is not None:
                                    _dimensions[i].update({'tickvals': self.plot['kwargs'].get('tickvals')})
                                if self.plot['kwargs'].get('ticktext') is not None:
                                    _dimensions[i].update({'ticktext': self.plot['kwargs'].get('ticktext')})
                            if len(_dimensions) == 0:
                                raise DataVisualizer('No continuous or semi-continuous feature found')
                            self.plot['kwargs'].update({'dimensions': _dimensions,
                                                        'line': dict(color=self.df[
                                                            group].values if _color_feature is None else _color_feature,
                                                                     colorscale='Jet' if _color_scale is None else _color_scale,
                                                                     showscale=True,
                                                                     reversescale=False,
                                                                     # cmin=np.array([_desc[f].get('min') for f in _desc.keys()]).min(),
                                                                     # cmax=np.array([_desc[f].get('max') for f in _desc.keys()]).max(),
                                                                     ) if self.plot['kwargs'].get('line') is None else
                                                        self.plot['kwargs'].get('line')
                                                        })
                            self.fig = PlotlyAdapter(plot=self.plot, offline=True).parallel_coordinates()
                            self._show_plotly_offline()
            #################
            # Geomap Chart: #
            #################
            # TODO: Implement 'group_by' and 'melt' functionality
            elif self.plot.get('plot_type') == 'geo':
                if self.plot['kwargs'].get('lon') is None:
                    raise DataVisualizerException('No longitude information found')
                if self.plot['kwargs'].get('lat') is None:
                    raise DataVisualizerException('No latitude information found')
                _visualize_features: List[str] = self.plot.get('features') + [self.plot['kwargs'].get('lon'), self.plot['kwargs'].get('lat')]
                for viz_feature in _visualize_features:
                    self.df = self.df.loc[~self.df[viz_feature].isnull(), _visualize_features]
                    if str(self.df[viz_feature].dtype).find('object') >= 0:
                        self.df[viz_feature] = self.df[viz_feature].astype(float)
                _lat_median: int = int(np.median(self.df[self.plot['kwargs'].get('lat')]))
                _lon_median: int = int(np.median(self.df[self.plot['kwargs'].get('lon')]))
                _max_marker_size: int = 50 if self.plot['kwargs'].get('max_marker_size') is None else self.plot['kwargs'].get('max_marker_size')
                self.plot['kwargs'].update({'lon': self.df[self.plot['kwargs'].get('lon')],
                                            'lat': self.df[self.plot['kwargs'].get('lat')],
                                            'fill': 'toself' if self.plot['kwargs'].get('fill') is None else self.plot[
                                                'kwargs'].get('fill')
                                            })
                self.plot['kwargs']['layout'].update({'mapbox': dict(
                    style='stamen-terrain' if self.plot['kwargs'].get('style') is None else self.plot['kwargs'].get(
                        'style'),
                    center=dict(lat=_lat_median, lon=_lon_median) if self.plot['kwargs'].get('center') is None else
                    self.plot['kwargs'].get('center'),
                    zoom=5 if self.plot['kwargs'].get('zoom') is None else self.plot['kwargs'].get('zoom')
                    )
                                                      })
                if len(self.plot.get('features')) > 1:
                    self.fig = go.Figure()
                    if self.plot.get('color_edges') is None:
                        _color_scale = n_colors(lowcolor='rgb(0, 0, 0)',
                                                highcolor='rgb(255, 255, 255)',
                                                n_colors=len(self.plot.get('features')),
                                                colortype='rgb')
                    else:
                        _color_scale = n_colors(lowcolor=self.plot.get('color_edges')[0],
                                                highcolor=self.plot.get('color_edges')[1],
                                                n_colors=len(self.plot.get('features')),
                                                colortype=self.color_type)
                    for i, ft in enumerate(self.plot.get('features')):
                        _size: list = [(val / self.df[ft].max()) * _max_marker_size for val in self.df[ft].values]
                        self.plot['kwargs'].update({'marker': dict(size=_size, color=_color_scale[i], opacity=0.6) if
                        self.plot['kwargs'].get('marker') is None else self.plot['kwargs'].get('marker'),
                                                    'text': self.df[ft].values.tolist() if self.plot['kwargs'].get(
                                                        'text') is None else self.plot['kwargs'].get('text')
                                                    })
                        # self.plot['kwargs'].update({'hovertemplate': '<b>{}</b>'.format(ft) + '%{marker:.2f}' + '<br><b>Lon</b></br>' + '%{lon}' + '<b>Lat</b>' + '%{lat}' if self.plot['kwargs'].get('hovertemplate') is None else self.plot['kwargs'].get('hovertemplate')})
                        self.fig.add_trace(PlotlyAdapter(plot=self.plot, offline=True).scatter_mapbox())
                    self._show_plotly_offline()
                else:
                    self.plot['kwargs'].update({'marker': dict(size=10,
                                                               color='blue' if _color_feature is None else _color_feature,
                                                               colorscale='IceFire' if _color_scale is None else _color_scale,
                                                               opacity=0.7
                                                               ) if self.plot['kwargs'].get('marker') is None else self.plot['kwargs'].get('marker')
                                                })
                    self.fig = PlotlyAdapter(plot=self.plot, offline=True).scatter_mapbox()
                    self._show_plotly_offline()
            #####################
            # Densitymap Chart: #
            #####################
            # TODO: Implement 'group_by' and 'melt' functionality
            elif self.plot.get('plot_type') == 'density':
                if self.plot['kwargs'].get('lon') is None:
                    raise DataVisualizerException('No longitude information found')
                if self.plot['kwargs'].get('lat') is None:
                    raise DataVisualizerException('No latitude information found')
                for feature in self.plot.get('features'):
                    _visualize_features: List[str] = [feature, self.plot['kwargs'].get('lon'), self.plot['kwargs'].get('lat')]
                    for viz_feature in _visualize_features:
                        self.df = self.df.loc[~self.df[viz_feature].isnull(), _visualize_features]
                        if str(self.df[viz_feature].dtype).find('object') >= 0:
                            self.df[viz_feature] = self.df[viz_feature].astype(float)
                    _lat_median: int = int(np.median(self.df[self.plot['kwargs'].get('lat')]))
                    _lon_median: int = int(np.median(self.df[self.plot['kwargs'].get('lon')]))
                    self.plot['kwargs'].update({'lon': self.df[self.plot['kwargs'].get('lon')],
                                                'lat': self.df[self.plot['kwargs'].get('lat')]
                                                })
                    self.plot['kwargs']['layout'].update({'mapbox': dict(
                        style='stamen-terrain' if self.plot['kwargs'].get('style') is None else self.plot['kwargs'].get(
                            'style'),
                        center=dict(lat=_lat_median, lon=_lon_median) if self.plot['kwargs'].get('center') is None else
                        self.plot['kwargs'].get('center'),
                        zoom=5 if self.plot['kwargs'].get('zoom') is None else self.plot['kwargs'].get('zoom')
                    )
                    })
                    self.plot['kwargs'].update({'z': self.df[feature].values})
                    self.fig = PlotlyAdapter(plot=self.plot, offline=True).densitymapbox()
                    self._show_plotly_offline()
            #######################
            # Chorolethmap Chart: #
            #######################
            elif self.plot.get('plot_type') == 'choro':
                _visualize_features: List[str] = self.plot.get('features') + [self.plot['kwargs'].get('lon'), self.plot['kwargs'].get('lat')]
                for viz_feature in _visualize_features:
                    self.df = self.df.loc[~self.df[viz_feature].isnull(), _visualize_features]
                    if str(self.df[viz_feature].dtype).find('object') >= 0:
                        self.df[viz_feature] = self.df[viz_feature].astype(float)
                _lat_median: int = int(np.median(self.df[self.plot['kwargs'].get('lat')]))
                _lon_median: int = int(np.median(self.df[self.plot['kwargs'].get('lon')]))
                if self.plot['kwargs'].get('geojson') is None:
                    if self.plot['kwargs'].get('lon') is None:
                        raise DataVisualizerException('No longitude information found')
                    if self.plot['kwargs'].get('lat') is None:
                        raise DataVisualizerException('No latitude information found')
                    self.plot['kwargs'].update({'geojson': self.plot['kwargs'].get('geojson')})
                    # generate geojson
                else:
                    if isinstance(self.plot['kwargs'].get('geojson'), dict):
                        self.plot['kwargs'].update({'locations': self.plot['kwargs'].get('locations'),
                                                    'z': self.plot['kwargs'].get('z'),
                                                    'marker': dict(opacity=0.7, line_width=0) if self.plot[
                                                                                                     'kwargs'].get(
                                                        'marker') is None else self.plot['kwargs'].get('marker'),
                                                    })
                    else:
                        pass
                        # generate geojson:
                self.plot['kwargs']['layout'].update({'mapbox': dict(
                    style='carto-positron' if self.plot['kwargs'].get('style') is None else self.plot['kwargs'].get(
                        'style'),
                    center=dict(lat=_lat_median, lon=_lon_median) if self.plot['kwargs'].get('center') is None else
                    self.plot['kwargs'].get('center'),
                    zoom=5 if self.plot['kwargs'].get('zoom') is None else self.plot['kwargs'].get('zoom'),
                    margin={'r': 0, 't': 0, 'l': 0, 'b': 0} if self.plot['kwargs'].get('margin') is None else self.plot[
                        'kwargs'].get('margin')
                )
                })
                for ft in self.plot.get('features'):
                    self.plot['kwargs'].update({'z': self.df[ft].values})
                    self.fig = PlotlyAdapter(plot=self.plot, offline=True).choroplethmapbox()
                    self._show_plotly_offline()
            ################
            # Joint Chart: #
            ################
            elif self.plot.get('plot_type') == 'joint':
                _features: List[str] = self.plot.get('features')
                if len(_features) < 2:
                    raise DataVisualizerException(
                        'Not enough features ({}) for generating scatter plot'.format(len(_features)))
                _pairs: List[tuple] = EasyExploreUtils().get_pairs(features=_features, max_features_each_pair=2)
                if self.plot.get('group_by') is None:
                    for j, pair in enumerate(_pairs, start=1):
                        _fig: go.Figure = go.Figure()
                        if str(self.df[pair[0]].dtype).find('float') < 0:
                            self.df[pair[0]] = self.df[pair[0]].astype(float)
                        if str(self.df[pair[1]].dtype).find('float') < 0:
                            self.df[pair[1]] = self.df[pair[1]].astype(float)
                        self.file_path_extension = self._trim(input_str='{}_{}'.format(pair[0], pair[1]))
                        self.plot['kwargs'].update({'x': self.df[pair[0]].values,
                                                    'y': self.df[pair[1]].values,
                                                    'mode': 'markers' if self.plot['kwargs'].get('mode') is None else
                                                    self.plot['kwargs'].get('mode'),
                                                    'name': self._trim(input_str='{} x {}'.format(pair[0], pair[1])),
                                                    'xaxis': 'x',
                                                    'yaxis': 'y',
                                                    'reversescale': True
                                                    })
                        _fig.add_trace(PlotlyAdapter(plot=self.plot, offline=True).histogram_2d_contour())
                        _fig.add_trace(PlotlyAdapter(plot=self.plot, offline=True).scatter_gl())
                        self.plot['kwargs'].update({'x': None,
                                                    'y': self.df[pair[1]].values,
                                                    'xaxis': 'x2',
                                                    'yaxis': None
                                                    })
                        _fig.add_trace(PlotlyAdapter(plot=self.plot, offline=True).histo())
                        self.plot['kwargs'].update({'x': self.df[pair[0]].values,
                                                    'y': None,
                                                    'xaxis': None,
                                                    'yaxis': 'y2'
                                                    })
                        _fig.add_trace(PlotlyAdapter(plot=self.plot, offline=True).histo())
                        self.plot['kwargs']['layout'].update({'autosize': False,
                                                              'bargap': 0,
                                                              'hovermode': 'closest',
                                                              'showlegend': False,
                                                              'xaxis': dict(zeroline=False,
                                                                            domain=[0, 0.65],
                                                                            showgrid=False,
                                                                            title=pair[0]
                                                                            ),
                                                              'yaxis': dict(zeroline=False,
                                                                            domain=[0, 0.65],
                                                                            showgrid=False,
                                                                            title=pair[1]
                                                                            ),
                                                              'xaxis2': dict(zeroline=False,
                                                                             domain=[0.65, 1],
                                                                             showgrid=False
                                                                             ),
                                                              'yaxis2': dict(zeroline=False,
                                                                             domain=[0.65, 1],
                                                                             showgrid=False
                                                                             )
                                                              })
                        self.fig = _fig
                        self._show_plotly_offline()
                else:
                    for pair in _pairs:
                        _data: List[go] = []
                        for group in self.plot.get('group_by'):
                            _group_val: List[str] = self.df[group].unique().tolist()
                            for ft in self.plot.get('features'):
                                for ext, val in enumerate(_group_val):
                                    _fig: go.Figure = go.Figure()
                                    self.file_path_extension = self._trim(input_str='{}_{}_{}'.format(ft, group, ext))
                                    self.plot['kwargs'].update({'mode': 'markers' if self.plot['kwargs'].get(
                                        'mode') is None else self.plot['kwargs'].get('mode'),
                                                                'name': self._trim(
                                                                    input_str='{} ({}={})'.format(ft, group, val)),
                                                                'xaxis': 'x',
                                                                'yaxis': 'y',
                                                                'reversescale': True
                                                                })
                                    if val in INVALID_VALUES:
                                        self.plot['kwargs'].update(
                                            {'x': self.df.loc[self.df[group].isnull(), pair[1]].values,
                                             'y': self.df.loc[self.df[group].isnull(), pair[0]].values
                                             })
                                    else:
                                        self.plot['kwargs'].update(
                                            {'x': self.df.loc[self.df[group] == val, pair[1]].values,
                                             'y': self.df.loc[self.df[group] == val, pair[0]].values
                                             })
                                    _fig.add_trace(PlotlyAdapter(plot=self.plot, offline=True).histogram_2d_contour())
                                    _fig.add_trace(PlotlyAdapter(plot=self.plot, offline=True).scatter_gl())
                                    self.plot['kwargs'].update({'x': None,
                                                                'y': self.df[pair[1]].values,
                                                                'xaxis': 'x2',
                                                                'yaxis': None
                                                                })
                                    _fig.add_trace(PlotlyAdapter(plot=self.plot, offline=True).histo())
                                    self.plot['kwargs'].update({'x': self.df[pair[0]].values,
                                                                'y': None,
                                                                'xaxis': None,
                                                                'yaxis': 'y2'
                                                                })
                                    if val in INVALID_VALUES:
                                        self.plot['kwargs'].update({'x': self.df[pair[0]].values,
                                                                    'y': None,
                                                                    'xaxis': None,
                                                                    'yaxis': 'y2'
                                                                    })
                                    else:
                                        self.plot['kwargs'].update({'x': self.df[pair[0]].values,
                                                                    'y': None,
                                                                    'xaxis': None,
                                                                    'yaxis': 'y2'
                                                                    })
                                    _fig.add_trace(PlotlyAdapter(plot=self.plot, offline=True).histo())
                                    self.plot['kwargs']['layout'].update({'autosize': False,
                                                                          'bargap': 0,
                                                                          'hovermode': 'closest',
                                                                          'showlegend': False,
                                                                          'xaxis': dict(zeroline=False,
                                                                                        domain=[0, 0.65],
                                                                                        showgrid=False),
                                                                          'yaxis': dict(zeroline=False,
                                                                                        domain=[0, 0.65],
                                                                                        showgrid=False),
                                                                          'xaxis2': dict(zeroline=False,
                                                                                         domain=[0.65, 1],
                                                                                         showgrid=False),
                                                                          'yaxis2': dict(zeroline=False,
                                                                                         domain=[0.65, 1],
                                                                                         showgrid=False)
                                                                          })
                                    self.fig = _fig
                                    self._show_plotly_offline()
            ######################
            # Candlestick Chart: #
            ######################
            elif self.plot.get('plot_type') == 'candlestick':
                _features: List[str] = self.plot.get('features')
                if len(_features) == 0:
                    raise DataVisualizerException('No feature found')
                if self.plot['kwargs'].get('open') is None:
                    if len(_features) > 1:
                        self.plot['kwargs'].update({'open': _features[1]})
                    else:
                        raise DataVisualizerException('No "open" value found')
                if self.plot['kwargs'].get('low') is None:
                    if len(_features) > 2:
                        self.plot['kwargs'].update({'low': _features[2]})
                    else:
                        raise DataVisualizerException('No "low" value found')
                if self.plot['kwargs'].get('high') is None:
                    if len(_features) > 3:
                        self.plot['kwargs'].update({'high': _features[3]})
                    else:
                        raise DataVisualizerException('No "high" value found')
                if self.plot['kwargs'].get('close') is None:
                    if len(_features) > 4:
                        self.plot['kwargs'].update({'close': _features[4]})
                    else:
                        raise DataVisualizerException('No "close" value found')
                if self.plot.get('group_by'):
                    for tft in self.plot.get('time_features'):
                        _data: List[go] = []
                        _sorted_df: pd.DataFrame = self.df.sort_values(by=[tft])
                        for group in self.plot.get('group_by'):
                            _group_val: np.array = _sorted_df[group].unique()
                            for ext, val in enumerate(_group_val, start=1):
                                self.file_path_extension = self._trim(input_str='{}_{}_{}'.format(tft, group, ext))
                                if ext == 0:
                                    self.grouping = True
                                elif ext == len(_group_val):
                                    self.grouping = False
                                self.plot['kwargs'].update(
                                    {'name': self._trim(input_str='{} ({}={})'.format(_features[0], group, val))})
                                if val in INVALID_VALUES:
                                    _sorted_df[group] = _sorted_df[group].replace(INVALID_VALUES, np.nan)
                                    self.plot['kwargs'].update(
                                        {'x': _sorted_df.loc[_sorted_df[group].isnull(), tft].values,
                                         'y': _sorted_df.loc[_sorted_df[group].isnull(), _features[0]].values
                                         })
                                else:
                                    self.plot['kwargs'].update(
                                        {'x': _sorted_df.loc[_sorted_df[group] == val, tft].values,
                                         'y': _sorted_df.loc[_sorted_df[group] == val, _features[0]].values
                                         })
                                _data.append(PlotlyAdapter(plot=self.plot, offline=True).line())
                        self.fig = _data
                        self._show_plotly_offline()
                else:
                    _data: List[go] = []
                    for j, tft in enumerate(self.plot.get('time_features'), start=1):
                        _sorted_df: pd.DataFrame = self.df.sort_values(by=[tft])
                        self.plot['kwargs'].update({'x': _sorted_df[tft].values,
                                                    'y': _sorted_df[_features[0]].values,
                                                    'name': self._trim(input_str=_features[0])
                                                    })
                        if self.plot.get('melt'):
                            _data.append(PlotlyAdapter(plot=self.plot, offline=True).line())
                        else:
                            self.grouping = False
                            self.fig = PlotlyAdapter(plot=self.plot, offline=True).line()
                            self._show_plotly_offline()
                    if len(_data) > 0:
                        self.fig = _data
                        self._show_plotly_offline()
            ###############
            # Line Chart: #
            ###############
            elif self.plot.get('plot_type') == 'line':
                for j, tft in enumerate(self.plot.get('time_features'), start=1):
                    if self.plot.get('group_by') is None:
                        if self.plot.get('melt'):
                            _data: List[go] = []
                            _sorted_df: pd.DataFrame = self.df.sort_values(by=[tft])
                            self.file_path_extension = self._trim(input_str=tft)
                            for k, ft in enumerate(self.plot.get('features'), start=1):
                                self.plot['kwargs'].update({'x': _sorted_df[tft].values,
                                                            'y': _sorted_df[ft].values,
                                                            'mode': 'lines' if self.plot['kwargs'].get(
                                                                'mode') is None else self.plot['kwargs'].get('mode'),
                                                            'name': self._trim(input_str=ft)
                                                            })
                                _data.append(PlotlyAdapter(plot=self.plot, offline=True).line())
                            self.fig = _data
                            self._show_plotly_offline()
                        else:
                            _pairs: List[tuple] = []
                            for ft in self.plot.get('features'):
                                _pairs.append(tuple([ft, tft]))
                            for k, pair in enumerate(_pairs, start=1):
                                _sorted_df: pd.DataFrame = self.df.sort_values(by=[pair[1]])
                                self.file_path_extension = self._trim(input_str='{}_{}'.format(pair[0], pair[1]))
                                self.plot['kwargs'].update({'x': _sorted_df[pair[1]].values,
                                                            'y': _sorted_df[pair[0]].values,
                                                            'mode': 'lines' if self.plot['kwargs'].get(
                                                                'mode') is None else self.plot['kwargs'].get('mode'),
                                                            'name': self._trim(input_str=pair[0])
                                                            })
                                _data.append(PlotlyAdapter(plot=self.plot, offline=True).line())
                            self.fig = _data
                            self._show_plotly_offline()
                    else:
                        _data: List[go] = []
                        _sorted_df: pd.DataFrame = self.df.sort_values(by=[tft])
                        for group in self.plot.get('group_by'):
                            _group_val: np.array = _sorted_df[group].unique()
                            for k, ft in enumerate(self.plot.get('features'), start=1):
                                if k == 0:
                                    self.grouping = True
                                elif k == len(_group_val):
                                    self.grouping = False
                                for ext, val in enumerate(_group_val):
                                    self.file_path_extension = self._trim(input_str='{}_{}_{}'.format(ft, group, ext))
                                    self.plot['kwargs'].update({'mode': 'lines' if self.plot['kwargs'].get(
                                        'mode') is None else self.plot['kwargs'].get('mode'),
                                                                'name': self._trim(
                                                                    input_str='{} ({}={})'.format(ft, group, val))
                                                                })
                                    if val in INVALID_VALUES:
                                        _sorted_df[group] = _sorted_df[group].replace(INVALID_VALUES, np.nan)
                                        self.plot['kwargs'].update(
                                            {'x': _sorted_df.loc[_sorted_df[group].isnull(), tft].values,
                                             'y': _sorted_df.loc[_sorted_df[group].isnull(), ft].values
                                             })
                                    else:
                                        self.plot['kwargs'].update(
                                            {'x': _sorted_df.loc[_sorted_df[group] == val, tft].values,
                                             'y': _sorted_df.loc[_sorted_df[group] == val, ft].values
                                             })
                                    _data.append(PlotlyAdapter(plot=self.plot, offline=True).line())
                                if self.plot.get('melt'):
                                    _data.append(PlotlyAdapter(plot=self.plot, offline=True).bar())
                                    if k == len(_group_val) and j == len(self.plot.get('features')):
                                        self.fig = _data
                                        self._show_plotly_offline()
                                else:
                                    _data.append(PlotlyAdapter(plot=self.plot, offline=True).bar())
                                    self.grouping = False
                                    self.fig = _data
                                    self._show_plotly_offline()
                                    _data = []
            ##################
            # Scatter Chart: #
            ##################
            elif self.plot.get('plot_type') == 'scatter':
                _features: List[str] = self.plot.get('features')
                if len(_features) < 2:
                    raise DataVisualizerException(
                        'Not enough features ({}) for generating scatter chart'.format(len(_features)))
                _pairs: List[tuple] = EasyExploreUtils().get_pairs(features=_features, max_features_each_pair=2)
                if self.plot.get('group_by') is None:
                    for i, pair in enumerate(_pairs, start=1):
                        self.plot['kwargs'].update({'x': self.df[pair[0]].values,
                                                    'y': self.df[pair[1]].values,
                                                    'mode': 'markers' if self.plot['kwargs'].get('mode') is None else self.plot['kwargs'].get('mode'),
                                                    'marker': dict(size=12,
                                                                   color=_color_feature,
                                                                   colorscale=_color_scale,
                                                                   opacity=0.75
                                                                   ) if self.plot['kwargs'].get('marker') is None else self.plot['kwargs'].get('marker'),
                                                    'name': self._trim(input_str='{} - {}'.format(pair[0], pair[1])),
                                                    'showlegend': True if self.plot['kwargs'].get(
                                                        'showlegend') is None else self.plot['kwargs'].get('showlegend')
                                                    })
                        if self.plot.get('melt'):
                            self.file_path_extension = '{}_{}'.format(self.file_path_extension,
                                                                      self._trim(input_str='{}_{}'.format(pair[0], pair[1]))
                                                                      )
                            _data.append(PlotlyAdapter(plot=self.plot, offline=True).scatter_gl())
                            if i == len(_pairs):
                                self.fig = _data
                                self._show_plotly_offline()
                        else:
                            self.grouping = False
                            if self.plot.get('xaxis_label') is None:
                                self.plot['kwargs']['layout'].update({'xaxis': dict(title=dict(text=pair[0]))})
                            if self.plot.get('yaxis_label') is None:
                                self.plot['kwargs']['layout'].update({'yaxis': dict(title=dict(text=pair[1]))})
                            self.file_path_extension = self._trim(input_str='{}_{}'.format(pair[0], pair[1]))
                            self.fig = PlotlyAdapter(plot=self.plot, offline=True).scatter_gl()
                            self._show_plotly_offline()
                else:
                    for i, pair in enumerate(_pairs, start=1):
                        for j, group in enumerate(self.plot.get('group_by'), start=1):
                            _group_val: np.array = self.df[group].unique()
                            for ext, val in enumerate(_group_val, start=1):
                                if ext == 0:
                                    self.grouping = True
                                elif ext == len(_group_val):
                                    self.grouping = False
                                self.plot['kwargs'].update({'mode': 'markers' if self.plot['kwargs'].get(
                                    'mode') is None else self.plot['kwargs'].get('mode'),
                                                            'marker': dict(size=12,
                                                                           color=_color_feature,
                                                                           colorscale=_color_scale,
                                                                           opacity=0.75
                                                                           ) if self.plot['kwargs'].get(
                                                                'marker') is None else self.plot['kwargs'].get(
                                                                'marker'),
                                                            'name': self._trim(
                                                                input_str='{} - {} ({}={})'.format(pair[0], pair[1],
                                                                                                   group, val))
                                                            })
                                if val in INVALID_VALUES:
                                    self.plot['kwargs'].update(
                                        {'x': self.df.loc[self.df[group].isnull(), pair[0]].values,
                                         'y': self.df.loc[self.df[group].isnull(), pair[1]].values
                                         })
                                else:
                                    self.plot['kwargs'].update(
                                        {'x': self.df.loc[self.df[group] == val, pair[0]].values,
                                         'y': self.df.loc[self.df[group] == val, pair[1]].values
                                         })
                                _data.append(PlotlyAdapter(plot=self.plot, offline=True).scatter_gl())
                                if self.plot.get('xaxis_label') is None:
                                    self.plot['kwargs']['layout'].update({'xaxis': dict(title=dict(text=pair[0]))})
                                if self.plot.get('yaxis_label') is None:
                                    self.plot['kwargs']['layout'].update({'yaxis': dict(title=dict(text=pair[1]))})
                                if self.plot.get('melt'):
                                    if i == len(_pairs) and j == len(self.plot.get('group_by')):
                                        self.file_path_extension = self._trim(input_str='{}_{}_{}'.format(pair[0],
                                                                                                          pair[1],
                                                                                                          group
                                                                                                          )
                                                                              )
                                        self.fig = _data
                                        self._show_plotly_offline()
                                else:
                                    self.grouping = False
                                    self.file_path_extension = self._trim(input_str='{}_{}_{}_{}'.format(pair[0],
                                                                                                         pair[1],
                                                                                                         group,
                                                                                                         ext
                                                                                                         )
                                                                          )
                                    self.fig = _data
                                    self._show_plotly_offline()
                                    _data = []
            #####################
            # Scatter 3D Chart: #
            #####################
            elif self.plot.get('plot_type') == 'scatter3d':
                _features: List[str] = self.plot.get('features')
                if len(_features) < 3:
                    raise DataVisualizerException(
                        'Not enough features ({}) for generating scatter 3D chart'.format(len(_features)))
                _pairs: List[tuple] = EasyExploreUtils().get_pairs(features=_features, max_features_each_pair=3)
                if self.plot.get('group_by') is None:
                    for i, pair in enumerate(_pairs, start=1):
                        self.plot['kwargs'].update({'x': self.df[pair[0]].values,
                                                    'y': self.df[pair[1]].values,
                                                    'z': self.df[pair[2]].values,
                                                    'mode': 'markers' if self.plot['kwargs'].get('mode') is None else
                                                    self.plot['kwargs'].get('mode'),
                                                    'marker': dict(size=12,
                                                                   color=_color_feature,
                                                                   colorscale=_color_scale,
                                                                   opacity=0.75
                                                                   ) if self.plot['kwargs'].get('marker') is None else
                                                    self.plot['kwargs'].get('marker'),
                                                    'name': self._trim(
                                                        input_str='{} - {} - {}'.format(pair[0], pair[1], pair[2])),
                                                    'showlegend': True if self.plot['kwargs'].get(
                                                        'showlegend') is None else self.plot['kwargs'].get('showlegend')
                                                    })
                        if self.plot.get('melt'):
                            self.file_path_extension = '{}_{}'.format(self.file_path_extension,
                                                                      self._trim(input_str='{}_{}_{}'.format(pair[0],
                                                                                                             pair[1],
                                                                                                             pair[2]
                                                                                                             )
                                                                                 )
                                                                      )
                            _data.append(PlotlyAdapter(plot=self.plot, offline=True).scatter3d())
                            if i == len(_pairs):
                                self.fig = _data
                                self._show_plotly_offline()
                        else:
                            self.grouping = False
                            if self.plot.get('xaxis_label') is None:
                                self.plot['kwargs']['layout'].update({'xaxis': dict(title=dict(text=pair[0]))})
                            if self.plot.get('yaxis_label') is None:
                                self.plot['kwargs']['layout'].update({'yaxis': dict(title=dict(text=pair[1]))})
                            if self.plot.get('zaxis_label') is None:
                                self.plot['kwargs']['layout'].update({'zaxis': dict(title=dict(text=pair[2]))})
                            self.file_path_extension = self._trim(input_str='{}_{}_{}'.format(pair[0],
                                                                                              pair[1],
                                                                                              pair[2]
                                                                                              )
                                                                  )
                            self.fig = PlotlyAdapter(plot=self.plot, offline=True).scatter3d()
                            self._show_plotly_offline()
                else:
                    for i, pair in enumerate(_pairs, start=1):
                        for j, group in enumerate(self.plot.get('group_by'), start=1):
                            _group_val: np.array = self.df[group].unique()
                            for ext, val in enumerate(_group_val, start=1):
                                if ext == 0:
                                    self.grouping = True
                                elif ext == len(_group_val):
                                    self.grouping = False
                                self.plot['kwargs'].update({'mode': 'markers' if self.plot['kwargs'].get(
                                    'mode') is None else self.plot['kwargs'].get('mode'),
                                                            'marker': dict(size=12,
                                                                           color=_color_feature,
                                                                           colorscale=_color_scale,
                                                                           opacity=0.75
                                                                           ) if self.plot['kwargs'].get(
                                                                'marker') is None else self.plot['kwargs'].get(
                                                                'marker'),
                                                            'name': self._trim(
                                                                input_str='{} - {} - {} ({}={})'.format(pair[0],
                                                                                                        pair[1],
                                                                                                        pair[2], group,
                                                                                                        val))
                                                            })
                                if val in INVALID_VALUES:
                                    self.plot['kwargs'].update(
                                        {'x': self.df.loc[self.df[group].isnull(), pair[0]].values,
                                         'y': self.df.loc[self.df[group].isnull(), pair[1]].values,
                                         'z': self.df.loc[self.df[group].isnull(), pair[2]].values
                                         })
                                else:
                                    self.plot['kwargs'].update(
                                        {'x': self.df.loc[self.df[group] == val, pair[0]].values,
                                         'y': self.df.loc[self.df[group] == val, pair[1]].values,
                                         'z': self.df.loc[self.df[group].isnull(), pair[2]].values
                                         })
                                _data.append(PlotlyAdapter(plot=self.plot, offline=True).scatter3d())
                            if self.plot.get('melt'):
                                if i == len(_pairs) and j == len(self.plot.get('group_by')):
                                    self.file_path_extension = self._trim(input_str='{}_{}_{}_{}'.format(pair[0],
                                                                                                         pair[1],
                                                                                                         pair[2],
                                                                                                         group
                                                                                                         )
                                                                          )
                                    self.fig = _data
                                    self._show_plotly_offline()
                            else:
                                self.grouping = False
                                self.file_path_extension = self._trim(input_str='{}_{}_{}_{}_{}'.format(pair[0],
                                                                                                        pair[1],
                                                                                                        pair[2],
                                                                                                        group,
                                                                                                        ext
                                                                                                        )
                                                                      )
                                self.fig = _data
                                self._show_plotly_offline()
                                _data = []
            #######################
            # Network Graph Chart #
            #######################
            elif self.plot.get('plot_type') == 'network':
                if len(self.plot.get('features')) < 2:
                    raise DataVisualizerException('Not enough features for visualizing network graph')
                if len(self.plot.get('features')) > 2:
                    Log(write=False).log(
                        'Only the first 2 features are used to define x and y axis. All other features ({}) are ignored'.format(
                            ''.join(self.plot.get('features')[2:])))
                # Generate network graph:
                _pos: dict = {self.df.loc[node, self.plot['graph_features'].get('node')]: (self.df.loc[node, self.plot.get('features')[0]], self.df.loc[node, self.plot.get('features')[1]]) for node in range(0, self.df.shape[0], 1)}
                _geometric_params: dict = dict(n=len(self.df[self.plot['graph_features'].get('node')].unique()),
                                               radius=0.2,
                                               pos=_pos
                                               )
                _graph: nx = EasyExploreUtils().generate_network(df=self.df,
                                                                 node_feature=self.plot['graph_features'].get('node'),
                                                                 edge_feature=self.plot['graph_features'].get('edge'),
                                                                 kind='geometric',
                                                                 **_geometric_params
                                                                 )
                # Set edges:
                _edge_x: list = []
                _edge_y: list = []
                for edge_x, edge_y in _graph.edges():
                    _x0, _y0 = _graph.nodes[edge_x]['pos']
                    _x1, _y1 = _graph.nodes[edge_y]['pos']
                    _edge_x.append(_x0)
                    _edge_x.append(_x1)
                    # _edge_x.append(None)
                    _edge_y.append(_y0)
                    _edge_y.append(_y1)
                    # _edge_y.append(None)
                # Set nodes:
                _node_x: list = []
                _node_y: list = []
                for node in _graph.nodes():
                    _x, _y = _graph.nodes[node]['pos']
                    _node_x.append(_x)
                    _node_y.append(_y)
                # Define edges (connections between nodes)
                self.plot['kwargs'].update({'x': _edge_x,
                                            'y': _edge_y,
                                            'mode': 'lines',
                                            'line': dict(width=0.5, color='#888') if self.plot['kwargs'].get(
                                                'line') is None else self.plot['kwargs'].get('line'),
                                            'hoverinfo': 'none' if self.plot['kwargs'].get('hoverinfo') is None else
                                            self.plot['kwargs'].get('hoverinfo')
                                            })
                _data.append(PlotlyAdapter(plot=self.plot).scatter_gl())
                # Define nodes:
                self.plot['kwargs'].update({'x': _node_x,
                                            'y': _node_y,
                                            'mode': 'markers',
                                            'marker': dict(size=10,
                                                           line_width=2,
                                                           opacity=0.75,
                                                           color=_color_feature,
                                                           colorscale=_color_scale,
                                                           reversescale=False,
                                                           colorbar=dict(thickness=15,
                                                                         title=self.plot.get('color_feature'),
                                                                         titleside='right',
                                                                         xanchor='left'
                                                                         )
                                                           ) if self.plot['kwargs'].get('marker') is None else
                                            self.plot['kwargs'].get('marker'),
                                            'hoverinfo': 'text' if self.plot['kwargs'].get('hoverinfo') is None else
                                            self.plot['kwargs'].get('hoverinfo'),
                                            'line': None
                                            })
                _data.append(PlotlyAdapter(plot=self.plot).scatter_gl())
                self.fig = go.Figure(data=_data)
                self._show_plotly_offline()
            ##############
            # Pie Chart: #
            ##############
            elif self.plot.get('plot_type') == 'pie':
                self.plot.update({'melt': False})
                if self.plot.get('features') is None:
                    self.plot.update({'features': []})
                if len(self.plot.get('features')) == 0:
                    if self.plot['kwargs'].get('labels') is None:
                        raise DataVisualizerException('Neither features nor parameter "labels" found')
                    if self.plot['kwargs'].get('values') is None:
                        raise DataVisualizerException('Neither features nor parameter "values" found')
                    self.plot['kwargs'].update(
                        {'hole': 0.2 if self.plot['kwargs'].get('hole') is None else self.plot['kwargs'].get('hole'),
                         'labels': self.plot['kwargs'].get('labels'),
                         'parents': self.plot['kwargs'].get('parents'),
                         'values': self.plot['kwargs'].get('values'),
                         'textposition': 'inside' if self.plot['kwargs'].get('textposition') is None else self.plot[
                             'kwargs'].get('textposition'),
                         'marker': dict(colors=_color_feature) if self.plot['kwargs'].get('marker') is None else
                         self.plot['kwargs'].get('marker')
                         })
                    self.fig = PlotlyAdapter(plot=self.plot, offline=True).pie()
                    self._show_plotly_offline()
                else:
                    for i, ft in enumerate(self.plot.get('features'), start=1):
                        if self.plot.get('group_by') is None:
                            _freq: pd.DataFrame = self.df[ft].value_counts()
                            self.file_path_extension = self._trim(input_str=ft)
                            self.plot['kwargs'].update({'hole': 0.2 if self.plot['kwargs'].get('hole') is None else self.plot['kwargs'].get('hole'),
                                                        'labels': _freq.index.values.tolist(),
                                                        'parents': self.plot['kwargs'].get('parents'),
                                                        'values': _freq.values,
                                                        'textposition': 'inside' if self.plot['kwargs'].get(
                                                            'textposition') is None else self.plot['kwargs'].get(
                                                            'textposition'),
                                                        'marker': dict(colors=_color_feature) if self.plot[
                                                                                                     'kwargs'].get(
                                                            'marker') is None else self.plot['kwargs'].get('marker')
                                                        })
                            self.fig = PlotlyAdapter(plot=self.plot, offline=True).pie()
                            self._show_plotly_offline()
                        else:
                            for group in self.plot.get('group_by'):
                                if str(self.df[group].dtype).find('date') >= 0:
                                    self.df[group] = self.df[group].astype(str)
                                _unique: np.array = self.df[group].unique()
                                for j, val in enumerate(_unique, start=1):
                                    if j == 0:
                                        self.grouping = True
                                    elif j == len(_unique):
                                        self.grouping = False
                                    if val in INVALID_VALUES:
                                        _freq: pd.DataFrame = self.df.loc[self.df[group].isnull(), ft].value_counts()
                                    else:
                                        _freq: pd.DataFrame = self.df.loc[self.df[group] == val, ft].value_counts()
                                    self.title = title
                                    self.title = '{}<br></br>{}'.format(self.title, self._trim(input_str='{} ({}={})'.format(ft, group, val)))
                                    self.file_path_extension = self._trim(input_str='{}_{}_{}'.format(ft, group, j))
                                    self.plot['kwargs'].update({'hole': 0.2 if self.plot['kwargs'].get('hole') is None else self.plot['kwargs'].get('hole'),
                                                                'labels': _freq.index.values.tolist(),
                                                                'parents': self.plot['kwargs'].get('parents'),
                                                                'values': _freq.values,
                                                                'textposition': 'inside' if self.plot['kwargs'].get(
                                                                    'textposition') is None else self.plot[
                                                                    'kwargs'].get('textposition'),
                                                                'marker': dict(colors=_color_feature) if self.plot['kwargs'].get(
                                                                    'marker') is None else self.plot['kwargs'].get(
                                                                    'marker')
                                                                })
                                    self.fig = PlotlyAdapter(plot=self.plot, offline=True).pie()
                                    self._show_plotly_offline()
            ##############
            # Bar Chart: #
            ##############
            elif self.plot.get('plot_type') == 'bar':
                if self.plot.get('features') is None:
                    self.plot.update({'features': []})
                if len(self.plot.get('features')) == 0:
                    if self.plot['kwargs'].get('x') is None:
                        if self.plot['kwargs'].get('y') is None:
                            raise DataVisualizerException('Neither features nor parameter "x" or "y" found')
                    self.plot['kwargs'].update({'x': self.plot['kwargs'].get('x'),
                                                'y': self.plot['kwargs'].get('y'),
                                                'base': 0 if self.plot['kwargs'].get('base') is None else self.plot[
                                                    'kwargs'].get('base'),
                                                'showlegend': False if self.plot['kwargs'].get(
                                                    'showlegend') is None else self.plot['kwargs'].get('showlegend'),
                                                'width': 1 if self.plot['kwargs'].get('width') is None else self.plot[
                                                    'kwargs'].get('width'),
                                                'text': '' if self.plot['kwargs'].get('text') is None else self.plot[
                                                    'kwargs'].get('text'),
                                                'textposition': 'auto' if self.plot['kwargs'].get(
                                                    'textposition') is None else self.plot['kwargs'].get(
                                                    'textposition'),
                                                'marker': dict(color=_color_feature,
                                                               colorscale=_color_scale,
                                                               opacity=0.75
                                                               ) if self.plot['kwargs'].get('marker') is None else
                                                self.plot['kwargs'].get('marker')
                                                })
                    self.fig = PlotlyAdapter(plot=self.plot, offline=True).bar()
                    self._show_plotly_offline()
                else:
                    for i, ft in enumerate(self.plot.get('features'), start=1):
                        if self.plot.get('group_by') is None:
                            _freq: pd.Series = self.df[ft].value_counts(
                                normalize=False if self.plot['kwargs'].get('normalize') is None else self.plot[
                                    'kwargs'].get('normalize'),
                                sort=False if self.plot['kwargs'].get('sort') is None else self.plot['kwargs'].get(
                                    'sort'),
                                ascending=False if self.plot['kwargs'].get('ascending') is None else self.plot[
                                    'kwargs'].get('ascending'),
                                dropna=False if self.plot['kwargs'].get('dropna') is None else self.plot['kwargs'].get(
                                    'dropna')
                                )
                            self.plot['kwargs'].update({'x': _freq.index.values,
                                                        'y': _freq.values,
                                                        'base': 0 if self.plot['kwargs'].get('base') is None else
                                                        self.plot['kwargs'].get('base'),
                                                        'name': self._trim(input_str=ft),
                                                        'showlegend': True if self.plot['kwargs'].get(
                                                            'showlegend') is None else self.plot['kwargs'].get(
                                                            'showlegend'),
                                                        'width': 1 if self.plot['kwargs'].get('width') is None else
                                                        self.plot['kwargs'].get('width'),
                                                        'text': '' if self.plot['kwargs'].get('text') is None else
                                                        self.plot['kwargs'].get('text'),
                                                        'textposition': 'auto' if self.plot['kwargs'].get(
                                                            'textposition') is None else self.plot['kwargs'].get(
                                                            'textposition'),
                                                        'marker': dict(color=_color_feature,
                                                                       colorscale=_color_scale,
                                                                       opacity=0.75
                                                                       ) if self.plot['kwargs'].get(
                                                            'marker') is None else self.plot['kwargs'].get('marker')
                                                        })
                            if self.plot.get('melt'):
                                self.file_path_extension = '{}_{}'.format(self.file_path_extension, self._trim(input_str=ft))
                                self.plot['kwargs'].update({'share_yaxis': True})
                                _data.append(PlotlyAdapter(plot=self.plot, offline=True).bar())
                                if i == len(self.plot.get('features')):
                                    self.fig = _data
                                    self._show_plotly_offline()
                            else:
                                self.file_path_extension = self._trim(input_str=ft)
                                self.fig = PlotlyAdapter(plot=self.plot, offline=True).bar()
                                self._show_plotly_offline()
                        else:
                            self.plot['kwargs']['layout'].update({'barmode': 'stack' if self.plot['kwargs'][
                                                                                            'layout'].get(
                                'barmode') is None else self.plot['kwargs']['layout'].get('barmode')})
                            for j, group in enumerate(self.plot.get('group_by'), start=1):
                                if str(self.df[group].dtype).find('date') >= 0:
                                    self.df[group] = self.df[group].astype(str)
                                _unique: np.array = self.df[group].unique()
                                for k, val in enumerate(_unique, start=1):
                                    if k == 0:
                                        self.grouping = True
                                    elif k == len(_unique):
                                        self.grouping = False
                                    if val in INVALID_VALUES:
                                        _freq: pd.DataFrame = self.df.loc[self.df[group].isnull(), ft].value_counts()
                                    else:
                                        _freq: pd.DataFrame = self.df.loc[self.df[group] == val, ft].value_counts()
                                    self.plot['kwargs'].update({'x': _freq.index.values,
                                                                'y': _freq.values,
                                                                'base': 0 if self.plot['kwargs'].get(
                                                                    'base') is None else self.plot['kwargs'].get(
                                                                    'base'),
                                                                'name': self._trim(
                                                                    input_str='{} ({}={})'.format(ft, group, val)),
                                                                'showlegend': True if self.plot['kwargs'].get(
                                                                    'showlegend') is None else self.plot['kwargs'].get(
                                                                    'showlegend'),
                                                                'width': 1 if self.plot['kwargs'].get(
                                                                    'width') is None else self.plot['kwargs'].get(
                                                                    'width'),
                                                                'text': '' if self.plot['kwargs'].get(
                                                                    'text') is None else self.plot['kwargs'].get(
                                                                    'text'),
                                                                'textposition': 'auto' if self.plot['kwargs'].get(
                                                                    'textposition') is None else self.plot[
                                                                    'kwargs'].get('textposition'),
                                                                'marker': dict(color=_color_feature,
                                                                               colorscale=_color_scale,
                                                                               opacity=0.75
                                                                               ) if self.plot['kwargs'].get(
                                                                    'marker') is None else self.plot['kwargs'].get(
                                                                    'marker')
                                                                })
                                    if self.plot.get('melt'):
                                        _data.append(PlotlyAdapter(plot=self.plot, offline=True).bar())
                                        if k == len(_unique) and i == len(self.plot.get('features')):
                                            self.file_path_extension = self._trim(input_str='{}_{}'.format(ft, group))
                                            self.fig = _data
                                            self._show_plotly_offline()
                                    else:
                                        self.grouping = False
                                        _data.append(PlotlyAdapter(plot=self.plot, offline=True).bar())
                                        self.file_path_extension = self._trim(input_str='{}_{}_{}'.format(ft,
                                                                                                          group,
                                                                                                          k
                                                                                                          )
                                                                              )
                                        self.fig = _data
                                        self._show_plotly_offline()
                                        _data = []
            ####################
            # Histogram Chart: #
            ####################
            elif self.plot.get('plot_type') == 'hist':
                for i, ft in enumerate(self.plot.get('features'), start=1):
                    if self.plot.get('group_by') is None:
                        _freq: pd.Series = self.df[ft].value_counts(
                            normalize=False if self.plot['kwargs'].get('normalize') is None else self.plot[
                                'kwargs'].get('normalize'),
                            sort=False if self.plot['kwargs'].get('sort') is None else self.plot['kwargs'].get('sort'),
                            ascending=False if self.plot['kwargs'].get('ascending') is None else self.plot[
                                'kwargs'].get('ascending'),
                            dropna=False if self.plot['kwargs'].get('dropna') is None else self.plot['kwargs'].get(
                                'dropna')
                            )
                        self.plot['kwargs'].update({'x': self.df[ft].values if self.plot['kwargs'].get(
                            'x') is None or i > 1 else self.plot['kwargs'].get('x'),
                                                    'y': self.plot['kwargs'].get('y'),
                                                    'name': self._trim(input_str=ft),
                                                    'showlegend': True if self.plot['kwargs'].get(
                                                        'showlegend') is None else self.plot['kwargs'].get(
                                                        'showlegend'),
                                                    'histnorm': 'probability density' if self.plot['kwargs'].get(
                                                        'histnorm') is None else self.plot['kwargs'].get('histnorm'),
                                                    'opacity': 0.75 if self.plot['kwargs'].get('opacity') is None else
                                                    self.plot['kwargs'].get('opacity'),
                                                    'marker': dict(color=_color_feature,
                                                                   colorscale=_color_scale,
                                                                   opacity=0.75
                                                                   ) if self.plot['kwargs'].get('marker') is None else
                                                    self.plot['kwargs'].get('marker')
                                                    })
                        if self.plot.get('melt'):
                            self.file_path_extension = '{}_{}'.format(self.file_path_extension, self._trim(input_str=ft))
                            self.plot['kwargs'].update({'barmode': 'overlay',
                                                        'share_yaxis': True
                                                        })
                            _data.append(PlotlyAdapter(plot=self.plot, offline=True).histo())
                            if i == len(self.plot.get('features')):
                                self.fig = _data
                                self._show_plotly_offline()
                        else:
                            self.file_path_extension = self._trim(input_str=ft)
                            self.fig = PlotlyAdapter(plot=self.plot, offline=True).histo()
                            self._show_plotly_offline()
                    else:
                        for j, group in enumerate(self.plot.get('group_by'), start=1):
                            if str(self.df[group].dtype).find('date') >= 0:
                                self.df[group] = self.df[group].astype(str)
                            _unique: np.array = self.df[group].unique()
                            for k, val in enumerate(_unique, start=1):
                                if k == 0:
                                    self.grouping = True
                                elif k == len(_unique):
                                    self.grouping = False
                                if val in INVALID_VALUES:
                                    _x: np.array = self.df.loc[self.df[group].isnull(), ft].values
                                else:
                                    _x: np.array = self.df.loc[self.df[group] == val, ft].values
                                self.plot['kwargs'].update({'x': _x,
                                                            'y': self.plot['kwargs'].get('y'),
                                                            'name': self._trim(
                                                                input_str='{} ({}={})'.format(ft, group, val)),
                                                            'histnorm': 'percent' if self.plot['kwargs'].get(
                                                                'histnorm') is None else self.plot['kwargs'].get(
                                                                'histnorm'),
                                                            'opacity': 0.75 if self.plot['kwargs'].get(
                                                                'opacity') is None else self.plot['kwargs'].get(
                                                                'opacity'),
                                                            'marker': dict(color=_color_feature,
                                                                           colorscale=_color_scale,
                                                                           opacity=0.75
                                                                           ) if self.plot['kwargs'].get(
                                                                'marker') is None else self.plot['kwargs'].get('marker')
                                                            })
                                if self.plot.get('melt'):
                                    self.plot['kwargs'].update({'barmode': 'overlay',
                                                                'share_yaxis': True
                                                                })
                                    _data.append(PlotlyAdapter(plot=self.plot, offline=True).histo())
                                    if k == len(_unique) and i == len(self.plot.get('features')):
                                        self.file_path_extension = self._trim(input_str='{}_{}'.format(ft, group))
                                        self.fig = _data
                                        self._show_plotly_offline()
                                else:
                                    self.grouping = False
                                    _data.append(PlotlyAdapter(plot=self.plot, offline=True).histo())
                                    self.file_path_extension = self._trim(input_str='{}_{}_{}'.format(ft,
                                                                                                      group,
                                                                                                      k
                                                                                                      )
                                                                          )
                                    self.fig = _data
                                    self._show_plotly_offline()
                                    _data = []
            ###################
            # Distplot Chart: #
            ###################
            elif self.plot.get('plot_type') is 'dist':
                _ft: List[np.array] = []
                _labels: List[str] = []
                for i, ft in enumerate(self.plot.get('features'), start=1):
                    if self.plot.get('group_by') is None:
                        self.plot['kwargs'].update({'bin_size': 1.0 if self.plot['kwargs'].get(
                            'bin_size') is None or i > 1 else self.plot['kwargs'].get('bin_size'),
                                                    'curve_type': 'kde' if self.plot['kwargs'].get(
                                                        'curve_type') is None else self.plot['kwargs'].get(
                                                        'curve_type'),
                                                    'histnorm': 'probability density' if self.plot['kwargs'].get(
                                                        'histnorm') is None else self.plot['kwargs'].get('histnorm'),
                                                    'name': self._trim(input_str=ft),
                                                    'show_hist': True if self.plot['kwargs'].get(
                                                        'show_hist') is None else self.plot['kwargs'].get('show_hist'),
                                                    'show_curve': True if self.plot['kwargs'].get(
                                                        'show_curve') is None else self.plot['kwargs'].get(
                                                        'show_curve'),
                                                    'show_rug': True if self.plot['kwargs'].get('show_rug') is None else
                                                    self.plot['kwargs'].get('show_rug'),
                                                    'colorscale': _color_scale if self.plot['kwargs'].get(
                                                        'colorscale') is None else self.plot['kwargs'].get('colorscale')
                                                    })
                        if self.plot.get('melt'):
                            self.file_path_extension = '{}_{}'.format(self.file_path_extension, self._trim(input_str=ft))
                            _ft.append(self.df.loc[~self.df[ft].isnull(), ft].values)
                            _labels.append(ft)
                            if i == len(self.plot.get('features')):
                                self.plot['kwargs'].update({'hist_data': _ft if self.plot['kwargs'].get(
                                    'hist_data') is None or i > 1 else self.plot['kwargs'].get('hist_data'),
                                                            'group_labels': _labels if self.plot['kwargs'].get(
                                                                'group_labels') is None or i > 1 else self.plot[
                                                                'kwargs'].get('group_labels'),
                                                            })
                                self.fig = PlotlyAdapter(plot=self.plot, offline=True).distplot()
                                self._show_plotly_offline()
                        else:
                            self.plot['kwargs'].update({'hist_data': [self.df.loc[~self.df[ft].isnull(), ft].values] if
                            self.plot['kwargs'].get('hist_data') is None or i > 1 else self.plot['kwargs'].get(
                                'hist_data'),
                                                        'group_labels': [ft] if self.plot['kwargs'].get(
                                                            'group_labels') is None or i > 1 else self.plot[
                                                            'kwargs'].get('group_labels'),
                                                        })
                            self.file_path_extension = self._trim(input_str=ft)
                            self.fig = PlotlyAdapter(plot=self.plot, offline=True).distplot()
                            self._show_plotly_offline()
                    else:
                        for j, group in enumerate(self.plot.get('group_by'), start=1):
                            if str(self.df[group].dtype).find('date') >= 0:
                                self.df[group] = self.df[group].astype(str)
                            _unique: np.array = self.df[group].unique()
                            for k, val in enumerate(_unique, start=1):
                                if k == 0:
                                    self.grouping = True
                                elif k == len(_unique):
                                    self.grouping = False
                                if val in INVALID_VALUES:
                                    _x: np.array = self.df.loc[self.df[group].isnull(), ft].values
                                else:
                                    _x: np.array = self.df.loc[self.df[group] == val, ft].values
                                self.plot['kwargs'].update({'bin_size': 1.0 if self.plot['kwargs'].get(
                                    'bin_size') is None or i > 1 else self.plot['kwargs'].get('bin_size'),
                                                            'curve_type': 'kde' if self.plot['kwargs'].get(
                                                                'curve_type') is None else self.plot['kwargs'].get(
                                                                'curve_type'),
                                                            'histnorm': 'probability density' if self.plot[
                                                                                                     'kwargs'].get(
                                                                'histnorm') is None else self.plot['kwargs'].get(
                                                                'histnorm'),
                                                            'name': self._trim(
                                                                input_str='{} ({}={})'.format(ft, group, val)),
                                                            'show_hist': True if self.plot['kwargs'].get(
                                                                'show_hist') is None else self.plot['kwargs'].get(
                                                                'show_hist'),
                                                            'show_curve': True if self.plot['kwargs'].get(
                                                                'show_curve') is None else self.plot['kwargs'].get(
                                                                'show_curve'),
                                                            'show_rug': True if self.plot['kwargs'].get(
                                                                'show_rug') is None else self.plot['kwargs'].get(
                                                                'show_rug'),
                                                            'colorscale': _color_scale if self.plot['kwargs'].get(
                                                                'colorscale') is None else self.plot['kwargs'].get(
                                                                'colorscale')
                                                            })
                                if self.plot.get('melt'):
                                    _ft.append(_x)
                                    _labels.append(ft)
                                    if k == len(self.df[group].unique()) and i == len(self.plot.get('features')):
                                        self.plot['kwargs'].update({'hist_data': _ft if self.plot['kwargs'].get(
                                            'hist_data') is None or i > 1 else self.plot['kwargs'].get('hist_data'),
                                                                    'group_labels': _labels if self.plot['kwargs'].get(
                                                                        'group_labels') is None or i > 1 else self.plot[
                                                                        'kwargs'].get('group_labels')
                                                                    })
                                        self.file_path_extension = self._trim(input_str='{}_{}'.format(ft, group))
                                        self.fig = PlotlyAdapter(plot=self.plot, offline=True).distplot()
                                        self._show_plotly_offline()
                                else:
                                    if k == len(_unique):
                                        self.plot['kwargs'].update({'hist_data': [_x] if self.plot['kwargs'].get(
                                            'hist_data') is None or i > 1 else self.plot['kwargs'].get('hist_data'),
                                                                    'group_labels': [ft] if self.plot['kwargs'].get(
                                                                        'group_labels') is None or i > 1 else self.plot[
                                                                        'kwargs'].get('group_labels')
                                                                    })
                                        self.file_path_extension = self._trim(input_str='{}_{}_{}'.format(ft, group, k))
                                        self.fig = PlotlyAdapter(plot=self.plot, offline=True).distplot()
                                        self._show_plotly_offline()
            ###############################
            # Box-Whisker & Violin Chart: #
            ###############################
            elif self.plot.get('plot_type') in ['box', 'violin']:
                _sub_fig = None
                for i, ft in enumerate(self.plot.get('features'), start=1):
                    if self.plot.get('group_by') is None:
                        if self.plot.get('plot_type') == 'box':
                            self.plot['kwargs'].update({'x': self.plot['kwargs'].get('x'),
                                                        'y': self.df[ft].values,
                                                        'name': self._trim(input_str=ft),
                                                        'marker': dict(color=_color_feature,
                                                                       opacity=0.75
                                                                       ) if self.plot['kwargs'].get(
                                                            'marker') is None else self.plot['kwargs'].get('marker'),
                                                        'boxpoints': 'outliers' if self.plot['kwargs'].get(
                                                            'boxpoints') is None else self.plot['kwargs'].get(
                                                            'boxpoints'),
                                                        'jitter': 0.3 if self.plot['kwargs'].get('jitter') is None else
                                                        self.plot['kwargs'].get('jitter'),
                                                        'pointpos': 0.3 if self.plot['kwargs'].get(
                                                            'pointpos') is None else self.plot['kwargs'].get(
                                                            'pointpos'),
                                                        'showlegend': True if self.plot['kwargs'].get(
                                                            'showlegend') is None else self.plot['kwargs'].get(
                                                            'showlegend')
                                                        })
                            _sub_fig: go.Box = PlotlyAdapter(plot=self.plot, offline=True).box_whisker()
                        elif self.plot.get('plot_type') == 'violin':
                            self.plot['kwargs'].update({'x': self.plot['kwargs'].get('x'),
                                                        'y': self.df[ft].values,
                                                        'name': self._trim(input_str=ft),
                                                        'marker': dict(color=_color_feature,
                                                                       opacity=0.75
                                                                       ) if self.plot['kwargs'].get(
                                                            'marker') is None else self.plot['kwargs'].get('marker'),
                                                        'box': dict(visible=True) if self.plot['kwargs'].get(
                                                            'box') is None else self.plot['kwargs'].get('box'),
                                                        'meanline': dict(visible=True) if self.plot['kwargs'].get(
                                                            'meanline') is None else self.plot['kwargs'].get(
                                                            'meanline'),
                                                        'points': 'outliers' if self.plot['kwargs'].get(
                                                            'points') is None else self.plot['kwargs'].get('points')
                                                        })
                            _sub_fig: go.Violin = PlotlyAdapter(plot=self.plot, offline=True).violin()
                        if self.plot.get('melt'):
                            self.file_path_extension = '{}_melt'.format(self._trim(input_str=ft))
                            _data.append(_sub_fig)
                            if i == len(self.plot.get('features')):
                                self.fig = _data
                                self._show_plotly_offline()
                        else:
                            self.file_path_extension = self._trim(input_str=ft)
                            self.fig = _sub_fig
                            self._show_plotly_offline()
                    else:
                        for j, group in enumerate(self.plot.get('group_by'), start=1):
                            if ft in self.feature_types.get('categorical') and group in self.feature_types.get('categorical'):
                                continue
                            if ft == group:
                                if self.plot.get('melt'):
                                    if i == len(self.plot.get('features')):
                                        self.file_path_extension = self._trim(input_str='{}_{}'.format(ft, group))
                                        self.fig = _data
                                        self._show_plotly_offline()
                            else:
                                if str(self.df[group].dtype).find('date') >= 0:
                                    self.df[group] = self.df[group].astype(str)
                                _unique: np.array = self.df[group].unique()
                                for k, val in enumerate(_unique, start=1):
                                    if k == 1:
                                        self.grouping = True
                                    elif k == len(_unique):
                                        self.grouping = False
                                    if val in INVALID_VALUES:
                                        _values: np.array = self.df.loc[self.df[group].isnull(), ft].values
                                    else:
                                        _values: np.array = self.df.loc[self.df[group] == val, ft].values
                                    if self.plot.get('plot_type') == 'box':
                                        self.plot['kwargs'].update({'x': self.plot['kwargs'].get('x'),
                                                                    'y': _values,
                                                                    'name': self._trim(
                                                                        input_str='{} ({}={})'.format(ft, group, val)),
                                                                    'marker': dict(color=_color_feature,
                                                                                   # colorscale=_color_scale,
                                                                                   opacity=0.75
                                                                                   ) if self.plot['kwargs'].get(
                                                                        'marker') is None else self.plot['kwargs'].get(
                                                                        'marker'),
                                                                    'boxpoints': 'outliers' if self.plot['kwargs'].get(
                                                                        'boxpoints') is None else self.plot[
                                                                        'kwargs'].get('boxpoints'),
                                                                    'jitter': 0.3 if self.plot['kwargs'].get(
                                                                        'jitter') is None else self.plot['kwargs'].get(
                                                                        'jitter'),
                                                                    'pointpos': 0.3 if self.plot['kwargs'].get(
                                                                        'pointpos') is None else self.plot[
                                                                        'kwargs'].get('pointpos'),
                                                                    'showlegend': True if self.plot['kwargs'].get(
                                                                        'showlegend') is None else self.plot[
                                                                        'kwargs'].get('showlegend')
                                                                    })
                                        _sub_fig: go.Box = PlotlyAdapter(plot=self.plot, offline=True).box_whisker()
                                    elif self.plot.get('plot_type') == 'violin':
                                        self.plot['kwargs'].update({'x': self.plot['kwargs'].get('x'),
                                                                    'y': _values,
                                                                    'name': self._trim(
                                                                        input_str='{} ({}={})'.format(ft, group, val)),
                                                                    'marker': dict(color=_color_feature,
                                                                                   # colorscale=_color_scale,
                                                                                   opacity=0.75
                                                                                   ) if self.plot['kwargs'].get(
                                                                        'marker') is None else self.plot['kwargs'].get(
                                                                        'marker'),
                                                                    'box': dict(visible=True) if self.plot[
                                                                                                     'kwargs'].get(
                                                                        'box') is None else self.plot['kwargs'].get(
                                                                        'box'),
                                                                    'meanline': dict(visible=True) if self.plot[
                                                                                                          'kwargs'].get(
                                                                        'meanline') is None else self.plot[
                                                                        'kwargs'].get('meanline'),
                                                                    'points': 'outliers' if self.plot['kwargs'].get(
                                                                        'points') is None else self.plot['kwargs'].get(
                                                                        'points')
                                                                    })
                                        _sub_fig: go.Violin = PlotlyAdapter(plot=self.plot, offline=True).violin()
                                    _data.append(_sub_fig)
                                    if self.plot.get('melt'):
                                        if i == len(self.plot.get('features')):
                                            self.fig = _data
                                            if k == len(_unique):
                                                self.file_path_extension = self._trim(input_str='{}_{}'.format(ft, group))
                                                self._show_plotly_offline()
                                    else:
                                        self.grouping = False
                                        self.file_path_extension = self._trim(input_str='{}_{}_{}'.format(ft, group, k))
                                        self.fig = _data
                                        self._show_plotly_offline()
                                        _data = []
            ####################
            # Ridgeline Chart: #
            ####################
            # TODO: Add color functionality
            elif self.plot.get('plot_type') == 'ridgeline':
                if self.plot.get('time_features') is None:
                    raise DataVisualizerException('No time features found')
                if self.plot.get('group_by') is None:
                    for ft in self.plot.get('features'):
                        self.fig = go.Figure()
                        for i, time in enumerate(self.plot.get('time_features')):
                            _unique_val: np.array = np.sort(self.df[time].unique())
                            _colors: list = n_colors('rgb(5, 200, 200)', 'rgb(200, 10, 10)', len(_unique_val),
                                                     colortype='rgb')
                            for ext, time_val in enumerate(_unique_val):
                                if time_val in INVALID_VALUES:
                                    _x: list = self.df.loc[self.df[time].isnull(), ft].values
                                else:
                                    _x: list = self.df.loc[self.df[time] == time_val, ft].values
                                self.file_path_extension = self._trim(input_str='{}_{}_{}'.format(ft, time, ext))
                                self.plot['kwargs'].update({'x': _x,
                                                            'name': self._trim(
                                                                input_str='{} ({}={})'.format(ft, time, time_val)),
                                                            'side': 'positive' if self.plot['kwargs'].get(
                                                                'side') is None else self.plot['kwargs'].get('side'),
                                                            'points': False if self.plot['kwargs'].get(
                                                                'points') is None else self.plot['kwargs'].get(
                                                                'points'),
                                                            'line': dict(color=_colors[i]) if self.plot['kwargs'].get(
                                                                'line') is None else self.plot['kwargs'].get('line'),
                                                            'width': 3 if self.plot['kwargs'].get('width') is None else
                                                            self.plot['kwargs'].get('width')
                                                            })
                                self.fig.add_trace(PlotlyAdapter(plot=self.plot, offline=True).violin())
                            self.fig.update_traces(
                                orientation='h' if self.plot['kwargs'].get('orientation') is None else self.plot[
                                    'kwargs'].get('orientation'))
                            self._show_plotly_offline()
                else:
                    for ft in self.plot.get('features'):
                        self.fig = go.Figure()
                        for group in self.plot.get('group_by'):
                            _unique_group_val: np.array = np.sort(self.df[group].unique())
                            for val in _unique_group_val:
                                if val in INVALID_VALUES:
                                    _df: pd.DataFrame = self.df.loc[self.df[group].isnull(), :]
                                else:
                                    _df: pd.DataFrame = self.df.loc[self.df[group] == val, :]
                                for i, time in enumerate(self.plot.get('time_features')):
                                    _unique_val: np.array = np.sort(_df[time].unique())
                                    _colors: list = n_colors('rgb(5, 200, 200)', 'rgb(200, 10, 10)', len(_unique_val),
                                                             colortype='rgb')
                                    for ext, time_val in enumerate(_unique_val):
                                        if time_val in INVALID_VALUES:
                                            _x: list = self.df.loc[_df[time].isnull(), ft].values
                                        else:
                                            _x: list = self.df.loc[_df[time] == time_val, ft].values
                                        self.file_path_extension = self._trim(input_str='{}_{}_{}'.format(ft, time, ext))
                                        self.plot['kwargs'].update({'x': _x,
                                                                    'name': self._trim(
                                                                        input_str='{} ({}={})'.format(ft, time,
                                                                                                      time_val)),
                                                                    'side': 'positive' if self.plot['kwargs'].get(
                                                                        'side') is None else self.plot['kwargs'].get(
                                                                        'side'),
                                                                    'points': False if self.plot['kwargs'].get(
                                                                        'points') is None else self.plot['kwargs'].get(
                                                                        'points'),
                                                                    'line': dict(color=_colors[i]) if self.plot[
                                                                                                          'kwargs'].get(
                                                                        'line') is None else self.plot['kwargs'].get(
                                                                        'line'),
                                                                    'width': 3 if self.plot['kwargs'].get(
                                                                        'width') is None else self.plot['kwargs'].get(
                                                                        'width')
                                                                    })
                                        self.fig.add_trace(PlotlyAdapter(plot=self.plot, offline=True).violin())
                                    self.fig.update_traces(
                                        orientation='h' if self.plot['kwargs'].get('orientation') is None else
                                        self.plot['kwargs'].get('orientation'))
                                    self._show_plotly_offline()
            ###########################
            # Histogram Decile Chart: #
            ###########################
            elif self.plot.get('plot_type') == 'hist_decile':
                for group in self.plot.get('group_by'):
                    _perc_table: pd.DataFrame = EasyExploreUtils().get_group_by_percentile(data=self.df,
                                                                                           group_by=group,
                                                                                           aggregate_by=self.plot.get('features'),
                                                                                           aggregation='median' if self.plot['kwargs'].get('aggregation') is None else self.plot['kwargs'].get('aggregation'),
                                                                                           percentiles=10 if self.plot['kwargs'].get('percentiles') is None else self.plot['kwargs'].get('percentiles'),
                                                                                           include_group=True if self.plot['kwargs'].get('include_group') is None else self.plot['kwargs'].get('include_group')
                                                                                           )
                    _min_table: pd.DataFrame = EasyExploreUtils().get_group_by_percentile(data=self.df,
                                                                                          group_by=group,
                                                                                          aggregate_by=self.plot.get('features'),
                                                                                          aggregation='min' if self.plot['kwargs'].get('error_bar_1') is None else self.plot['kwargs'].get('error_bar_1'),
                                                                                          percentiles=10 if self.plot['kwargs'].get('percentiles') is None else self.plot['kwargs'].get('percentiles'),
                                                                                          include_group=True if self.plot['kwargs'].get('include_group') is None else self.plot['kwargs'].get('include_group')
                                                                                          )
                    _max_table: pd.DataFrame = EasyExploreUtils().get_group_by_percentile(data=self.df,
                                                                                          group_by=group,
                                                                                          aggregate_by=self.plot.get('features'),
                                                                                          aggregation='max' if self.plot['kwargs'].get('error_bar_2') is None else self.plot['kwargs'].get('error_bar_2'),
                                                                                          percentiles=10 if self.plot['kwargs'].get('percentiles') is None else self.plot['kwargs'].get('percentiles'),
                                                                                          include_group=True if self.plot['kwargs'].get('include_group') is None else self.plot['kwargs'].get('include_group')
                                                                                          )
                    _multi: list = []
                    for col in _perc_table.keys():
                        self.plot['kwargs'].update({'y': _perc_table[col].values,
                                                    'base': 0 if self.plot['kwargs'].get('base') is None else self.plot[
                                                        'kwargs'].get('base'),
                                                    'name': col,  # '{} - {}'.format(group, col),
                                                    'showlegend': True if self.plot['kwargs'].get(
                                                        'showlegend') is None else self.plot['kwargs'].get(
                                                        'showlegend'),
                                                    'width': 1 if self.plot['kwargs'].get('width') is None else
                                                    self.plot['kwargs'].get('width'),
                                                    'text': '' if self.plot['kwargs'].get('text') is None else
                                                    self.plot['kwargs'].get('text'),
                                                    'textposition': 'auto' if self.plot['kwargs'].get(
                                                        'textposition') is None else self.plot['kwargs'].get(
                                                        'textposition'),
                                                    'error_y': dict(type='data',
                                                                    array=_max_table[col].values - _min_table[
                                                                        col].values),
                                                    'marker': dict(color=_color_feature,
                                                                   colorscale=_color_scale,
                                                                   opacity=0.75
                                                                   ) if self.plot['kwargs'].get('marker') is None else
                                                    self.plot['kwargs'].get('marker')
                                                    })
                        _multi.append(PlotlyAdapter(plot=self.plot, offline=True).bar())
                    self.plot['kwargs']['layout'].update({'barmode': 'group',
                                                          'xaxis': dict(tickmode='array',
                                                                        tickvals=[p for p in
                                                                                  range(0, _perc_table.shape[0], 1)],
                                                                        ticktext=['P{}\n{}'.format(str(p + 1), str(t))
                                                                                  for p, t in enumerate(
                                                                                _perc_table.index.values.tolist())]
                                                                        )
                                                          })
                    self.fig = go.Figure(_multi)
                    self._show_plotly_offline()
            ################
            # Multi Chart: #
            ################
            elif self.plot.get('plot_type') == 'multi':
                if self.plot['kwargs'].get('multi') is None:
                    raise DataVisualizerException('No parameter configuration for multi chart found')
                if isinstance(self.plot['kwargs'].get('multi'), dict):
                    _multi: dict = self.plot['kwargs'].get('multi')
                    self.fig = go.FigureWidget()
                    for m in _multi.keys():
                        for param in _multi.get(m).keys():
                            self.plot['kwargs'].update({param: _multi[m][param]})
                        if m.find('bar') >= 0:
                            self.fig.add_trace(PlotlyAdapter(plot=self.plot, offline=True).bar())
                        elif m.find('box') >= 0:
                            self.fig.add_trace(PlotlyAdapter(plot=self.plot, offline=True).box_whisker())
                        elif m.find('candlestick') >= 0:
                            self.fig.add_trace(PlotlyAdapter(plot=self.plot, offline=True).candlestick())
                        elif m.find('cluster3d') >= 0:
                            self.fig.add_trace(PlotlyAdapter(plot=self.plot, offline=True).mesh_3d())
                        elif m.find('contour') >= 0:
                            self.fig.add_trace(PlotlyAdapter(plot=self.plot, offline=True).contour())
                        elif m.find('contour_hist') >= 0:
                            self.fig.add_trace(PlotlyAdapter(plot=self.plot, offline=True).histogram_2d_contour())
                        elif m.find('dendro') >= 0:
                            self.fig.add_trace(PlotlyAdapter(plot=self.plot, offline=True).dendrogram())
                        elif m.find('density') >= 0:
                            self.fig.add_trace(PlotlyAdapter(plot=self.plot, offline=True).densitymapbox())
                        elif m.find('distplot') >= 0:
                            self.fig.add_trace(PlotlyAdapter(plot=self.plot, offline=True).distplot())
                        elif m.find('geo') >= 0:
                            self.fig.add_trace(PlotlyAdapter(plot=self.plot, offline=True).scatter_mapbox())
                        elif m.find('heat') >= 0:
                            self.fig.add_trace(PlotlyAdapter(plot=self.plot, offline=True).heat_map())
                        elif m.find('hist') >= 0:
                            self.fig.add_trace(PlotlyAdapter(plot=self.plot, offline=True).histo())
                        elif m.find('line') >= 0:
                            self.fig.add_trace(PlotlyAdapter(plot=self.plot, offline=True).line())
                        elif m.find('radar') >= 0:
                            self.fig.add_trace(PlotlyAdapter(plot=self.plot, offline=True).scatterpolar())
                        elif m.find('ridgeline') >= 0:
                            self.fig.add_trace(PlotlyAdapter(plot=self.plot, offline=True).ridgeline())
                        elif m.find('parcats') >= 0:
                            self.fig.add_trace(PlotlyAdapter(plot=self.plot, offline=True).parallel_category())
                        elif m.find('paarcoords') >= 0:
                            self.fig.add_trace(PlotlyAdapter(plot=self.plot, offline=True).parallel_coordinates())
                        elif m.find('pie') >= 0:
                            self.fig.add_trace(PlotlyAdapter(plot=self.plot, offline=True).pie())
                        elif m.find('scatter') >= 0:
                            self.fig.add_trace(PlotlyAdapter(plot=self.plot, offline=True).scatter_gl())
                        elif m.find('sunburst') >= 0:
                            self.fig.add_trace(PlotlyAdapter(plot=self.plot, offline=True).sunburst())
                        elif m.find('table') >= 0:
                            self.fig.add_trace(PlotlyAdapter(plot=self.plot, offline=True).table())
                        elif m.find('tree') >= 0:
                            self.fig.add_trace(PlotlyAdapter(plot=self.plot, offline=True).treemap())
                        elif m.find('violin') >= 0:
                            self.fig.add_trace(PlotlyAdapter(plot=self.plot, offline=True).violin())
                        else:
                            raise DataVisualizerException('Plot type ({}) not supported'.format(m))
                        self.plot['kwargs'] = dict(layout={})
                    self._show_plotly_offline()
                else:
                    raise DataVisualizerException(
                        'Parameter "multi" has to be a dictionary not a "{}" containing the specific chart-based configuration'.format(
                            type(self.plot['kwargs'].get('multi'))))
            else:
                raise DataVisualizerException(
                    'Plot type ({}) not supported interactively'.format(self.plot.get('plot_type')))

    def _show_plotly_offline(self):
        """
        Show plotly visualization in jupyter notebook
        """
        _fig: go.Figure = go.Figure(data=self.fig)
        _fig.update_layout(angularaxis=self.plot['kwargs']['layout'].get('angularaxis'),
                           annotations=self.plot['kwargs']['layout'].get('annotations'),
                           autosize=self.plot['kwargs']['layout'].get('autosize'),
                           bargap=self.plot['kwargs']['layout'].get('bargap'),
                           bargroupgap=self.plot['kwargs']['layout'].get('bargroupgap'),
                           barmode=self.plot['kwargs']['layout'].get('barmode'),
                           barnorm=self.plot['kwargs']['layout'].get('barnorm'),
                           boxgap=self.plot['kwargs']['layout'].get('boxgap'),
                           boxgroupgap=self.plot['kwargs']['layout'].get('boxgroupgap'),
                           boxmode=self.plot['kwargs']['layout'].get('boxmode'),
                           calendar=self.plot['kwargs']['layout'].get('calendar'),
                           clickmode=self.plot['kwargs']['layout'].get('clickmode'),
                           coloraxis=self.plot['kwargs']['layout'].get('coloraxis'),
                           colorscale=self.plot['kwargs']['layout'].get('colorscale'),
                           colorway=self.plot['kwargs']['layout'].get('colorway'),
                           datarevision=self.plot['kwargs']['layout'].get('datarevision'),
                           direction=self.plot['kwargs']['layout'].get('direction'),
                           dragmode=self.plot['kwargs']['layout'].get('dragmode'),
                           editrevision=self.plot['kwargs']['layout'].get('editrevision'),
                           extendfunnelareacolors=self.plot['kwargs']['layout'].get('extendfunnelareacolors'),
                           extendpiecolors=self.plot['kwargs']['layout'].get('extendpiecolors'),
                           extendsunburstcolors=self.plot['kwargs']['layout'].get('extendsunburstcolors'),
                           extendtreemapcolors=self.plot['kwargs']['layout'].get('extendtreemapcolors'),
                           font=self.plot['kwargs']['layout'].get('font'),
                           funnelareacolorway=self.plot['kwargs']['layout'].get('funnelareacolorway'),
                           funnelgap=self.plot['kwargs']['layout'].get('funnelgap'),
                           funnelgroupgap=self.plot['kwargs']['layout'].get('funnelgroupgap'),
                           funnelmode=self.plot['kwargs']['layout'].get('funnelmode'),
                           geo=self.plot['kwargs']['layout'].get('geo'),
                           grid=self.plot['kwargs']['layout'].get('grid'),
                           height=self.height,
                           hiddenlabels=self.plot['kwargs']['layout'].get('hiddenlabels'),
                           hiddenlabelssrc=self.plot['kwargs']['layout'].get('hiddenlabelssrc'),
                           hidesources=self.plot['kwargs']['layout'].get('hidesources'),
                           hoverdistance=self.plot['kwargs']['layout'].get('hoverdistance'),
                           hoverlabel=self.plot['kwargs']['layout'].get('hoverlabel'),
                           hovermode=self.plot['kwargs']['layout'].get('hovermode'),
                           images=self.plot['kwargs']['layout'].get('images'),
                           legend=self.plot['kwargs']['layout'].get('legend'),
                           mapbox=self.plot['kwargs']['layout'].get('mapbox'),
                           margin=self.plot['kwargs']['layout'].get('margin'),
                           meta=self.plot['kwargs']['layout'].get('meta'),
                           metasrc=self.plot['kwargs']['layout'].get('metasrc'),
                           modebar=self.plot['kwargs']['layout'].get('modebar'),
                           orientation=self.plot['kwargs']['layout'].get('orientation'),
                           paper_bgcolor=self.plot['kwargs']['layout'].get('paper_bgcolor'),
                           piecolorway=self.plot['kwargs']['layout'].get('piecolorway'),
                           plot_bgcolor=self.plot['kwargs']['layout'].get('plot_bgcolor'),
                           polar=self.plot['kwargs']['layout'].get('polar'),
                           radialaxis=self.plot['kwargs']['layout'].get('radialaxis'),
                           scene=self.plot['kwargs']['layout'].get('scene'),
                           selectdirection=self.plot['kwargs']['layout'].get('selectdirection'),
                           selectionrevision=self.plot['kwargs']['layout'].get('selectionrevision'),
                           separators=self.plot['kwargs']['layout'].get('separators'),
                           shapes=self.plot['kwargs']['layout'].get('shapes'),
                           showlegend=self.plot['kwargs']['layout'].get('showlegend'),
                           sliders=self.plot['kwargs']['layout'].get('sliders'),
                           spikedistance=self.plot['kwargs']['layout'].get('spikedistance'),
                           sunburstcolorway=self.plot['kwargs']['layout'].get('sunburstcolorway'),
                           template=self.plot['kwargs']['layout'].get('template'),
                           ternary=self.plot['kwargs']['layout'].get('ternary'),
                           title=dict(text=self.title, xanchor='left', yanchor='top'),
                           titlefont=self.plot['kwargs']['layout'].get('titlefont'),
                           transition=self.plot['kwargs']['layout'].get('transition'),
                           treemapcolorway=self.plot['kwargs']['layout'].get('treemapcolorway'),
                           uirevision=self.plot['kwargs']['layout'].get('uirevision'),
                           updatemenus=self.plot['kwargs']['layout'].get('updatemenus'),
                           violingap=self.plot['kwargs']['layout'].get('violingap'),
                           violingroupgap=self.plot['kwargs']['layout'].get('violingroupgap'),
                           waterfallgap=self.plot['kwargs']['layout'].get('waterfallgap'),
                           waterfallgroupgap=self.plot['kwargs']['layout'].get('waterfallgroupgap'),
                           waterfallmode=self.plot['kwargs']['layout'].get('waterfallmode'),
                           #width=self.plot['kwargs']['layout'].get('width'),
                           xaxis=self.plot['kwargs']['layout'].get('xaxis'),
                           xaxis2=self.plot['kwargs']['layout'].get('xaxis2'),
                           yaxis=self.plot['kwargs']['layout'].get('yaxis'),
                           yaxis2=self.plot['kwargs']['layout'].get('yaxis2')
                           )
        if self.plot.get('file_path') is not None:
            _original_file_path: str = self.plot.get('file_path')
            if len(self.plot.get('file_path')) > 0:
                if len(self.file_path_extension) > 0:
                    _file_type: str = self.plot.get('file_path').split('.')[-1]
                    _file_name: str = self.plot.get('file_path').split('/')[-1].split('.')[0]
                    _file_path: str = '/'.join(self.plot.get('file_path').split('/')[0:len(self.plot.get('file_path').split('/'))-1]).split('.')[0]
                    self.plot['file_path'] = '{}/{}{}.{}'.format(_file_path,
                                                                 '{}_'.format(_file_name) if len(_file_name) > 0 else _file_name,
                                                                 self.file_path_extension,
                                                                 _file_type
                                                                 )
                    self.file_path_extension = ''
                if not self.grouping:
                    Log(write=False).log('Saving plotly chart locally at: {}'.format(self.plot.get('file_path')))
                    PlotlyAdapter(plot=self.plot, offline=True, fig=_fig).save()
                self.plot['file_path'] = copy.deepcopy(_original_file_path)
            else:
                Log(write=False).log('Cannot save file locally because file path is empty')
        if self.render:
            Log(write=False).log('Rendering plotly chart offline ...')
            PlotlyAdapter(plot=self.plot, offline=True, fig=_fig).render()

    def _trim(self, input_str: str) -> str:
        """
        Trim strings to ensure a certain length

        :param input_str: str
            Input string

        :return: str
            Trimmed string if it reaches certain length
        """
        if len(input_str) > self.max_str_length:
            return input_str[0:self.max_str_length]
        else:
            return input_str

    def _unit_conversion(self, to_unit: str = 'pixel'):
        """
        Convert measurement units

        :param to_unit: str
            Name of the unit to convert
                -> in, inch: Inch
                -> px, pixel: Pixel
        """
        _units: List[str] = ['cm', 'in', 'px', 'centimeter', 'inch', 'px']
        if self.unit.lower() in _units:
            if 'in' in to_unit or 'inch' in to_unit:
                if 'cm' in self.unit or 'centimeter' in self.unit:
                    self.width = self.width * 0.3937
                    self.height = self.height * 0.3937
                elif 'px' in self.unit or 'pixel' in self.unit:
                    self.width = self.width * 0.010417
                    self.height = self.height * 0.010417
            elif 'px' in to_unit or 'pixel' in to_unit:
                if 'cm' in self.unit or 'centimeter' in self.unit:
                    self.width = self.width * 37.7952755906
                    self.height = self.height * 37.7952755906
                elif 'in' in self.unit or 'inch' in self.unit:
                    self.width = self.width * 300
                    self.height = self.height * 300
            else:
                raise DataVisualizerException('Unit ({}) not supported'.format(to_unit))
        else:
            raise DataVisualizerException('Unit ({}) not supported'.format(self.unit))

    def brushing_update_color(self, trace, points, state):
        """
        Update brushing color interactively

        :param trace:
        :param points:
        :param state:
        """
        # Update scatter selection
        self.fig.data[0].selectedpoints = points.point_inds
        # Update parcats colors
        new_color = np.zeros(len(self.df), dtype='uint8')
        new_color[points.point_inds] = 1
        self.fig.data[1].line.color = new_color

    def brushing_update_color_toogle(self, trace, points, state):
        """
        Update brushing color buttons interactively

        :param trace:
        :param points:
        :param state:
        """
        new_color = np.array(self.fig.data[0].marker.color)
        new_color[points.point_inds] = self.color_toggle.index
        with self.fig.batch_update():
            # Update scatter color
            self.fig.data[0].marker.color = new_color
            # Update parcats colors
            self.fig.data[1].line.color = new_color

    def load(self):
        """
        Load serialized plotly figure from json file and visualize it
        """
        _fig: dict = DataImporter(file_path=self.plot.get('file_path'), as_data_frame=False).file()
        if _fig.get('data') is None:
            raise DataVisualizerException('JSON file does not contain data for plotly figure')
        iplot(figure_or_data=go.FigureWidget(data=_fig.get('data'), layout=_fig.get('layout')),
              show_link=False if self.plot['kwargs'].get('show_link') is None else self.plot['kwargs'].get('show_link'),
              link_text='Export to plot.ly' if self.plot['kwargs'].get('link_text') is None else self.plot['kwargs'].get('link_text'),
              validate=True if self.plot['kwargs'].get('validate') is None else self.plot['kwargs'].get('validate'),
              image=self.plot['kwargs'].get('image'),
              filename=None,
              image_width=self.width,
              image_height=self.height,
              config=self.plot['kwargs'].get('config'),
              auto_play=True if self.plot['kwargs'].get('auto_play') is None else self.plot['kwargs'].get('auto_play'),
              animation_opts=self.plot['kwargs'].get('animation_opts')
              )

    def run(self):
        """
        Run visualization
        """
        if self.subplots is None:
            if self.df is None:
                raise DataVisualizerException('No data set found')
            if self.df.shape[0] == 0:
                raise DataVisualizerException('No cases found')
            if self.plot_type is None:
                raise DataVisualizerException('Plot type not found')
        if self.interactive:
            self._unit_conversion(to_unit='pixel')
            self._config_plotly_offline()
            self._run_plotly_offline()
            del self.df
            del self.plot
            del self.subplots
        else:
            raise NotImplementedError('Static visualization not supported')
