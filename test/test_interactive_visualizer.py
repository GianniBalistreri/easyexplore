import plotly.graph_objs as go
import unittest

from easyexplore.interactive_visualizer import PlotlyAdapter

PLOT: dict = dict(df=None, kwargs={})


class PlotlyAdapterTest(unittest.TestCase):
    """
    Unit test for class PlotlyAdapter
    """
    def test_bar(self):
        self.assertTrue(expr=isinstance(PlotlyAdapter(plot=PLOT).bar(), go.Bar))

    def test_barpolar(self):
        pass

    def test_box_whisker(self):
        self.assertTrue(expr=isinstance(PlotlyAdapter(plot=PLOT).box_whisker(), go.Box))

    def test_candlestick(self):
        self.assertTrue(expr=isinstance(PlotlyAdapter(plot=PLOT).candlestick(), go.Candlestick))

    def test_choroplethmapbox(self):
        self.assertTrue(expr=isinstance(PlotlyAdapter(plot=PLOT).choroplethmapbox(), go.Choroplethmapbox))

    def test_contour(self):
        self.assertTrue(expr=isinstance(PlotlyAdapter(plot=PLOT).contour(), go.Contour))

    def test_mesh_3d(self):
        self.assertTrue(expr=isinstance(PlotlyAdapter(plot=PLOT).mesh_3d(), go.Mesh3d))

    def test_dendrogram(self):
        pass

    def test_densitymapbox(self):
        self.assertTrue(expr=isinstance(PlotlyAdapter(plot=PLOT).densitymapbox(), go.Densitymapbox))

    def test_distplot(self):
        pass

    def test_funnel(self):
        self.assertTrue(expr=isinstance(PlotlyAdapter(plot=PLOT).funnel(), go.Funnel))

    def test_funnel_area(self):
        self.assertTrue(expr=isinstance(PlotlyAdapter(plot=PLOT).funnel_area(), go.Funnelarea))

    def test_generate_subplots(self):
        pass

    def test_heat_map(self):
        self.assertTrue(expr=isinstance(PlotlyAdapter(plot=PLOT).heat_map(), go.Heatmap))

    def test_heat_map_annotated(self):
        pass

    def test_histo(self):
        self.assertTrue(expr=isinstance(PlotlyAdapter(plot=PLOT).histo(), go.Histogram))

    def test_histo_2d_contour(self):
        self.assertTrue(expr=isinstance(PlotlyAdapter(plot=PLOT).histogram_2d_contour(), go.Histogram2dContour))

    def test_line(self):
        self.assertTrue(expr=isinstance(PlotlyAdapter(plot=PLOT).line(), go.Scatter))

    def test_load(self):
        pass

    def test_parallel_category(self):
        self.assertTrue(expr=isinstance(PlotlyAdapter(plot=PLOT).parallel_category(), go.Parcats))

    def test_parallel_coordinates(self):
        self.assertTrue(expr=isinstance(PlotlyAdapter(plot=PLOT).parallel_coordinates(), go.Parcoords))

    def test_pie(self):
        self.assertTrue(expr=isinstance(PlotlyAdapter(plot=PLOT).pie(), go.Pie))

    def test_ridgeline(self):
        self.assertTrue(expr=isinstance(PlotlyAdapter(plot=PLOT).ridgeline(), go.Violin))

    def test_render(self):
        pass

    def test_save(self):
        pass

    def test_scatter(self):
        self.assertTrue(expr=isinstance(PlotlyAdapter(plot=PLOT).scatter(), go.Scatter))

    def test_scatter3d(self):
        self.assertTrue(expr=isinstance(PlotlyAdapter(plot=PLOT).scatter3d(), go.Scatter3d))

    def test_scatter_gl(self):
        self.assertTrue(expr=isinstance(PlotlyAdapter(plot=PLOT).scatter_gl(), go.Scattergl))

    def test_scatter_geo(self):
        self.assertTrue(expr=isinstance(PlotlyAdapter(plot=PLOT).scatter_geo(), go.Scattergeo))

    def test_scatter_mapbox(self):
        self.assertTrue(expr=isinstance(PlotlyAdapter(plot=PLOT).scatter_mapbox(), go.Scattermapbox))

    def test_show_plotly_offline(self):
        pass

    def test_subplots(self):
        pass

    def test_sunburst(self):
        pass

    def test_table(self):
        self.assertTrue(expr=isinstance(PlotlyAdapter(plot=PLOT).table(), go.Table))

    def test_treemap(self):
        self.assertTrue(expr=isinstance(PlotlyAdapter(plot=PLOT).treemap(), go.Treemap))

    def test_violin(self):
        self.assertTrue(expr=isinstance(PlotlyAdapter(plot=PLOT).violin(), go.Violin))


if __name__ == '__main__':
    unittest.main()
