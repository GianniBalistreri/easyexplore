import json
import plotly.figure_factory as ff
import plotly.graph_objs as go

from .data_import_export import DataExporter, DataImporter
from .utils import Log
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot, plot
from plotly.subplots import make_subplots
from plotly.utils import PlotlyJSONEncoder
from typing import List

# TODO:
#  Add charts: a) Mesh3D  b) Funnel  c) Dendro-Heatmap

# Supported visualization methods
plots: List[str] = ['bar',
                    'box',
                    'candlestick',
                    'cluster',
                    'cluster3d',
                    'contour',
                    'contour_hist',
                    'dendro',
                    'density',
                    'distplot',
                    'funnel',
                    'funnel_area',
                    'geo',
                    'heat',
                    'hist',
                    'joint',
                    'line',
                    'multi',
                    'radar',
                    'ridgeline',
                    'parcats',
                    'parcoords',
                    'pie',
                    'scatter',
                    'scatter3d',
                    'sunburst',
                    'table',
                    'tree',
                    'violin'
                    ]


class PlotlyAdapterExceptions(Exception):
    """
    Class for handling exceptions for class PlotlyAdapter
    """
    pass


class PlotlyAdapter:
    """
    Class for wrapping up Plotly functionality
    """
    def __init__(self,
                 plot: dict,
                 offline: bool = True,
                 fig: go.Figure = None,
                 title: str = None,
                 width: int = 500,
                 height: int = 500,
                 **kwargs
                 ):
        """
        :param plot: dict
            Pre-configured parameters

        :param offline: bool
            Run plotly offline or online

        :param kwargs: dict
            Key word arguments for handling plotly
        """
        self.plot: dict = plot
        if 'kwargs' not in self.plot.keys():
            raise PlotlyAdapterExceptions('No key-word arguments regarding plotly figure found')
        self.offline: bool = offline
        if not self.offline:
            raise PlotlyAdapterExceptions('Online version not supported')
        self.fig = fig
        self.width: int = width if width >= 50 else 500
        self.height: int = height if height >= 50 else 500
        self.title: str = '' if title is None else title
        self.kwargs: dict = kwargs

    def bar(self) -> go.Bar:
        """
        Generate interactive bar chart using plotly

        :return: go.Bar
            Plotly graphic object containing the bar plot
        """
        return go.Bar(alignmentgroup=self.plot['kwargs'].get('alignmentgroup'),
                      base=self.plot['kwargs'].get('base'),
                      cliponaxis=self.plot['kwargs'].get('cliponaxis'),
                      constraintext=self.plot['kwargs'].get('constraintext'),
                      dx=self.plot['kwargs'].get('dx'),
                      dy=self.plot['kwargs'].get('dy'),
                      error_x=self.plot['kwargs'].get('error_x'),
                      error_y=self.plot['kwargs'].get('error_y'),
                      hoverinfo=self.plot['kwargs'].get('hoverinfo'),
                      hoverinfosrc=self.plot['kwargs'].get('hoverinfosrc'),
                      hoverlabel=self.plot['kwargs'].get('hoverlabel'),
                      hovertemplate=self.plot['kwargs'].get('hovertemplate'),
                      hovertemplatesrc=self.plot['kwargs'].get('hovertemplatesrc'),
                      hovertext=self.plot['kwargs'].get('hovertext'),
                      hovertextsrc=self.plot['kwargs'].get('hovertextsrc'),
                      ids=self.plot['kwargs'].get('ids'),
                      idssrc=self.plot['kwargs'].get('idssrc'),
                      insidetextanchor=self.plot['kwargs'].get('insidetextanchor'),
                      insidetextfont=self.plot['kwargs'].get('insidetextfont'),
                      legendgroup=self.plot['kwargs'].get('legendgroup'),
                      marker=self.plot['kwargs'].get('marker'),
                      meta=self.plot['kwargs'].get('meta'),
                      metasrc=self.plot['kwargs'].get('metasrc'),
                      name=self.plot['kwargs'].get('name'),
                      offset=self.plot['kwargs'].get('offset'),
                      offsetgroup=self.plot['kwargs'].get('offsetgroup'),
                      opacity=self.plot['kwargs'].get('opacity'),
                      orientation=self.plot['kwargs'].get('orientation'),
                      outsidetextfont=self.plot['kwargs'].get('outsidetextfont'),
                      r=self.plot['kwargs'].get('r'),
                      rsrc=self.plot['kwargs'].get('rsrc'),
                      selected=self.plot['kwargs'].get('selected'),
                      selectedpoints=self.plot['kwargs'].get('selectedpoints'),
                      showlegend=self.plot['kwargs'].get('showlegend'),
                      stream=self.plot['kwargs'].get('stream'),
                      t=self.plot['kwargs'].get('t'),
                      text=self.plot['kwargs'].get('text'),
                      textangle=self.plot['kwargs'].get('textangle'),
                      textfont=self.plot['kwargs'].get('textfont'),
                      textposition=self.plot['kwargs'].get('textposition'),
                      textpositionsrc=self.plot['kwargs'].get('textpositionsrc'),
                      textsrc=self.plot['kwargs'].get('textsrc'),
                      texttemplate=self.plot['kwargs'].get('texttemplate'),
                      texttemplatesrc=self.plot['kwargs'].get('texttemplatesrc'),
                      tsrc=self.plot['kwargs'].get('tsrc'),
                      uid=self.plot['kwargs'].get('uid'),
                      uirevision=self.plot['kwargs'].get('uirevision'),
                      unselected=self.plot['kwargs'].get('unselected'),
                      visible=self.plot['kwargs'].get('visible'),
                      width=self.plot['kwargs'].get('width'),
                      widthsrc=self.plot['kwargs'].get('widthsrc'),
                      x=self.plot['kwargs'].get('x'),
                      xaxis=self.plot['kwargs'].get('xaxis'),
                      xcalendar=self.plot['kwargs'].get('xcalendar'),
                      xsrc=self.plot['kwargs'].get('xsrc'),
                      y=self.plot['kwargs'].get('y'),
                      yaxis=self.plot['kwargs'].get('yaxis'),
                      ycalendar=self.plot['kwargs'].get('ycalendar'),
                      ysrc=self.plot['kwargs'].get('ysrc')
                      )

    def barpolar(self) -> go.Barpolar:
        """
        Generate interactive barpolar (radar) chart using plotly

        :return: go.Barpolar
            Plotly graphic object containing the barpolar chart
        """
        return go.Barpolar(base=self.plot['kwargs'].get('base'),
                           basesrc=self.plot['kwargs'].get('basesrc'),
                           customdata=self.plot['kwargs'].get('customdata'),
                           customdatasrc=self.plot['kwargs'].get('customdatasrc'),
                           dr=self.plot['kwargs'].get('dr'),
                           dtheta=self.plot['kwargs'].get('dtheta'),
                           hoverinfo=self.plot['kwargs'].get('hoverinfo'),
                           hoverinfosrc=self.plot['kwargs'].get('hoverinfosrc'),
                           hoverlabel=self.plot['kwargs'].get('hoverlabel'),
                           hoveron=self.plot['kwargs'].get('hoveron'),
                           hovertemplate=self.plot['kwargs'].get('hovertemplate'),
                           hovertemplatesrc=self.plot['kwargs'].get('hovertemplatesrc'),
                           hovertext=self.plot['kwargs'].get('hovertext'),
                           hovertextsrc=self.plot['kwargs'].get('hovertextsrc'),
                           ids=self.plot['kwargs'].get('ids'),
                           idssrc=self.plot['kwargs'].get('idssrc'),
                           legendgroup=self.plot['kwargs'].get('legendgroup'),
                           marker=self.plot['kwargs'].get('marker'),
                           meta=self.plot['kwargs'].get('meta'),
                           metasrc=self.plot['kwargs'].get('metasrc'),
                           name=self.plot['kwargs'].get('name'),
                           offset=self.plot['kwargs'].get('offset'),
                           offsetsrc=self.plot['kwargs'].get('offsetsrc'),
                           opacity=self.plot['kwargs'].get('opacity'),
                           parent=self.plot['kwargs'].get('parent'),
                           r=self.plot['kwargs'].get('r'),
                           r0=self.plot['kwargs'].get('r0'),
                           rsrc=self.plot['kwargs'].get('rsrc'),
                           selected=self.plot['kwargs'].get('selected'),
                           selectedpoints=self.plot['kwargs'].get('selectedpoints'),
                           showlegend=self.plot['kwargs'].get('showlegend'),
                           stream=self.plot['kwargs'].get('stream'),
                           subplot=self.plot['kwargs'].get('subplot'),
                           text=self.plot['kwargs'].get('text'),
                           textsrc=self.plot['kwargs'].get('textsrc'),
                           theta=self.plot['kwargs'].get('theta'),
                           theta0=self.plot['kwargs'].get('theta0'),
                           thetasrc=self.plot['kwargs'].get('thetasrc'),
                           thetaunit=self.plot['kwargs'].get('thetaunit'),
                           uid=self.plot['kwargs'].get('uid'),
                           uirevision=self.plot['kwargs'].get('uirevision'),
                           unselected=self.plot['kwargs'].get('unselected'),
                           visible=self.plot['kwargs'].get('visible'),
                           width=self.plot['kwargs'].get('width'),
                           widthsrc=self.plot['kwargs'].get('widthsrc')
                           )

    def box_whisker(self) -> go.Box:
        """
        Generate interactive box-whisker chart using plotly

        :return: go.Box
            Plotly graph object containing the box-whisker plot
        """
        return go.Box(alignmentgroup=self.plot['kwargs'].get('alignmentgroup'),
                      boxmean=self.plot['kwargs'].get('boxmean'),
                      boxpoints=self.plot['kwargs'].get('boxpoints'),
                      fillcolor=self.plot['kwargs'].get('fillcolor'),
                      hoverinfo=self.plot['kwargs'].get('hoverinfo'),
                      hoverinfosrc=self.plot['kwargs'].get('hoverinfosrc'),
                      hoverlabel=self.plot['kwargs'].get('hoverlabel'),
                      hoveron=self.plot['kwargs'].get('hoveron'),
                      hovertemplate=self.plot['kwargs'].get('hovertemplate'),
                      hovertemplatesrc=self.plot['kwargs'].get('hovertemplatesrc'),
                      hovertext=self.plot['kwargs'].get('hovertext'),
                      hovertextsrc=self.plot['kwargs'].get('hovertextsrc'),
                      ids=self.plot['kwargs'].get('ids'),
                      idssrc=self.plot['kwargs'].get('idssrc'),
                      jitter=self.plot['kwargs'].get('jitter'),
                      legendgroup=self.plot['kwargs'].get('legendgroup'),
                      line=self.plot['kwargs'].get('line'),
                      marker=self.plot['kwargs'].get('marker'),
                      meta=self.plot['kwargs'].get('meta'),
                      metasrc=self.plot['kwargs'].get('metasrc'),
                      name=self.plot['kwargs'].get('name'),
                      notched=self.plot['kwargs'].get('notched'),
                      notchwidth=self.plot['kwargs'].get('notchwidth'),
                      offsetgroup=self.plot['kwargs'].get('offsetgroup'),
                      opacity=self.plot['kwargs'].get('opacity'),
                      orientation=self.plot['kwargs'].get('orientation'),
                      pointpos=self.plot['kwargs'].get('pointpos'),  # -1.8,
                      selected=self.plot['kwargs'].get('selected'),
                      selectedpoints=self.plot['kwargs'].get('selectedpoints'),
                      showlegend=self.plot['kwargs'].get('showlegend'),
                      stream=self.plot['kwargs'].get('stream'),
                      text=self.plot['kwargs'].get('text'),
                      textsrc=self.plot['kwargs'].get('textsrc'),
                      uid=self.plot['kwargs'].get('uid'),
                      uirevision=self.plot['kwargs'].get('uirevision'),
                      unselected=self.plot['kwargs'].get('unselected'),
                      visible=self.plot['kwargs'].get('visible'),
                      whiskerwidth=self.plot['kwargs'].get('whiskerwidth'),
                      width=self.plot['kwargs'].get('width'),
                      x=self.plot['kwargs'].get('x'),
                      xaxis=self.plot['kwargs'].get('xaxis'),
                      xcalendar=self.plot['kwargs'].get('xcalendar'),
                      xsrc=self.plot['kwargs'].get('xsrc'),
                      y=self.plot['kwargs'].get('y'),
                      yaxis=self.plot['kwargs'].get('yaxis'),
                      ycalendar=self.plot['kwargs'].get('ycalendar'),
                      ysrc=self.plot['kwargs'].get('ysrc')
                      )

    def candlestick(self) -> go.Candlestick:
        """
        Generate interactive candlestick chart using plotly

        :return: go.Candlestick
            Plotly graph object containing the candlestick plot
        """
        return go.Candlestick(arg=self.plot['kwargs'].get('arg'),
                              close=self.plot['kwargs'].get('close'),
                              closesrc=self.plot['kwargs'].get('closesrc'),
                              customdata=self.plot['kwargs'].get('customdata'),
                              customdatasrc=self.plot['kwargs'].get('customdatasrc'),
                              decreasing=self.plot['kwargs'].get('decreasing'),
                              high=self.plot['kwargs'].get('high'),
                              highsrc=self.plot['kwargs'].get('highsrc'),
                              hoverinfo=self.plot['kwargs'].get('hoverinfo'),
                              hoverinfosrc=self.plot['kwargs'].get('hoverinfosrc'),
                              hoverlabel=self.plot['kwargs'].get('hoverlabel'),
                              hovertext=self.plot['kwargs'].get('hovertext'),
                              hovertextsrc=self.plot['kwargs'].get('hovertextsrc'),
                              ids=self.plot['kwargs'].get('ids'),
                              idssrc=self.plot['kwargs'].get('idssrc'),
                              increasing=self.plot['kwargs'].get('increasing'),
                              legendgroup=self.plot['kwargs'].get('legendgroup'),
                              line=self.plot['kwargs'].get('line'),
                              low=self.plot['kwargs'].get('low'),
                              lowsrc=self.plot['kwargs'].get('lowsrc'),
                              meta=self.plot['kwargs'].get('meta'),
                              metasrc=self.plot['kwargs'].get('metasrc'),
                              name=self.plot['kwargs'].get('name'),
                              opacity=self.plot['kwargs'].get('opacity'),
                              open=self.plot['kwargs'].get('open'),
                              opensrc=self.plot['kwargs'].get('opensrc'),
                              selectedpoints=self.plot['kwargs'].get('selectedpoints'),
                              showlegend=self.plot['kwargs'].get('showlegend'),
                              stream=self.plot['kwargs'].get('stream'),
                              text=self.plot['kwargs'].get('text'),
                              textsrc=self.plot['kwargs'].get('textsrc'),
                              uid=self.plot['kwargs'].get('uid'),
                              uirevision=self.plot['kwargs'].get('uirevision'),
                              visible=self.plot['kwargs'].get('visible'),
                              whiskerwidth=self.plot['kwargs'].get('whiskerwidth'),
                              x=self.plot['kwargs'].get('x'),
                              xaxis=self.plot['kwargs'].get('xaxis'),
                              xcalendar=self.plot['kwargs'].get('xcalendar'),
                              xsrc=self.plot['kwargs'].get('xsrc'),
                              yaxis=self.plot['kwargs'].get('yaxis')
                              )

    def choroplethmapbox(self) -> go.Choroplethmapbox:
        """
        Generate interactive choroplethmapbox using plotly

        :return: go.Choroplethmapbox
            Plotly graph object containing the choroplethmapbox
        """
        return go.Choroplethmapbox(autocolorscale=self.plot['kwargs'].get('autocolorscale'),
                                   below=self.plot['kwargs'].get('below'),
                                   coloraxis=self.plot['kwargs'].get('coloraxis'),
                                   colorbar=self.plot['kwargs'].get('colorbar'),
                                   colorscale=self.plot['kwargs'].get('colorscale'),
                                   customdata=self.plot['kwargs'].get('customdata'),
                                   customdatasrc=self.plot['kwargs'].get('customdatasrc'),
                                   geojson=self.plot['kwargs'].get('geojson'),
                                   hoverinfo=self.plot['kwargs'].get('hoverinfo'),
                                   hoverinfosrc=self.plot['kwargs'].get('hoverinfosrc'),
                                   hoverlabel=self.plot['kwargs'].get('hoverlabel'),
                                   hovertemplate=self.plot['kwargs'].get('hovertemplate'),
                                   hovertemplatesrc=self.plot['kwargs'].get('hovertemplatesrc'),
                                   hovertext=self.plot['kwargs'].get('hovertext'),
                                   hovertextsrc=self.plot['kwargs'].get('hovertextsrc'),
                                   ids=self.plot['kwargs'].get('ids'),
                                   idssrc=self.plot['kwargs'].get('idssrc'),
                                   locations=self.plot['kwargs'].get('locations'),
                                   locationssrc=self.plot['kwargs'].get('locationssrc'),
                                   marker=self.plot['kwargs'].get('marker'),
                                   meta=self.plot['kwargs'].get('meta'),
                                   metasrc=self.plot['kwargs'].get('metasrc'),
                                   name=self.plot['kwargs'].get('name'),
                                   reversescale=self.plot['kwargs'].get('reversescale'),
                                   selected=self.plot['kwargs'].get('selected'),
                                   selectedpoints=self.plot['kwargs'].get('selectedpoints'),
                                   showscale=self.plot['kwargs'].get('showscale'),
                                   stream=self.plot['kwargs'].get('stream'),
                                   subplot=self.plot['kwargs'].get('subplot'),
                                   text=self.plot['kwargs'].get('text'),
                                   textsrc=self.plot['kwargs'].get('textsrc'),
                                   uid=self.plot['kwargs'].get('uid'),
                                   uirevision=self.plot['kwargs'].get('uirevision'),
                                   unselected=self.plot['kwargs'].get('unselected'),
                                   visible=self.plot['kwargs'].get('visible'),
                                   z=self.plot['kwargs'].get('z'),
                                   zauto=self.plot['kwargs'].get('zauto'),
                                   zmax=self.plot['kwargs'].get('zmax'),
                                   zmid=self.plot['kwargs'].get('zmid'),
                                   zmin=self.plot['kwargs'].get('zmin'),
                                   zsrc=self.plot['kwargs'].get('zsrc')
                                   )

    def mesh_3d(self) -> go.Mesh3d:
        """
        Generate interactive 3-dimensional set of triangles

        :return go.Mesh3d
            Plotly graph object containing the 3d mesh plot
        """
        return go.Mesh3d()

    def contour(self) -> go.Contour:
        """
        Generate interactive contour chart using plotly

        :return: go.Contour
            Plotly graph object containing the contour plot
        """
        return go.Contour(autocolorscale=self.plot['kwargs'].get('autocolorscale'),
                          autocontour=self.plot['kwargs'].get('autocontour'),
                          coloraxis=self.plot['kwargs'].get('coloraxis'),
                          colorbar=self.plot['kwargs'].get('colorbar'),
                          colorscale=self.plot['kwargs'].get('colorscale'),
                          connectgaps=self.plot['kwargs'].get('connectgaps'),
                          contours=self.plot['kwargs'].get('contours'),
                          customdata=self.plot['kwargs'].get('customdata'),
                          customdatasrc=self.plot['kwargs'].get('customdatasrc'),
                          dx=self.plot['kwargs'].get('dx'),
                          dy=self.plot['kwargs'].get('dy'),
                          fillcolor=self.plot['kwargs'].get('fillcolor'),
                          hoverinfo=self.plot['kwargs'].get('hoverinfo'),
                          hoverinfosrc=self.plot['kwargs'].get('hoverinfosrc'),
                          hoverlabel=self.plot['kwargs'].get('hoverlabel'),
                          hovertemplate=self.plot['kwargs'].get('hovertemplate'),
                          hovertemplatesrc=self.plot['kwargs'].get('hovertemplatesrc'),
                          hovertext=self.plot['kwargs'].get('hovertext'),
                          hovertextsrc=self.plot['kwargs'].get('hovertextsrc'),
                          ids=self.plot['kwargs'].get('ids'),
                          idssrc=self.plot['kwargs'].get('idssrc'),
                          meta=self.plot['kwargs'].get('meta'),
                          metasrc=self.plot['kwargs'].get('metasrc'),
                          name=self.plot['kwargs'].get('name'),
                          ncontours=self.plot['kwargs'].get('ncontours'),
                          opacity=self.plot['kwargs'].get('opacity'),
                          reversescale=self.plot['kwargs'].get('reversescale'),
                          showlegend=self.plot['kwargs'].get('showlegend'),
                          showscale=self.plot['kwargs'].get('showscale'),
                          stream=self.plot['kwargs'].get('stream'),
                          text=self.plot['kwargs'].get('text'),
                          textsrc=self.plot['kwargs'].get('textsrc'),
                          transpose=self.plot['kwargs'].get('transpose'),
                          uid=self.plot['kwargs'].get('uid'),
                          uirevision=self.plot['kwargs'].get('uirevision'),
                          visible=self.plot['kwargs'].get('visible'),
                          x=self.plot['kwargs'].get('x'),
                          xaxis=self.plot['kwargs'].get('xaxis'),
                          xcalendar=self.plot['kwargs'].get('xcalendar'),
                          xsrc=self.plot['kwargs'].get('xsrc'),
                          xtype=self.plot['kwargs'].get('xtype'),
                          y=self.plot['kwargs'].get('y'),
                          yaxis=self.plot['kwargs'].get('yaxis'),
                          ycalendar=self.plot['kwargs'].get('ycalendar'),
                          ysrc=self.plot['kwargs'].get('ysrc'),
                          ytype=self.plot['kwargs'].get('ytype'),
                          z=self.plot['kwargs'].get('z'),
                          zauto=self.plot['kwargs'].get('zauto'),
                          zhoverformat=self.plot['kwargs'].get('zhoverformat'),
                          zmax=self.plot['kwargs'].get('zmax'),
                          zmid=self.plot['kwargs'].get('zmid'),
                          zmin=self.plot['kwargs'].get('zmin'),
                          zsrc=self.plot['kwargs'].get('zsrc')
                          )

    def dendrogram(self) -> ff:
        """
        Generate interactive dendrogram chart using plotly

        :return: ff
            Plotly graph object containing the dendrogram chart
        """
        return ff.create_dendrogram(X=self.plot['kwargs'].get('X'),
                                    orientation=self.plot['kwargs'].get('orientation'),
                                    labels=self.plot['kwargs'].get('labels'),
                                    colorscale=self.plot['kwargs'].get('colorscale'),
                                    distfun=self.plot['kwargs'].get('distfun'),
                                    linkagefun=self.plot['kwargs'].get('linkagefun'),
                                    hovertext=self.plot['kwargs'].get('hovertext'),
                                    color_threshold=self.plot['kwargs'].get('color_threshold')
                                    )

    def densitymapbox(self) -> go.Densitymapbox:
        """
        Generate interactive density mapbox using plotly

        :return: go.Densitymapbox
            Plotly graphic object containing the density map
        """
        return go.Densitymapbox(autocolorscale=self.plot['kwargs'].get('autocolorscale'),
                                below=self.plot['kwargs'].get('below'),
                                coloraxis=self.plot['kwargs'].get('coloraxis'),
                                colorbar=self.plot['kwargs'].get('colorbar'),
                                colorscale=self.plot['kwargs'].get('colorscale'),
                                customdata=self.plot['kwargs'].get('customdata'),
                                customdatasrc=self.plot['kwargs'].get('customdatasrc'),
                                hoverinfo=self.plot['kwargs'].get('hoverinfo'),
                                hoverinfosrc=self.plot['kwargs'].get('hoverinfosrc'),
                                hoverlabel=self.plot['kwargs'].get('hoverlabel'),
                                hovertemplate=self.plot['kwargs'].get('hovertemplate'),
                                hovertemplatesrc=self.plot['kwargs'].get('hovertemplatesrc'),
                                hovertext=self.plot['kwargs'].get('hovertext'),
                                hovertextsrc=self.plot['kwargs'].get('hovertextsrc'),
                                ids=self.plot['kwargs'].get('ids'),
                                idssrc=self.plot['kwargs'].get('idssrc'),
                                lat=self.plot['kwargs'].get('lat'),
                                latsrc=self.plot['kwargs'].get('latsrc'),
                                lon=self.plot['kwargs'].get('lon'),
                                lonsrc=self.plot['kwargs'].get('lonsrc'),
                                meta=self.plot['kwargs'].get('meta'),
                                metasrc=self.plot['kwargs'].get('metasrc'),
                                name=self.plot['kwargs'].get('name'),
                                opacity=self.plot['kwargs'].get('opacity'),
                                reversescale=self.plot['kwargs'].get('reversescale'),
                                showscale=self.plot['kwargs'].get('showscale'),
                                stream=self.plot['kwargs'].get('stream'),
                                subplot=self.plot['kwargs'].get('subplot'),
                                text=self.plot['kwargs'].get('text'),
                                textsrc=self.plot['kwargs'].get('textsrc'),
                                uid=self.plot['kwargs'].get('uid'),
                                uirevision=self.plot['kwargs'].get('uirevision'),
                                visible=self.plot['kwargs'].get('visible'),
                                z=self.plot['kwargs'].get('z'),
                                zauto=self.plot['kwargs'].get('zauto'),
                                zmax=self.plot['kwargs'].get('zmax'),
                                zmid=self.plot['kwargs'].get('zmid'),
                                zmin=self.plot['kwargs'].get('zmin'),
                                zsrc=self.plot['kwargs'].get('zsrc')
                                )

    def distplot(self) -> ff:
        """
        Generate interactive distplot using plotly

        :return: ff
            Plotly graphic object containing the distplot chart
        """
        return ff.create_distplot(hist_data=self.plot['kwargs'].get('hist_data'),
                                  group_labels=self.plot['kwargs'].get('group_labels'),
                                  bin_size=self.plot['kwargs'].get('bin_size'),
                                  curve_type=self.plot['kwargs'].get('curve_type'),
                                  colors=self.plot['kwargs'].get('colors'),
                                  rug_text=self.plot['kwargs'].get('rug_text'),
                                  histnorm=self.plot['kwargs'].get('histnorm'),
                                  show_hist=self.plot['kwargs'].get('show_hist'),
                                  show_curve=self.plot['kwargs'].get('show_curve'),
                                  show_rug=self.plot['kwargs'].get('show_rug'),
                                  )

    def funnel(self) -> go.Funnel:
        """
        Generate interactive funnel charts using plotly

        :return: go.Funnel
            Plotly graphic object containing the funnel chart
        """
        return go.Funnel(alignmentgroup=self.plot['kwargs'].get('alignmentgroup'),
                         cliponaxis=self.plot['kwargs'].get('cliponaxis'),
                         connector=self.plot['kwargs'].get('connector'),
                         constraintext=self.plot['kwargs'].get('constraintext'),
                         customdata=self.plot['kwargs'].get('customdata'),
                         customdatasrc=self.plot['kwargs'].get('customdatasrc'),
                         dx=self.plot['kwargs'].get('dx'),
                         dy=self.plot['kwargs'].get('dy'),
                         hoverinfo=self.plot['kwargs'].get('hoverinfo'),
                         hoverinfosrc=self.plot['kwargs'].get('hoverinfosrc'),
                         hoverlabel=self.plot['kwargs'].get('hoverlabel'),
                         hovertemplate=self.plot['kwargs'].get('hovertemplate'),
                         hovertemplatesrc=self.plot['kwargs'].get('hovertemplatesrc'),
                         hovertext=self.plot['kwargs'].get('hovertext'),
                         hovertextsrc=self.plot['kwargs'].get('hovertextsrc'),
                         ids=self.plot['kwargs'].get('ids'),
                         idssrc=self.plot['kwargs'].get('idssrc'),
                         insidetextanchor=self.plot['kwargs'].get('insidetextanchor'),
                         insidetextfont=self.plot['kwargs'].get('insidetextfont'),
                         legendgroup=self.plot['kwargs'].get('legendgroup'),
                         marker=self.plot['kwargs'].get('marker'),
                         meta=self.plot['kwargs'].get('meta'),
                         metasrc=self.plot['kwargs'].get('metasrc'),
                         name=self.plot['kwargs'].get('name'),
                         offset=self.plot['kwargs'].get('offset'),
                         offsetgroup=self.plot['kwargs'].get('offsetgroup'),
                         opacity=self.plot['kwargs'].get('opacity'),
                         orientation=self.plot['kwargs'].get('orientation'),
                         outsidetextfont=self.plot['kwargs'].get('outsidetextfont'),
                         selectedpoints=self.plot['kwargs'].get('selectedpoints'),
                         showlegend=self.plot['kwargs'].get('showlegend'),
                         stream=self.plot['kwargs'].get('stream'),
                         text=self.plot['kwargs'].get('text'),
                         textfont=self.plot['kwargs'].get('textfont'),
                         textinfo=self.plot['kwargs'].get('textinfo'),
                         textposition=self.plot['kwargs'].get('textposition'),
                         textpositionsrc=self.plot['kwargs'].get('textpositionsrc'),
                         textsrc=self.plot['kwargs'].get('textsrc'),
                         texttemplate=self.plot['kwargs'].get('texttemplate'),
                         texttemplatesrc=self.plot['kwargs'].get('texttemplatesrc'),
                         uid=self.plot['kwargs'].get('uid'),
                         uirevision=self.plot['kwargs'].get('uirevision'),
                         visible=self.plot['kwargs'].get('visible'),
                         width=self.plot['kwargs'].get('width'),
                         x=self.plot['kwargs'].get('x'),
                         x0=self.plot['kwargs'].get('x0'),
                         xaxis=self.plot['kwargs'].get('xaxis'),
                         xsrc=self.plot['kwargs'].get('xsrc'),
                         y=self.plot['kwargs'].get('y'),
                         y0=self.plot['kwargs'].get('y0'),
                         yaxis=self.plot['kwargs'].get('yaxis'),
                         ysrc=self.plot['kwargs'].get('ysrc')
                         )

    def funnel_area(self) -> go.Funnel:
        """
        Generate interactive funnel area charts using plotly

        :return: go.Funnel
            Plotly graphic object containing the funnel area chart
        """
        return go.Funnelarea(aspectratio=self.plot['kwargs'].get('aspectratio'),
                             baseratio=self.plot['kwargs'].get('baseratio'),
                             customdata=self.plot['kwargs'].get('customdata'),
                             customdatasrc=self.plot['kwargs'].get('customdatasrc'),
                             dlabel=self.plot['kwargs'].get('dlabel'),
                             domain=self.plot['kwargs'].get('domain'),
                             hoverinfo=self.plot['kwargs'].get('hoverinfo'),
                             hoverinfosrc=self.plot['kwargs'].get('hoverinfosrc'),
                             hoverlabel=self.plot['kwargs'].get('hoverlabel'),
                             hovertemplate=self.plot['kwargs'].get('hovertemplate'),
                             hovertemplatesrc=self.plot['kwargs'].get('hovertemplatesrc'),
                             hovertext=self.plot['kwargs'].get('hovertext'),
                             hovertextsrc=self.plot['kwargs'].get('hovertextsrc'),
                             ids=self.plot['kwargs'].get('ids'),
                             idssrc=self.plot['kwargs'].get('idssrc'),
                             insidetextfont=self.plot['kwargs'].get('insidetextfont'),
                             label0=self.plot['kwargs'].get('label0'),
                             labels=self.plot['kwargs'].get('labels'),
                             labelssrc=self.plot['kwargs'].get('labelssrc'),
                             legendgroup=self.plot['kwargs'].get('legendgroup'),
                             marker=self.plot['kwargs'].get('marker'),
                             meta=self.plot['kwargs'].get('meta'),
                             metasrc=self.plot['kwargs'].get('metasrc'),
                             name=self.plot['kwargs'].get('name'),
                             opacity=self.plot['kwargs'].get('opacity'),
                             scalegroup=self.plot['kwargs'].get('scalegroup'),
                             showlegend=self.plot['kwargs'].get('showlegend'),
                             stream=self.plot['kwargs'].get('stream'),
                             text=self.plot['kwargs'].get('text'),
                             textfont=self.plot['kwargs'].get('textfont'),
                             textinfo=self.plot['kwargs'].get('textinfo'),
                             textposition=self.plot['kwargs'].get('textposition'),
                             textpositionsrc=self.plot['kwargs'].get('textpositionsrc'),
                             textsrc=self.plot['kwargs'].get('textsrc'),
                             texttemplate=self.plot['kwargs'].get('texttemplate'),
                             texttemplatesrc=self.plot['kwargs'].get('texttemplatesrc'),
                             title=self.plot['kwargs'].get('title'),
                             uid=self.plot['kwargs'].get('uid'),
                             uirevision=self.plot['kwargs'].get('uirevision'),
                             values=self.plot['kwargs'].get('values'),
                             valuessrc=self.plot['kwargs'].get('valuessrc'),
                             visible=self.plot['kwargs'].get('visible')
                             )

    def generate_subplots(self, subplot_titles: List[str], rows: int, cols: int) -> go.Figure:
        """
        Generate subplots using plotly

        :param subplot_titles: List[str]
            Title of each subplot

        :param rows: int
            Number of rows of plotly subplots

        :param cols: int
            Number of columns of plotly subplots

        :return go.Figure
            Plotly subplots
        """
        _specs: List[List[dict]] = []
        _subplot_titles: List[str] = subplot_titles
        _rowspan: int = 1 if self.plot['kwargs'].get('rowspan') is None else self.plot['kwargs'].get('rowspan')
        _colspan: int = 1 if self.plot['kwargs'].get('colspan') is None else self.plot['kwargs'].get('colspan')
        for _ in range(0, rows, 1):
            _row: List[dict] = []
            for _ in range(0, cols, 1):
                _row.append({'type': self.plot.get('type'), 'rowspan': _rowspan, 'colspan': _colspan})
            _specs.append(_row)
        _vertical_spacing: float = 1.0 / rows
        _horizontal_spacing: float = 1.0 / cols
        _fig: make_subplots = make_subplots(rows=rows,
                                            cols=cols,
                                            shared_xaxes=False if self.plot['kwargs'].get('shared_xaxes') is None else self.plot['kwargs'].get('shared_xaxes'),
                                            shared_yaxes=False if self.plot['kwargs'].get('shared_yaxes') is None else self.plot['kwargs'].get('shared_yaxes'),
                                            start_cell='top-left' if self.plot['kwargs'].get('start_cell') is None else self.plot['kwargs'].get('start_cell'),
                                            print_grid=False if self.plot['kwargs'].get('print_grid') is None else self.plot['kwargs'].get('print_grid'),
                                            vertical_spacing=_vertical_spacing,
                                            horizontal_spacing=_horizontal_spacing,
                                            subplot_titles=_subplot_titles,
                                            column_widths=self.plot['kwargs'].get('column_widths'),
                                            row_heights=self.plot['kwargs'].get('row_heights'),
                                            specs=_specs,
                                            insets=self.plot['kwargs'].get('insets'),
                                            column_titles=self.plot['kwargs'].get('column_titles'),
                                            row_titles=self.plot['kwargs'].get('row_titles'),
                                            x_title=None if self.plot['kwargs'].get('x_title') is None else self.plot['kwargs'].get('x_title'),
                                            y_title=None if self.plot['kwargs'].get('y_title') is None else self.plot['kwargs'].get('y_title')
                                            )
        return _fig

    def heat_map(self) -> go.Heatmap:
        """
        Generate interactive heat map

        :return go.Heatmap
            Plotly graphic object containing the heat map
        """
        return go.Heatmap(autocolorscale=self.plot['kwargs'].get('autocolorscale'),
                          coloraxis=self.plot['kwargs'].get('coloraxis'),
                          colorscale=self.plot['kwargs'].get('colorscale'),
                          connectgaps=self.plot['kwargs'].get('connectgaps'),
                          dx=self.plot['kwargs'].get('dx'),
                          dy=self.plot['kwargs'].get('dy'),
                          hoverinfo=self.plot['kwargs'].get('hoverinfo'),
                          hoverinfosrc=self.plot['kwargs'].get('hoverinfosrc'),
                          hoverlabel=self.plot['kwargs'].get('hoverlabel'),
                          hovertemplate=self.plot['kwargs'].get('hovertemplate'),
                          hovertemplatesrc=self.plot['kwargs'].get('hovertemplatesrc'),
                          hovertext=self.plot['kwargs'].get('hovertext'),
                          hovertextsrc=self.plot['kwargs'].get('hovertextsrc'),
                          ids=self.plot['kwargs'].get('ids'),
                          idssrc=self.plot['kwargs'].get('idssrc'),
                          meta=self.plot['kwargs'].get('meta'),
                          metasrc=self.plot['kwargs'].get('metasrc'),
                          name=self.plot['kwargs'].get('name'),
                          opacity=self.plot['kwargs'].get('opacity'),
                          reversescale=self.plot['kwargs'].get('reversescale'),
                          showscale=self.plot['kwargs'].get('showscale'),
                          stream=self.plot['kwargs'].get('stream'),
                          text=self.plot['kwargs'].get('text'),
                          textsrc=self.plot['kwargs'].get('textsrc'),
                          transpose=self.plot['kwargs'].get('transpose'),
                          uid=self.plot['kwargs'].get('uid'),
                          uirevision=self.plot['kwargs'].get('uirevision'),
                          visible=self.plot['kwargs'].get('visible'),
                          x=self.plot['kwargs'].get('x'),
                          xaxis=self.plot['kwargs'].get('xaxis'),
                          xcalendar=self.plot['kwargs'].get('xcalendar'),
                          xgap=self.plot['kwargs'].get('xgap'),
                          xsrc=self.plot['kwargs'].get('xsrc'),
                          xtype=self.plot['kwargs'].get('xtype'),
                          y=self.plot['kwargs'].get('y'),
                          yaxis=self.plot['kwargs'].get('yaxis'),
                          ycalendar=self.plot['kwargs'].get('ycalendar'),
                          ygap=self.plot['kwargs'].get('ygap'),
                          ysrc=self.plot['kwargs'].get('ysrc'),
                          ytype=self.plot['kwargs'].get('ytype'),
                          z=self.plot['kwargs'].get('z'),
                          zauto=self.plot['kwargs'].get('zauto'),
                          zhoverformat=self.plot['kwargs'].get('zhoverformat'),
                          zmax=self.plot['kwargs'].get('zmax'),
                          zmid=self.plot['kwargs'].get('zmid'),
                          zmin=self.plot['kwargs'].get('zmin'),
                          zsmooth=self.plot['kwargs'].get('zsmooth'),
                          zsrc=self.plot['kwargs'].get('zsrc')
                          )

    def heat_map_annotated(self) -> ff:
        """
        Generate annotated heat map using plotly

        :return: ff
            Plotly graphic object containing the annotated heat map
        """
        return ff.create_annotated_heatmap(z=self.plot['kwargs'].get('z'),
                                           x=self.plot['kwargs'].get('x'),
                                           y=self.plot['kwargs'].get('y'),
                                           annotation_text=self.plot['kwargs'].get('annotation_text'),
                                           colorscale=self.plot['kwargs'].get('colorscale'),
                                           font_colors=self.plot['kwargs'].get('font_colors'),
                                           showscale=self.plot['kwargs'].get('showscale'),
                                           reversescale=self.plot['kwargs'].get('reversescale'),
                                           )

    def histo(self) -> go.Histogram:
        """
        Generate interactive histogram using plotly

        :return: go.Histogram
            Plotly graphic object containing the histogram
        """
        return go.Histogram(alignmentgroup=self.plot['kwargs'].get('alignmentgroup'),
                            autobinx=self.plot['kwargs'].get('autobinx'),
                            autobiny=self.plot['kwargs'].get('autobiny'),
                            bingroup=self.plot['kwargs'].get('bingroup'),
                            cumulative=self.plot['kwargs'].get('cumulative'),
                            histfunc=self.plot['kwargs'].get('histfunc'),
                            histnorm=self.plot['kwargs'].get('histnorm'),
                            hoverinfo=self.plot['kwargs'].get('hoverinfo'),
                            hoverinfosrc=self.plot['kwargs'].get('hoverinfosrc'),
                            hoverlabel=self.plot['kwargs'].get('hoverlabel'),
                            hovertemplate=self.plot['kwargs'].get('hovertemplate'),
                            hovertemplatesrc=self.plot['kwargs'].get('hovertemplatesrc'),
                            hovertext=self.plot['kwargs'].get('hovertext'),
                            hovertextsrc=self.plot['kwargs'].get('hovertextsrc'),
                            ids=self.plot['kwargs'].get('ids'),
                            idssrc=self.plot['kwargs'].get('idssrc'),
                            legendgroup=self.plot['kwargs'].get('legendgroup'),
                            marker=self.plot['kwargs'].get('marker'),
                            meta=self.plot['kwargs'].get('meta'),
                            metasrc=self.plot['kwargs'].get('metasrc'),
                            name=self.plot['kwargs'].get('name'),
                            nbinsx=self.plot['kwargs'].get('nbinsx'),
                            nbinsy=self.plot['kwargs'].get('nbinsy'),
                            offsetgroup=self.plot['kwargs'].get('offsetgroup'),
                            opacity=self.plot['kwargs'].get('opacity'),
                            orientation=self.plot['kwargs'].get('orientation'),
                            selected=self.plot['kwargs'].get('selected'),
                            selectedpoints=self.plot['kwargs'].get('selectedpoints'),
                            showlegend=self.plot['kwargs'].get('showlegend'),
                            stream=self.plot['kwargs'].get('stream'),
                            text=self.plot['kwargs'].get('text'),
                            textsrc=self.plot['kwargs'].get('textsrc'),
                            uid=self.plot['kwargs'].get('uid'),
                            uirevision=self.plot['kwargs'].get('uirevision'),
                            unselected=self.plot['kwargs'].get('unselected'),
                            visible=self.plot['kwargs'].get('visible'),
                            x=self.plot['kwargs'].get('x'),
                            xaxis=self.plot['kwargs'].get('xaxis'),
                            xbins=self.plot['kwargs'].get('xbins'),
                            xcalendar=self.plot['kwargs'].get('xcalendar'),
                            xsrc=self.plot['kwargs'].get('xsrc'),
                            y=self.plot['kwargs'].get('y'),
                            yaxis=self.plot['kwargs'].get('yaxis'),
                            ybins=self.plot['kwargs'].get('ybins'),
                            ycalendar=self.plot['kwargs'].get('ycalendar'),
                            ysrc=self.plot['kwargs'].get('ysrc')
                            )

    def histogram_2d_contour(self) -> go.Histogram2dContour:
        """
        Generate interactive histogram 2d contour chart using plotly

        :return: go.Histogram2dContour
            Plotly graphic object containing the histogram 2d contour plot
        """
        return go.Histogram2dContour(autobinx=self.plot['kwargs'].get('autobinx'),
                                     autobiny=self.plot['kwargs'].get('autobiny'),
                                     autocolorscale=self.plot['kwargs'].get('autocolorscale'),
                                     autocontour=self.plot['kwargs'].get('autocontour'),
                                     bingroup=self.plot['kwargs'].get('bingroup'),
                                     coloraxis=self.plot['kwargs'].get('coloraxis'),
                                     colorbar=self.plot['kwargs'].get('colorbar'),
                                     colorscale=self.plot['kwargs'].get('colorscale'),
                                     contours=self.plot['kwargs'].get('contours'),
                                     customdata=self.plot['kwargs'].get('customdata'),
                                     customdatasrc=self.plot['kwargs'].get('customdatasrc'),
                                     histfunc=self.plot['kwargs'].get('histfunc'),
                                     histnorm=self.plot['kwargs'].get('histnorm'),
                                     hoverinfo=self.plot['kwargs'].get('hoverinfo'),
                                     hoverinfosrc=self.plot['kwargs'].get('hoverinfosrc'),
                                     hoverlabel=self.plot['kwargs'].get('hoverlabel'),
                                     hovertemplate=self.plot['kwargs'].get('hovertemplate'),
                                     hovertemplatesrc=self.plot['kwargs'].get('hovertemplatesrc'),
                                     ids=self.plot['kwargs'].get('ids'),
                                     idssrc=self.plot['kwargs'].get('idssrc'),
                                     legendgroup=self.plot['kwargs'].get('legendgroup'),
                                     line=self.plot['kwargs'].get('line'),
                                     marker=self.plot['kwargs'].get('marker'),
                                     meta=self.plot['kwargs'].get('meta'),
                                     metasrc=self.plot['kwargs'].get('metasrc'),
                                     name=self.plot['kwargs'].get('name'),
                                     nbinsx=self.plot['kwargs'].get('nbinsx'),
                                     nbinsy=self.plot['kwargs'].get('nbinsy'),
                                     ncontours=self.plot['kwargs'].get('ncontours'),
                                     opacity=self.plot['kwargs'].get('opacity'),
                                     reversescale=self.plot['kwargs'].get('reversescale'),
                                     showlegend=self.plot['kwargs'].get('showlegend'),
                                     showscale=self.plot['kwargs'].get('showscale'),
                                     stream=self.plot['kwargs'].get('stream'),
                                     uid=self.plot['kwargs'].get('uid'),
                                     uirevision=self.plot['kwargs'].get('uirevision'),
                                     visible=self.plot['kwargs'].get('visible'),
                                     x=self.plot['kwargs'].get('x'),
                                     xaxis=self.plot['kwargs'].get('xaxis'),
                                     xbingroup=self.plot['kwargs'].get('xbingroup'),
                                     xbins=self.plot['kwargs'].get('xbins'),
                                     xcalendar=self.plot['kwargs'].get('xcalendar'),
                                     xsrc=self.plot['kwargs'].get('xsrc'),
                                     y=self.plot['kwargs'].get('y'),
                                     yaxis=self.plot['kwargs'].get('yaxis'),
                                     ybingroup=self.plot['kwargs'].get('ybingroup'),
                                     ybins=self.plot['kwargs'].get('ybins'),
                                     ycalendar=self.plot['kwargs'].get('ycalendar'),
                                     ysrc=self.plot['kwargs'].get('ysrc'),
                                     z=self.plot['kwargs'].get('z'),
                                     zauto=self.plot['kwargs'].get('zauto'),
                                     zhoverformat=self.plot['kwargs'].get('zhoverformat'),
                                     zmax=self.plot['kwargs'].get('zmax'),
                                     zmid=self.plot['kwargs'].get('zmid'),
                                     zmin=self.plot['kwargs'].get('zmin'),
                                     zsrc=self.plot['kwargs'].get('zsrc')
                                     )

    def line(self) -> go.Scatter:
        """
        Generate interactive line chart using plotly

        :return: go.Scatter
            Plotly graphic object containing the line plot
        """
        return go.Scatter(cliponaxis=self.plot['kwargs'].get('cliponaxis'),
                          connectgaps=self.plot['kwargs'].get('connectgaps'),
                          dx=self.plot['kwargs'].get('dx'),
                          dy=self.plot['kwargs'].get('dy'),
                          fill=self.plot['kwargs'].get('fill'),
                          fillcolor=self.plot['kwargs'].get('fillcolor'),
                          groupnorm=self.plot['kwargs'].get('groupnorm'),
                          hoverinfo=self.plot['kwargs'].get('hoverinfo'),
                          hoverinfosrc=self.plot['kwargs'].get('hoverinfosrc'),
                          hoverlabel=self.plot['kwargs'].get('hoverlabel'),
                          hoveron=self.plot['kwargs'].get('hoveron'),
                          hovertemplate=self.plot['kwargs'].get('hovertemplate'),
                          hovertemplatesrc=self.plot['kwargs'].get('hovertemplatesrc'),
                          hovertext=self.plot['kwargs'].get('hovertext'),
                          hovertextsrc=self.plot['kwargs'].get('hovertextsrc'),
                          ids=self.plot['kwargs'].get('ids'),
                          idssrc=self.plot['kwargs'].get('idssrc'),
                          legendgroup=self.plot['kwargs'].get('legendgroup'),
                          line=self.plot['kwargs'].get('line'),
                          marker=self.plot['kwargs'].get('marker'),
                          meta=self.plot['kwargs'].get('meta'),
                          metasrc=self.plot['kwargs'].get('metasrc'),
                          mode=self.plot['kwargs'].get('mode'),
                          name=self.plot['kwargs'].get('name'),
                          opacity=self.plot['kwargs'].get('opacity'),
                          orientation=self.plot['kwargs'].get('orientation'),
                          r=self.plot['kwargs'].get('r'),
                          rsrc=self.plot['kwargs'].get('rsrc'),
                          selected=self.plot['kwargs'].get('selected'),
                          selectedpoints=self.plot['kwargs'].get('selectedpoints'),
                          showlegend=self.plot['kwargs'].get('showlegend'),
                          stackgaps=self.plot['kwargs'].get('stackgaps'),
                          stackgroup=self.plot['kwargs'].get('stackgroup'),
                          stream=self.plot['kwargs'].get('stream'),
                          t=self.plot['kwargs'].get('t'),
                          text=self.plot['kwargs'].get('text'),
                          textfont=self.plot['kwargs'].get('textfont'),
                          textposition=self.plot['kwargs'].get('textposition'),
                          textpositionsrc=self.plot['kwargs'].get('textpositionsrc'),
                          textsrc=self.plot['kwargs'].get('textsrc'),
                          texttemplate=self.plot['kwargs'].get('texttemplate'),
                          texttemplatesrc=self.plot['kwargs'].get('texttemplatesrc'),
                          tsrc=self.plot['kwargs'].get('tsrc'),
                          uid=self.plot['kwargs'].get('uid'),
                          uirevision=self.plot['kwargs'].get('uirevision'),
                          unselected=self.plot['kwargs'].get('unselected'),
                          visible=self.plot['kwargs'].get('visible'),
                          x=self.plot['kwargs'].get('x'),
                          xaxis=self.plot['kwargs'].get('xaxis'),
                          xcalendar=self.plot['kwargs'].get('xcalendar'),
                          xsrc=self.plot['kwargs'].get('xsrc'),
                          y=self.plot['kwargs'].get('y'),
                          yaxis=self.plot['kwargs'].get('yaxis'),
                          ycalendar=self.plot['kwargs'].get('ycalendar'),
                          ysrc=self.plot['kwargs'].get('ysrc')
                          )

    def load(self):
        """
        Load serialized plotly figure from json file and visualize it
        """
        _fig: dict = DataImporter(file_path=self.file_path, as_data_frame=False).file()
        if _fig.get('data') is None:
            raise PlotlyAdapterExceptions('JSON file does not contain data for plotly figure')
        iplot(figure_or_data=go.FigureWidget(data=_fig.get('data'), layout=_fig.get('layout')),
              show_link=False if self.plot['kwargs'].get('show_link') is None else self.plot['kwargs'].get('show_link'),
              link_text='Export to plot.ly' if self.plot['kwargs'].get('link_text') is None else self.plot['kwargs'].get('link_text'),
              validate=True if self.plot['kwargs'].get('validate') is None else self.plot['kwargs'].get('validate'),
              image=self.plot['kwargs'].get('image'),
              filename=None,
              image_width=self.plot.get('width'),
              image_height=self.plot.get('height'),
              config=self.plot['kwargs'].get('config'),
              auto_play=True if self.plot['kwargs'].get('auto_play') is None else self.plot['kwargs'].get('auto_play'),
              animation_opts=self.plot['kwargs'].get('animation_opts')
              )

    def parallel_category(self) -> go.Parcats:
        """
        Generate interactive parallel category chart using plotly

        :return: go.Parcats
            Plotly graphic object containing the parallel category chart
        """
        return go.Parcats(arrangement=self.plot['kwargs'].get('arrangement'),
                          bundlecolors=self.plot['kwargs'].get('bundlecolors'),
                          counts=self.plot['kwargs'].get('counts'),
                          countssrc=self.plot['kwargs'].get('countssrc'),
                          dimensiondefaults=self.plot['kwargs'].get('dimensiondefaults'),
                          dimensions=self.plot['kwargs'].get('dimensions'),
                          domain=self.plot['kwargs'].get('domain'),
                          hoverinfo=self.plot['kwargs'].get('hoverinfo'),
                          hoveron=self.plot['kwargs'].get('hoveron'),
                          hovertemplate=self.plot['kwargs'].get('hovertemplate'),
                          labelfont=self.plot['kwargs'].get('labelfont'),
                          line=self.plot['kwargs'].get('line'),
                          meta=self.plot['kwargs'].get('meta'),
                          metasrc=self.plot['kwargs'].get('metasrc'),
                          name=self.plot['kwargs'].get('name'),
                          stream=self.plot['kwargs'].get('stream'),
                          tickfont=self.plot['kwargs'].get('tickfont'),
                          uid=self.plot['kwargs'].get('uid'),
                          uirevision=self.plot['kwargs'].get('uirevision'),
                          visible=self.plot['kwargs'].get('visible')
                          )

    def parallel_coordinates(self) -> go.Parcoords:
        """
        Generate interactive parallel coordinates chart using plotly

        :return go.Parcoords
            Plotly graphic object containing the parallel coordinates chart
        """
        return go.Parcoords(customdata=self.plot['kwargs'].get('customdata'),
                            customdatasrc=self.plot['kwargs'].get('customdatasrc'),
                            dimensiondefaults=self.plot['kwargs'].get('dimensiondefaults'),
                            dimensions=self.plot['kwargs'].get('dimensions'),
                            domain=self.plot['kwargs'].get('domain'),
                            ids=self.plot['kwargs'].get('ids'),
                            idssrc=self.plot['kwargs'].get('idssrc'),
                            labelangle=self.plot['kwargs'].get('labelangle'),
                            labelfont=self.plot['kwargs'].get('labelfont'),
                            labelside=self.plot['kwargs'].get('labelside'),
                            line=self.plot['kwargs'].get('line'),
                            meta=self.plot['kwargs'].get('meta'),
                            metasrc=self.plot['kwargs'].get('metasrc'),
                            name=self.plot['kwargs'].get('name'),
                            rangefont=self.plot['kwargs'].get('rangefont'),
                            stream=self.plot['kwargs'].get('stream'),
                            tickfont=self.plot['kwargs'].get('tickfont'),
                            uid=self.plot['kwargs'].get('uid'),
                            uirevision=self.plot['kwargs'].get('uirevision'),
                            visible=self.plot['kwargs'].get('visible')
                            )

    def pie(self) -> go.Pie:
        """
        Generate interactive pie chart using plotly

        :return go.Pie
            Plotly graphic object containing the pie plot
        """
        return go.Pie(automargin=self.plot['kwargs'].get('automargin'),
                      customdata=self.plot['kwargs'].get('customdata'),
                      customdatasrc=self.plot['kwargs'].get('customdatasrc'),
                      direction=self.plot['kwargs'].get('direction'),
                      dlabel=self.plot['kwargs'].get('dlabel'),
                      hole=self.plot['kwargs'].get('hole'),
                      hoverinfo=self.plot['kwargs'].get('hoverinfo'),
                      hoverinfosrc=self.plot['kwargs'].get('hoverinfosrc'),
                      hoverlabel=self.plot['kwargs'].get('hoverlabel'),
                      hovertemplate=self.plot['kwargs'].get('hovertemplate'),
                      hovertemplatesrc=self.plot['kwargs'].get('hovertemplatesrc'),
                      hovertext=self.plot['kwargs'].get('hovertext'),
                      hovertextsrc=self.plot['kwargs'].get('hovertextsrc'),
                      ids=self.plot['kwargs'].get('ids'),
                      idssrc=self.plot['kwargs'].get('idssrc'),
                      insidetextfont=self.plot['kwargs'].get('insidetextfont'),
                      label0=self.plot['kwargs'].get('label0'),
                      labels=self.plot['kwargs'].get('labels'),
                      labelssrc=self.plot['kwargs'].get('labelssrc'),
                      legendgroup=self.plot['kwargs'].get('legendgroup'),
                      marker=self.plot['kwargs'].get('marker'),
                      meta=self.plot['kwargs'].get('meta'),
                      metasrc=self.plot['kwargs'].get('metasrc'),
                      name=self.plot['kwargs'].get('name'),
                      opacity=self.plot['kwargs'].get('opacity'),
                      outsidetextfont=self.plot['kwargs'].get('outsidetextfont'),
                      pull=self.plot['kwargs'].get('pull'),
                      pullsrc=self.plot['kwargs'].get('pullsrc'),
                      rotation=self.plot['kwargs'].get('rotation'),
                      scalegroup=self.plot['kwargs'].get('scalegroup'),
                      showlegend=self.plot['kwargs'].get('showlegend'),
                      sort=self.plot['kwargs'].get('sort'),
                      stream=self.plot['kwargs'].get('stream'),
                      text=self.plot['kwargs'].get('text'),
                      textfont=self.plot['kwargs'].get('textfont'),
                      textinfo=self.plot['kwargs'].get('textinfo'),
                      textposition=self.plot['kwargs'].get('textposition'),
                      textpositionsrc=self.plot['kwargs'].get('textpositionsrc'),
                      textsrc=self.plot['kwargs'].get('textsrc'),
                      texttemplate=self.plot['kwargs'].get('texttemplate'),
                      texttemplatesrc=self.plot['kwargs'].get('texttemplatesrc'),
                      title=self.plot['kwargs'].get('title'),
                      titlefont=self.plot['kwargs'].get('titlefont'),
                      titleposition=self.plot['kwargs'].get('titleposition'),
                      uid=self.plot['kwargs'].get('uid'),
                      uirevision=self.plot['kwargs'].get('uirevision'),
                      values=self.plot['kwargs'].get('values'),
                      valuessrc=self.plot['kwargs'].get('valuessrc'),
                      visible=self.plot['kwargs'].get('visible')
                      )

    def ridgeline(self) -> go.Violin:
        """
        Generate interactive ridgeline chart using plotly

        :return go.Violin
            Plotly graphic object containing the ridgeline plot
        """
        return go.Violin(box_visible=self.plot['kwargs'].get('box_visible'),
                         legendgroup=self.plot['kwargs'].get('legendgroup'),
                         meanline_visible=self.plot['kwargs'].get('meanline_visible'),
                         name=self.plot['kwargs'].get('name'),
                         opacity=self.plot['kwargs'].get('opacity'),
                         pointpos=self.plot['kwargs'].get('pointpos'),
                         scalegroup=self.plot['kwargs'].get('scalegroup'),
                         showlegend=self.plot['kwargs'].get('showlegend'),
                         side=self.plot['kwargs'].get('side'),
                         x0=self.plot['kwargs'].get('x0'),
                         x=self.plot['kwargs'].get('x'),
                         y=self.plot['kwargs'].get('y')
                         )

    def render(self):
        """
        Render plotly chart offline
        """
        iplot(figure_or_data=go.FigureWidget(self.fig),
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

    def save(self):
        """
        Save plotly chart as local file
        """
        if self.plot.get('file_path').split('.')[-1] is 'json':
            self._write_plotly_json()
        elif self.plot.get('file_path').split('.')[-1] in ['html', 'png']:
            plot(figure_or_data=go.FigureWidget(self.fig),
                 show_link=False if self.plot['kwargs'].get('show_link') is None else self.plot['kwargs'].get('show_link'),
                 link_text='Export to plot.ly' if self.plot['kwargs'].get('link_text') is None else self.plot['kwargs'].get('link_text'),
                 validate=True if self.plot['kwargs'].get('validate') is None else self.plot['kwargs'].get('validate'),
                 image=self.plot['kwargs'].get('image'),
                 filename=self.plot.get('file_path'),
                 auto_open=False if self.plot['kwargs'].get('auto_open') is None else self.plot['kwargs'].get('auto_open'),
                 image_width=self.width,
                 image_height=self.height,
                 config=self.plot['kwargs'].get('config'),
                 auto_play=True if self.plot['kwargs'].get('auto_play') is None else self.plot['kwargs'].get('auto_play'),
                 animation_opts=self.plot['kwargs'].get('animation_opts')
                 )
        else:
            Log(write=False, level='error').log('File format (.{}) not supported for saving plotly charts'.format(self.plot.get('file_path').split('.')[-1]))

    def scatterpolar(self) -> go.Scatterpolar:
        """
        Generate interactive scatterpolar (radar) chart using plotly

        :return go.Scatterpolar
            Plotly graphic object containing the radar plot
        """
        return go.Scatterpolar(cliponaxis=self.plot['kwargs'].get('cliponaxis'),
                               connectgaps=self.plot['kwargs'].get('connectgaps'),
                               customdata=self.plot['kwargs'].get('customdata'),
                               customdatasrc=self.plot['kwargs'].get('customdatasrc'),
                               dr=self.plot['kwargs'].get('dr'),
                               dtheta=self.plot['kwargs'].get('dtheta'),
                               fill=self.plot['kwargs'].get('fill'),
                               fillcolor=self.plot['kwargs'].get('fillcolor'),
                               hoverinfo=self.plot['kwargs'].get('hoverinfo'),
                               hoverinfosrc=self.plot['kwargs'].get('hoverinfosrc'),
                               hoverlabel=self.plot['kwargs'].get('hoverlabel'),
                               hoveron=self.plot['kwargs'].get('hoveron'),
                               hovertemplate=self.plot['kwargs'].get('hovertemplate'),
                               hovertemplatesrc=self.plot['kwargs'].get('hovertemplatesrc'),
                               hovertext=self.plot['kwargs'].get('hovertext'),
                               hovertextsrc=self.plot['kwargs'].get('hovertextsrc'),
                               ids=self.plot['kwargs'].get('ids'),
                               idssrc=self.plot['kwargs'].get('idssrc'),
                               legendgroup=self.plot['kwargs'].get('legendgroup'),
                               line=self.plot['kwargs'].get('line'),
                               marker=self.plot['kwargs'].get('marker'),
                               meta=self.plot['kwargs'].get('meta'),
                               metasrc=self.plot['kwargs'].get('metasrc'),
                               mode=self.plot['kwargs'].get('mode'),
                               name=self.plot['kwargs'].get('name'),
                               opacity=self.plot['kwargs'].get('opacity'),
                               r=self.plot['kwargs'].get('r'),
                               rsrc=self.plot['kwargs'].get('rsrc'),
                               selected=self.plot['kwargs'].get('selected'),
                               selectedpoints=self.plot['kwargs'].get('selectedpoints'),
                               showlegend=self.plot['kwargs'].get('showlegend'),
                               stream=self.plot['kwargs'].get('stream'),
                               subplot=self.plot['kwargs'].get('subplot'),
                               text=self.plot['kwargs'].get('text'),
                               textfont=self.plot['kwargs'].get('textfont'),
                               textposition=self.plot['kwargs'].get('textposition'),
                               textpositionsrc=self.plot['kwargs'].get('textpositionsrc'),
                               textsrc=self.plot['kwargs'].get('textsrc'),
                               texttemplate=self.plot['kwargs'].get('texttemplate'),
                               texttemplatesrc=self.plot['kwargs'].get('texttemplatesrc'),
                               theta=self.plot['kwargs'].get('theta'),
                               theta0=self.plot['kwargs'].get('theta0'),
                               thetasrc=self.plot['kwargs'].get('thetasrc'),
                               thetaunit=self.plot['kwargs'].get('thetaunit'),
                               uid=self.plot['kwargs'].get('uid'),
                               uirevision=self.plot['kwargs'].get('uirevision'),
                               unselected=self.plot['kwargs'].get('unselected'),
                               visible=self.plot['kwargs'].get('visible')
                               )

    def scatter(self) -> go.Scatter:
        """
        Generate interactive scatter plot

        :return go.Scatter
            Plotly graphic object containing the scatter chart
        """
        return go.Scatter(cliponaxis=self.plot['kwargs'].get('cliponaxis'),
                          connectgaps=self.plot['kwargs'].get('connectgaps'),
                          dx=self.plot['kwargs'].get('dx'),
                          dy=self.plot['kwargs'].get('dy'),
                          fill=self.plot['kwargs'].get('fill'),
                          fillcolor=self.plot['kwargs'].get('fillcolor'),
                          groupnorm=self.plot['kwargs'].get('groupnorm'),
                          hoverinfo=self.plot['kwargs'].get('hoverinfo'),
                          hoverinfosrc=self.plot['kwargs'].get('hoverinfosrc'),
                          hoverlabel=self.plot['kwargs'].get('hoverlabel'),
                          hoveron=self.plot['kwargs'].get('hoveron'),
                          hovertemplate=self.plot['kwargs'].get('hovertemplate'),
                          hovertemplatesrc=self.plot['kwargs'].get('hovertemplatesrc'),
                          hovertext=self.plot['kwargs'].get('hovertext'),
                          hovertextsrc=self.plot['kwargs'].get('hovertextsrc'),
                          ids=self.plot['kwargs'].get('ids'),
                          idssrc=self.plot['kwargs'].get('idssrc'),
                          legendgroup=self.plot['kwargs'].get('legendgroup'),
                          line=self.plot['kwargs'].get('line'),
                          marker=self.plot['kwargs'].get('marker'),
                          meta=self.plot['kwargs'].get('meta'),
                          metasrc=self.plot['kwargs'].get('metasrc'),
                          mode=self.plot['kwargs'].get('mode'),
                          name=self.plot['kwargs'].get('name'),
                          opacity=self.plot['kwargs'].get('opacity'),
                          orientation=self.plot['kwargs'].get('orientation'),
                          r=self.plot['kwargs'].get('r'),
                          rsrc=self.plot['kwargs'].get('rsrc'),
                          selected=self.plot['kwargs'].get('selected'),
                          selectedpoints=self.plot['kwargs'].get('selectedpoints'),
                          showlegend=self.plot['kwargs'].get('showlegend'),
                          stackgaps=self.plot['kwargs'].get('stackgaps'),
                          stackgroup=self.plot['kwargs'].get('stackgroup'),
                          stream=self.plot['kwargs'].get('stream'),
                          t=self.plot['kwargs'].get('t'),
                          text=self.plot['kwargs'].get('text'),
                          textfont=self.plot['kwargs'].get('textfont'),
                          textposition=self.plot['kwargs'].get('textposition'),
                          textpositionsrc=self.plot['kwargs'].get('textpositionsrc'),
                          textsrc=self.plot['kwargs'].get('textsrc'),
                          texttemplate=self.plot['kwargs'].get('texttemplate'),
                          texttemplatesrc=self.plot['kwargs'].get('texttemplatesrc'),
                          tsrc=self.plot['kwargs'].get('tsrc'),
                          uid=self.plot['kwargs'].get('uid'),
                          uirevision=self.plot['kwargs'].get('uirevision'),
                          unselected=self.plot['kwargs'].get('unselected'),
                          visible=self.plot['kwargs'].get('visible'),
                          x=self.plot['kwargs'].get('x'),
                          xaxis=self.plot['kwargs'].get('xaxis'),
                          xcalendar=self.plot['kwargs'].get('xcalendar'),
                          xsrc=self.plot['kwargs'].get('xsrc'),
                          y=self.plot['kwargs'].get('y'),
                          yaxis=self.plot['kwargs'].get('yaxis'),
                          ycalendar=self.plot['kwargs'].get('ycalendar'),
                          ysrc=self.plot['kwargs'].get('ysrc')
                          )

    def scatter3d(self) -> go.Scatter3d:
        """
        Generate interactive 3d scatter plot using plotly

        :return go.Scatter3d
            Plotly graphic object containing the 3d scatter chart
        """
        return go.Scatter3d(connectgaps=self.plot['kwargs'].get('connectgaps'),
                            error_x=self.plot['kwargs'].get('error_x'),
                            error_y=self.plot['kwargs'].get('error_y'),
                            error_z=self.plot['kwargs'].get('error_z'),
                            hoverinfo=self.plot['kwargs'].get('hoverinfo'),
                            hoverinfosrc=self.plot['kwargs'].get('hoverinfosrc'),
                            hoverlabel=self.plot['kwargs'].get('hoverlabel'),
                            hovertemplate=self.plot['kwargs'].get('hovertemplate'),
                            hovertemplatesrc=self.plot['kwargs'].get('hovertemplatesrc'),
                            hovertext=self.plot['kwargs'].get('hovertext'),
                            hovertextsrc=self.plot['kwargs'].get('hovertextsrc'),
                            ids=self.plot['kwargs'].get('ids'),
                            idssrc=self.plot['kwargs'].get('idssrc'),
                            legendgroup=self.plot['kwargs'].get('legendgroup'),
                            line=self.plot['kwargs'].get('line'),
                            marker=self.plot['kwargs'].get('marker'),
                            meta=self.plot['kwargs'].get('meta'),
                            metasrc=self.plot['kwargs'].get('metasrc'),
                            mode=self.plot['kwargs'].get('mode'),
                            name=self.plot['kwargs'].get('name'),
                            opacity=self.plot['kwargs'].get('opacity'),
                            projection=self.plot['kwargs'].get('projection'),
                            scene=self.plot['kwargs'].get('scene'),
                            showlegend=self.plot['kwargs'].get('showlegend'),
                            stream=self.plot['kwargs'].get('stream'),
                            surfaceaxis=self.plot['kwargs'].get('surfaceaxis'),
                            surfacecolor=self.plot['kwargs'].get('surfacecolor'),
                            text=self.plot['kwargs'].get('text'),
                            textfont=self.plot['kwargs'].get('textfont'),
                            textposition=self.plot['kwargs'].get('textposition'),
                            textpositionsrc=self.plot['kwargs'].get('textpositionsrc'),
                            textsrc=self.plot['kwargs'].get('textsrc'),
                            texttemplate=self.plot['kwargs'].get('texttemplate'),
                            texttemplatesrc=self.plot['kwargs'].get('texttemplatesrc'),
                            uid=self.plot['kwargs'].get('uid'),
                            uirevision=self.plot['kwargs'].get('uirevision'),
                            visible=self.plot['kwargs'].get('visible'),
                            x=self.plot['kwargs'].get('x'),
                            xcalendar=self.plot['kwargs'].get('xcalendar'),
                            xsrc=self.plot['kwargs'].get('xsrc'),
                            y=self.plot['kwargs'].get('y'),
                            ycalendar=self.plot['kwargs'].get('ycalendar'),
                            ysrc=self.plot['kwargs'].get('ysrc'),
                            z=self.plot['kwargs'].get('z'),
                            zcalendar=self.plot['kwargs'].get('zcalendar'),
                            zsrc=self.plot['kwargs'].get('zsrc')
                            )

    def scatter_gl(self) -> go.Scattergl:
        """
        Generate interactive scatter chart using web graphics library (WebGL)

        :return go.Scattergl
            Plotly graphic object containing the geo map
        """
        return go.Scattergl(connectgaps=self.plot['kwargs'].get('connectgaps'),
                            customdata=self.plot['kwargs'].get('customdata'),
                            customdatasrc=self.plot['kwargs'].get('customdatasrc'),
                            dx=self.plot['kwargs'].get('dx'),
                            dy=self.plot['kwargs'].get('dy'),
                            error_x=self.plot['kwargs'].get('error_x'),
                            error_y=self.plot['kwargs'].get('error_y'),
                            fill=self.plot['kwargs'].get('fill'),
                            fillcolor=self.plot['kwargs'].get('fillcolor'),
                            hoverinfo=self.plot['kwargs'].get('hoverinfo'),
                            hoverinfosrc=self.plot['kwargs'].get('hoverinfosrc'),
                            hoverlabel=self.plot['kwargs'].get('hoverlabel'),
                            hovertemplate=self.plot['kwargs'].get('hovertemplate'),
                            hovertemplatesrc=self.plot['kwargs'].get('hovertemplatesrc'),
                            hovertext=self.plot['kwargs'].get('hovertext'),
                            hovertextsrc=self.plot['kwargs'].get('hovertextsrc'),
                            ids=self.plot['kwargs'].get('ids'),
                            idssrc=self.plot['kwargs'].get('idssrc'),
                            legendgroup=self.plot['kwargs'].get('legendgroup'),
                            line=self.plot['kwargs'].get('line'),
                            marker=self.plot['kwargs'].get('marker'),
                            meta=self.plot['kwargs'].get('meta'),
                            metasrc=self.plot['kwargs'].get('metasrc'),
                            mode=self.plot['kwargs'].get('mode'),
                            name=self.plot['kwargs'].get('name'),
                            opacity=self.plot['kwargs'].get('opacity'),
                            selected=self.plot['kwargs'].get('selected'),
                            selectedpoints=self.plot['kwargs'].get('selectedpoints'),
                            showlegend=self.plot['kwargs'].get('showlegend'),
                            stream=self.plot['kwargs'].get('stream'),
                            text=self.plot['kwargs'].get('text'),
                            textfont=self.plot['kwargs'].get('textfont'),
                            textposition=self.plot['kwargs'].get('textposition'),
                            textpositionsrc=self.plot['kwargs'].get('textpositionsrc'),
                            textsrc=self.plot['kwargs'].get('textsrc'),
                            texttemplate=self.plot['kwargs'].get('texttemplate'),
                            texttemplatesrc=self.plot['kwargs'].get('texttemplatesrc'),
                            uid=self.plot['kwargs'].get('uid'),
                            uirevision=self.plot['kwargs'].get('uirevision'),
                            unselected=self.plot['kwargs'].get('unselected'),
                            visible=self.plot['kwargs'].get('visible'),
                            x=self.plot['kwargs'].get('x'),
                            x0=self.plot['kwargs'].get('x0'),
                            xaxis=self.plot['kwargs'].get('xaxis'),
                            xcalendar=self.plot['kwargs'].get('xcalendar'),
                            xsrc=self.plot['kwargs'].get('xsrc'),
                            y=self.plot['kwargs'].get('y'),
                            y0=self.plot['kwargs'].get('y0'),
                            yaxis=self.plot['kwargs'].get('yaxis'),
                            ycalendar=self.plot['kwargs'].get('ycalendar'),
                            ysrc=self.plot['kwargs'].get('ysrc')
                            )

    def scatter_geo(self) -> go.Scattergeo:
        """
        Generate interactive scatter geo map

        :return go.Scattergeo
            Plotly graphic object containing the geo map
        """
        return go.Scattergeo(connectgaps=self.plot['kwargs'].get('connectgaps'),
                             fill=self.plot['kwargs'].get('fill'),
                             fillcolor=self.plot['kwargs'].get('fillcolor'),
                             hoverinfo=self.plot['kwargs'].get('hoverinfo'),
                             hoverinfosrc=self.plot['kwargs'].get('hoverinfosrc'),
                             hoverlabel=self.plot['kwargs'].get('hoverlabel'),
                             hovertemplate=self.plot['kwargs'].get('hovertemplate'),
                             hovertemplatesrc=self.plot['kwargs'].get('hovertemplatesrc'),
                             hovertext=self.plot['kwargs'].get('hovertext'),
                             hovertextsrc=self.plot['kwargs'].get('hovertextsrc'),
                             ids=self.plot['kwargs'].get('ids'),
                             idssrc=self.plot['kwargs'].get('idssrc'),
                             lat=self.plot['kwargs'].get('lat'),
                             latsrc=self.plot['kwargs'].get('latsrc'),
                             legendgroup=self.plot['kwargs'].get('legendgroup'),
                             line=self.plot['kwargs'].get('line'),
                             locationmode=self.plot['kwargs'].get('locationmode'),
                             locations=self.plot['kwargs'].get('locations'),
                             locationssrc=self.plot['kwargs'].get('locationssrc'),
                             lon=self.plot['kwargs'].get('lon'),
                             lonsrc=self.plot['kwargs'].get('lonsrc'),
                             marker=self.plot['kwargs'].get('marker'),
                             meta=self.plot['kwargs'].get('meta'),
                             metasrc=self.plot['kwargs'].get('metasrc'),
                             mode=self.plot['kwargs'].get('mode'),
                             name=self.plot['kwargs'].get('name'),
                             opacity=self.plot['kwargs'].get('opacity'),
                             selected=self.plot['kwargs'].get('selected'),
                             selectedpoints=self.plot['kwargs'].get('selectedpoints'),
                             showlegend=self.plot['kwargs'].get('showlegend'),
                             text=self.plot['kwargs'].get('text'),
                             textfont=self.plot['kwargs'].get('textfont'),
                             textposition=self.plot['kwargs'].get('textposition'),
                             textpositionsrc=self.plot['kwargs'].get('textpositionsrc'),
                             textsrc=self.plot['kwargs'].get('textsrc'),
                             texttemplate=self.plot['kwargs'].get('texttemplate'),
                             texttemplatesrc=self.plot['kwargs'].get('texttemplatesrc'),
                             uid=self.plot['kwargs'].get('uid'),
                             uirevision=self.plot['kwargs'].get('uirevision'),
                             unselected=self.plot['kwargs'].get('unselected'),
                             visible=self.plot['kwargs'].get('visible')
                             )

    def scatter_mapbox(self) -> go.Scattermapbox:
        """
        Generate interactive scattermapbox charts using plotly

        :return go.Scattermapbox
            Plotly graphic object containing the scattermapbox chart
        """
        return go.Scattermapbox(below=self.plot['kwargs'].get('below'),
                                connectgaps=self.plot['kwargs'].get('connectgaps'),
                                customdata=self.plot['kwargs'].get('customdata'),
                                customdatasrc=self.plot['kwargs'].get('customdatasrc'),
                                fill=self.plot['kwargs'].get('fill'),
                                fillcolor=self.plot['kwargs'].get('fillcolor'),
                                hoverinfo=self.plot['kwargs'].get('hoverinfo'),
                                hoverinfosrc=self.plot['kwargs'].get('hoverinfosrc'),
                                hoverlabel=self.plot['kwargs'].get('hoverlabel'),
                                hovertemplate=self.plot['kwargs'].get('hovertemplate'),
                                hovertemplatesrc=self.plot['kwargs'].get('hovertemplatesrc'),
                                hovertext=self.plot['kwargs'].get('hovertext'),
                                hovertextsrc=self.plot['kwargs'].get('hovertextsrc'),
                                ids=self.plot['kwargs'].get('ids'),
                                idssrc=self.plot['kwargs'].get('idssrc'),
                                lat=self.plot['kwargs'].get('lat'),
                                latsrc=self.plot['kwargs'].get('latsrc'),
                                legendgroup=self.plot['kwargs'].get('legendgroup'),
                                line=self.plot['kwargs'].get('line'),
                                lon=self.plot['kwargs'].get('lon'),
                                lonsrc=self.plot['kwargs'].get('lonsrc'),
                                marker=self.plot['kwargs'].get('marker'),
                                meta=self.plot['kwargs'].get('meta'),
                                metasrc=self.plot['kwargs'].get('metasrc'),
                                mode=self.plot['kwargs'].get('mode'),
                                name=self.plot['kwargs'].get('name'),
                                opacity=self.plot['kwargs'].get('opacity'),
                                selected=self.plot['kwargs'].get('selected'),
                                selectedpoints=self.plot['kwargs'].get('selectedpoints'),
                                showlegend=self.plot['kwargs'].get('showlegend'),
                                stream=self.plot['kwargs'].get('stream'),
                                subplot=self.plot['kwargs'].get('subplot'),
                                text=self.plot['kwargs'].get('text'),
                                textfont=self.plot['kwargs'].get('textfont'),
                                textposition=self.plot['kwargs'].get('textposition'),
                                textsrc=self.plot['kwargs'].get('textsrc'),
                                texttemplate=self.plot['kwargs'].get('texttemplate'),
                                texttemplatesrc=self.plot['kwargs'].get('texttemplatesrc'),
                                uid=self.plot['kwargs'].get('uid'),
                                uirevision=self.plot['kwargs'].get('uirevision'),
                                unselected=self.plot['kwargs'].get('unselected'),
                                visible=self.plot['kwargs'].get('visible')
                                )

    def show_plotly_offline(self):
        """
        Show plotly visualization in jupyter notebook
        """
        self.fig.update_layout(angularaxis=self.plot['kwargs']['layout'].get('angularaxis'),
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
                               height=self.plot['kwargs']['layout'].get('height'),
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
                               width=self.plot['kwargs']['layout'].get('width'),
                               xaxis=self.plot['kwargs']['layout'].get('xaxis'),
                               xaxis2=self.plot['kwargs']['layout'].get('xaxis2'),
                               yaxis=self.plot['kwargs']['layout'].get('yaxis'),
                               yaxis2=self.plot['kwargs']['layout'].get('yaxis2')
                               )
        if self.plot.get('file_path') is not None:
            if self.plot.get('file_path').split('.')[-1] is 'json':
                self._write_plotly_json()
            elif self.plot.get('file_path').split('.')[-1] in ['html', 'png']:
                Log(write=False).log(msg='Saving plotly chart locally at: {}'.format(self.plot.get('file_path')))
                plot(figure_or_data=go.FigureWidget(self.fig),
                     show_link=False if self.plot['kwargs'].get('show_link') is None else self.plot['kwargs'].get('show_link'),
                     link_text='Export to plot.ly' if self.plot['kwargs'].get('link_text') is None else self.plot['kwargs'].get('link_text'),
                     validate=True if self.plot['kwargs'].get('validate') is None else self.plot['kwargs'].get('validate'),
                     image=self.plot['kwargs'].get('image'),
                     filename=self.plot.get('file_path'),
                     auto_open=False if self.plot['kwargs'].get('auto_open') is None else self.plot['kwargs'].get('auto_open'),
                     image_width=self.plot.get('width'),
                     image_height=self.plot.get('height'),
                     config=self.plot['kwargs'].get('config'),
                     auto_play=True if self.plot['kwargs'].get('auto_play') is None else self.plot['kwargs'].get('auto_play'),
                     animation_opts=self.plot['kwargs'].get('animation_opts')
                     )
            else:
                Log(write=False, level='error').log(msg='File format (.{}) not supported for saving plotly charts'.format(self.plot.get('file_path').split('.')[-1]))
        if self.plot.get('render'):
            Log(write=False).log(msg='Rendering plotly chart offline ...')
            iplot(figure_or_data=go.FigureWidget(self.fig),
                  show_link=False if self.plot['kwargs'].get('show_link') is None else self.plot['kwargs'].get('show_link'),
                  link_text='Export to plot.ly' if self.plot['kwargs'].get('link_text') is None else self.plot['kwargs'].get('link_text'),
                  validate=True if self.plot['kwargs'].get('validate') is None else self.plot['kwargs'].get('validate'),
                  image=self.plot['kwargs'].get('image'),
                  filename=None,
                  image_width=self.plot.get('width'),
                  image_height=self.plot.get('height'),
                  config=self.plot['kwargs'].get('config'),
                  auto_play=True if self.plot['kwargs'].get('auto_play') is None else self.plot['kwargs'].get('auto_play'),
                  animation_opts=self.plot['kwargs'].get('animation_opts')
                  )

    def subplots(self, subplot_titles: List[str], rows: int, cols: int) -> go.Figure:
        """
        Generate subplots using plotly

        :param subplot_titles: List[str]
            Title of each subplot

        :param rows: int
            Number of rows of plotly subplots

        :param cols: int
            Number of columns of plotly subplots

        :return go.Figure
            Plotly subplots
        """
        _specs: List[List[dict]] = []
        _subplot_titles: List[str] = subplot_titles
        _rowspan: int = 1 if self.plot['kwargs'].get('rowspan') is None else self.plot['kwargs'].get('rowspan')
        _colspan: int = 1 if self.plot['kwargs'].get('colspan') is None else self.plot['kwargs'].get('colspan')
        for _ in range(0, rows, 1):
            _row: List[dict] = []
            for _ in range(0, cols, 1):
                _row.append({'type': self.plot.get('type'), 'rowspan': _rowspan, 'colspan': _colspan})
            _specs.append(_row)
        _vertical_spacing: float = 1.0 / rows
        _horizontal_spacing: float = 1.0 / cols
        self.fig: make_subplots = make_subplots(rows=rows,
                                                cols=cols,
                                                shared_xaxes=False if self.plot['kwargs'].get('shared_xaxes') is None else self.plot['kwargs'].get('shared_xaxes'),
                                                shared_yaxes=False if self.plot['kwargs'].get('shared_yaxes') is None else self.plot['kwargs'].get('shared_yaxes'),
                                                start_cell='top-left' if self.plot['kwargs'].get('start_cell') is None else self.plot['kwargs'].get('start_cell'),
                                                print_grid=False if self.plot['kwargs'].get('print_grid') is None else self.plot['kwargs'].get('print_grid'),
                                                vertical_spacing=_vertical_spacing,
                                                horizontal_spacing=_horizontal_spacing,
                                                subplot_titles=_subplot_titles,
                                                column_widths=self.plot['kwargs'].get('column_widths'),
                                                row_heights=self.plot['kwargs'].get('row_heights'),
                                                specs=_specs,
                                                insets=self.plot['kwargs'].get('insets'),
                                                column_titles=self.plot['kwargs'].get('column_titles'),
                                                row_titles=self.plot['kwargs'].get('row_titles'),
                                                x_title=None if self.plot['kwargs'].get('x_title') is None else self.plot['kwargs'].get('x_title'),
                                                y_title=None if self.plot['kwargs'].get('y_title') is None else self.plot['kwargs'].get('y_title')
                                                )
        return self.fig

    def sunburst(self) -> go.Sunburst:
        """
        Generate interactive sunburst chart using plotly

        :return go.Sunburst
            Plotly graphic object containing the sunburst chart
        """
        return go.Sunburst(branchvalues=self.plot['kwargs'].get('branchvalues'),
                           count=self.plot['kwargs'].get('count'),
                           customdata=self.plot['kwargs'].get('customdata'),
                           customdatasrc=self.plot['kwargs'].get('customdatasrc'),
                           domain=self.plot['kwargs'].get('domain'),
                           hoverinfo=self.plot['kwargs'].get('hoverinfo'),
                           hoverinfosrc=self.plot['kwargs'].get('hoverinfosrc'),
                           hoverlabel=self.plot['kwargs'].get('hoverlabel'),
                           hoveron=self.plot['kwargs'].get('hoveron'),
                           hovertemplate=self.plot['kwargs'].get('hovertemplate'),
                           hovertemplatesrc=self.plot['kwargs'].get('hovertemplatesrc'),
                           hovertext=self.plot['kwargs'].get('hovertext'),
                           hovertextsrc=self.plot['kwargs'].get('hovertextsrc'),
                           ids=self.plot['kwargs'].get('ids'),
                           idssrc=self.plot['kwargs'].get('idssrc'),
                           insidetextfont=self.plot['kwargs'].get('insidetextfont'),
                           labels=self.plot['kwargs'].get('labels'),
                           labelssrc=self.plot['kwargs'].get('labelssrc'),
                           leaf=self.plot['kwargs'].get('leaf'),
                           level=self.plot['kwargs'].get('level'),
                           marker=self.plot['kwargs'].get('marker'),
                           maxdepth=self.plot['kwargs'].get('maxdepth'),
                           meta=self.plot['kwargs'].get('meta'),
                           metasrc=self.plot['kwargs'].get('metasrc'),
                           mode=self.plot['kwargs'].get('mode'),
                           name=self.plot['kwargs'].get('name'),
                           opacity=self.plot['kwargs'].get('opacity'),
                           outsidetextfont=self.plot['kwargs'].get('outsidetextfont'),
                           parent=self.plot['kwargs'].get('parent'),
                           stream=self.plot['kwargs'].get('stream'),
                           text=self.plot['kwargs'].get('text'),
                           textfont=self.plot['kwargs'].get('textfont'),
                           textsrc=self.plot['kwargs'].get('textsrc'),
                           texttemplate=self.plot['kwargs'].get('texttemplate'),
                           texttemplatesrc=self.plot['kwargs'].get('texttemplatesrc'),
                           uid=self.plot['kwargs'].get('uid'),
                           uirevision=self.plot['kwargs'].get('uirevision'),
                           values=self.plot['kwargs'].get('values'),
                           valuessrc=self.plot['kwargs'].get('valuessrc'),
                           visible=self.plot['kwargs'].get('visible')
                           )

    def table(self) -> go.Table:
        """
        Generate interactive table chart using plotly

        :return go.Table
            Plotly graphic object containing the table plot
        """
        return go.Table(cells=self.plot['kwargs'].get('cells'),
                        columnorder=self.plot['kwargs'].get('columnorder'),
                        columnwidth=self.plot['kwargs'].get('columnwidth'),
                        domain=self.plot['kwargs'].get('domain'),
                        header=self.plot['kwargs'].get('header'),
                        hoverinfo=self.plot['kwargs'].get('hoverinfo'),
                        hoverlabel=self.plot['kwargs'].get('hoverlabel'),
                        ids=self.plot['kwargs'].get('ids'),
                        idssrc=self.plot['kwargs'].get('idssrc'),
                        meta=self.plot['kwargs'].get('meta'),
                        metasrc=self.plot['kwargs'].get('metasrc'),
                        name=self.plot['kwargs'].get('name'),
                        stream=self.plot['kwargs'].get('stream'),
                        uid=self.plot['kwargs'].get('uid'),
                        uirevision=self.plot['kwargs'].get('uirevision'),
                        visible=self.plot['kwargs'].get('visible')
                        )

    def treemap(self) -> go.Treemap:
        """
        Generate interactive tree map using pltoly

        :return go.Treemap
            Plotly graphic object containing the tree map
        """
        return go.Treemap(branchvalues=self.plot['kwargs'].get('branchvalues'),
                          count=self.plot['kwargs'].get('count'),
                          customdata=self.plot['kwargs'].get('customdata'),
                          customdatasrc=self.plot['kwargs'].get('customdatasrc'),
                          domain=self.plot['kwargs'].get('domain'),
                          hoverinfo=self.plot['kwargs'].get('hoverinfo'),
                          hoverinfosrc=self.plot['kwargs'].get('hoverinfosrc'),
                          hoverlabel=self.plot['kwargs'].get('hoverlabel'),
                          hovertemplate=self.plot['kwargs'].get('hovertemplate'),
                          hovertemplatesrc=self.plot['kwargs'].get('hovertemplatesrc'),
                          hovertext=self.plot['kwargs'].get('hovertext'),
                          hovertextsrc=self.plot['kwargs'].get('hovertextsrc'),
                          ids=self.plot['kwargs'].get('ids'),
                          idssrc=self.plot['kwargs'].get('idssrc'),
                          insidetextfont=self.plot['kwargs'].get('insidetextfont'),
                          labels=self.plot['kwargs'].get('labels'),
                          labelssrc=self.plot['kwargs'].get('labelssrc'),
                          level=self.plot['kwargs'].get('level'),
                          marker=self.plot['kwargs'].get('marker'),
                          maxdepth=self.plot['kwargs'].get('maxdepth'),
                          meta=self.plot['kwargs'].get('meta'),
                          metasrc=self.plot['kwargs'].get('metasrc'),
                          name=self.plot['kwargs'].get('name'),
                          opacity=self.plot['kwargs'].get('opacity'),
                          outsidetextfont=self.plot['kwargs'].get('outsidetextfont'),
                          pathbar=self.plot['kwargs'].get('pathbar'),
                          stream=self.plot['kwargs'].get('stream'),
                          text=self.plot['kwargs'].get('text'),
                          textposition=self.plot['kwargs'].get('textposition'),
                          textsrc=self.plot['kwargs'].get('textsrc'),
                          texttemplate=self.plot['kwargs'].get('texttemplate'),
                          texttemplatesrc=self.plot['kwargs'].get('texttemplatesrc'),
                          tiling=self.plot['kwargs'].get('tiling'),
                          uid=self.plot['kwargs'].get('uid'),
                          uirevision=self.plot['kwargs'].get('uirevision'),
                          values=self.plot['kwargs'].get('values'),
                          valuessrc=self.plot['kwargs'].get('valuessrc'),
                          visible=self.plot['kwargs'].get('visible')
                          )

    def violin(self) -> go.Violin:
        """
        Generate interactive violin chart using plotly

        :return go.Violin
            Plotly graphic object containing the violin plot
        """
        return go.Violin(alignmentgroup=self.plot['kwargs'].get('alignmentgroup'),
                         bandwidth=self.plot['kwargs'].get('bandwidth'),
                         box=self.plot['kwargs'].get('box'),
                         customdata=self.plot['kwargs'].get('customdata'),
                         customdatasrc=self.plot['kwargs'].get('customdatasrc'),
                         fillcolor=self.plot['kwargs'].get('fillcolor'),
                         hoverinfo=self.plot['kwargs'].get('hoverinfo'),
                         hoverinfosrc=self.plot['kwargs'].get('hoverinfosrc'),
                         hoverlabel=self.plot['kwargs'].get('hoverlabel'),
                         hoveron=self.plot['kwargs'].get('hoveron'),
                         hovertemplate=self.plot['kwargs'].get('hovertemplate'),
                         hovertemplatesrc=self.plot['kwargs'].get('hovertemplatesrc'),
                         hovertext=self.plot['kwargs'].get('hovertext'),
                         hovertextsrc=self.plot['kwargs'].get('hovertextsrc'),
                         ids=self.plot['kwargs'].get('ids'),
                         idssrc=self.plot['kwargs'].get('idssrc'),
                         jitter=self.plot['kwargs'].get('jitter'),
                         legendgroup=self.plot['kwargs'].get('legendgroup'),
                         line=self.plot['kwargs'].get('line'),
                         marker=self.plot['kwargs'].get('marker'),
                         meanline=self.plot['kwargs'].get('meanline'),
                         meta=self.plot['kwargs'].get('meta'),
                         metasrc=self.plot['kwargs'].get('metasrc'),
                         name=self.plot['kwargs'].get('name'),
                         offsetgroup=self.plot['kwargs'].get('offsetgroup'),
                         opacity=self.plot['kwargs'].get('opacity'),
                         orientation=self.plot['kwargs'].get('orientation'),
                         pointpos=self.plot['kwargs'].get('pointpos'),
                         points=self.plot['kwargs'].get('points'),
                         scalegroup=self.plot['kwargs'].get('scalegroup'),
                         scalemode=self.plot['kwargs'].get('scalemode'),
                         selected=self.plot['kwargs'].get('selected'),
                         selectedpoints=self.plot['kwargs'].get('selectedpoints'),
                         showlegend=self.plot['kwargs'].get('showlegend'),
                         side=self.plot['kwargs'].get('side'),
                         span=self.plot['kwargs'].get('span'),
                         spanmode=self.plot['kwargs'].get('spanmode'),
                         stream=self.plot['kwargs'].get('stream'),
                         text=self.plot['kwargs'].get('text'),
                         textsrc=self.plot['kwargs'].get('textsrc'),
                         uid=self.plot['kwargs'].get('uid'),
                         uirevision=self.plot['kwargs'].get('uirevision'),
                         unselected=self.plot['kwargs'].get('unselected'),
                         visible=self.plot['kwargs'].get('visible'),
                         width=self.plot['kwargs'].get('width'),
                         x=self.plot['kwargs'].get('x'),
                         x0=self.plot['kwargs'].get('x0'),
                         xaxis=self.plot['kwargs'].get('xaxis'),
                         xsrc=self.plot['kwargs'].get('xsrc'),
                         y=self.plot['kwargs'].get('y'),
                         y0=self.plot['kwargs'].get('y0'),
                         yaxis=self.plot['kwargs'].get('yaxis'),
                         ysrc=self.plot['kwargs'].get('ysrc')
                         #box_visible=self.plot['kwargs'].get('box_visible'),
                         #meanline_visible=self.plot['kwargs'].get('meanline_visible'),
                         )

    def _write_plotly_json(self):
        """
        Export plotly graph data by writing json file
        """
        _data: dict = json.loads(json.dumps(self.fig.data, cls=PlotlyJSONEncoder))
        _layout: dict = json.loads(json.dumps(self.fig.layout, cls=PlotlyJSONEncoder))
        DataExporter(obj=json.dumps(dict(data=_data, layout=_layout)),
                     file_path=self.plot.get('file_path'),
                     create_dir=True,
                     overwrite=False
                     ).file()
