# EasyExplore

## Description:
Toolbox for easy and effective data exploration in Python. It is designed to work with Jupyter notebooks especially, but it can also be used in any python module.

## Table of Content:
1. Installation
2. Requirements
3. Introduction
    - Practical Usage
    - Utilities
        - DataImporter
        - DataExporter
    - DataExplorer
    - DataVisualizer


## 1. Installation:
You can easily install EasyExplore via pip install easyexplore on every operating system.

## 2. Requirements:
 - dask>=2.23.0
 - geojson>=2.5.0
 - ipywidgets>=0.5.1
 - joblib>=0.14.1
 - networkx>=2.2
 - numpy>=1.18.1
 - pandas>=1.1.0
 - plotly>=4.5.4
 - pyod>=0.7.7.1
 - psutil>=5.5.1
 - scipy>=1.4.1
 - scikit-learn>=0.23.1
 - sqlalchemy>=1.3.15
 - statsmodels>=0.9.0
 - wheel>=0.35.1
 - xlrd>=1.2.0

## 3. Introduction:
 - Practical Usage:
 
 EasyExplore is designed as a wrapper which helps Data Scientists to explore data more convinient and efficient.
 
 - Data Importer:
 
 You can easily import data set from several files as well as databases into a Pandas or dask DataFrame.
 
 - Data Exporter:
 
 You can easily import data set from Pandas DataFrame or other data objects into several files or databases.
 
 - Data Explorer:
 
 Explore your data set quickly and efficiently using the DataExplorer:

    -- Data Typing:

        Check whether represented data types of Pandas is equal to the real data types occuring in the data

    -- Data Health Check:

        Check the health of the data set in order to detecting, describing and visualizing ...
            ... the ammount of missing or invalid data vs. valid observations
            ... the amount of duplicated data
            ... the amount of invariant data

    -- Data Distribution:

        Describing and visualizing statistical distribution of ...
            ... categorical features
            ... continuous features
            ... date features

    -- Outlier Detection:

        Analyze outliers or anomalies of continuous features using univariate and multivariate methods:
            a) Univariate: Examines outlier values for each features separately using Inter-Quantile-Range (IQR)
            b) Multivarite: Examines outliers for each possible feature pair combined using a bunch of different machine learning algorithms. For further information just look at the PyOD packages documentation, because it is used under the hood.

    -- Categorical Breakdown Statistics:

        Descriptive statistics of continuous features grouped by values of each categorical feature in the data set:


    -- Correlation:

        Correlation analysis of continuous features. For analyzing multi-collinearity there is a partial correlation method implemented. The differences between marginal and partial correlations are inspected by visualizing the differences of the coefficients in a heat map as well.

    -- Geo Statistics:

        Descriptive statistics of continuous features grouped by values of each geo features in the data set. Additionally, there is a geo map (OpenStreetMap) generated to visualize statistical distribution.

    -- Text Analyzer:

        Analyze potential text features and generate various numerical features from those

- Data Visualizer:

Visualize your data set very easily using Plot.ly an interactive visualization library under the hood. The DataVisualizer is an efficient wrapper to abstract the most important elements for data exploration:

    -- Table Chart:
        Visualize matrix (Pandas DataFrame) as an interactive table

    -- Heat Map:
        Visualize value range of continuous features as heat map

    -- Geo Map:
        Visualize statistics of categorical and continuous features as interactive OpenStreetMap

    -- Contour Chart:
        Visualize value ranges of at least two continuous features as contours

    -- Pie Chart:
        Visualize occurances of values of categorical features as an interactive pie chart

    -- Bar Chart:
        Visualize occurances of values of categorical features as an interactive bar chart

    -- Histogram:
        Visualize distribution of continuous features as an interactive histogram

    -- Box-Whisker-Plot:
        Visualize descriptive statistics of continuous features as an interactive box-whisker-plot

    -- Violin Chart:
        Visualize descriptive statistics of continuous features as an interactive violin chart

    -- Parallel Category Chart:
        Visualize relationships interactively between categorical features especially, but it can also be used for mixed relations between values of categorical and continuous features by using brushing as well.

    -- Parallel Coordinate Chart:
        Visualize relationships interactively between ranges of continuous features especially, but it can also be used for mixed relations between values of categorical and ranges of continuous features as well.

    -- Scatter Chart:
        Visualize values of continuous features interactively.

    -- Scatter3D Chart:
        Visualize values of three continuous features in one chart interactively.

    -- Joint Distribution Chart:
        Visualize values of two continuous features interactively, including contours and histogram for each continuous feature.

    -- Ridgeline Chart:
        Visualize changes in distribution of continuous features on certain time steps separately.

    -- Line Chart:
        Visualize distribution after certain time steps as an interactive line chart.

    -- Candlestick Chart:
        Visualize descritive statistics for each time steps as an interactive candlestick chart.

    -- Dendrogram:
        Visualize hierarchical clusters.

    -- Silhoutte Chart:
        Visualize partitionized clusters.

## 4. Examples:

Check the jupyter notebook for examples. Happy exploring :)
