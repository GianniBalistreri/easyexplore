"""

Unsupervised (clustering) machine learning algorithms

"""

import numpy as np
import os
import pandas as pd

from .data_import_export import DataExporter
from .data_visualizer import DataVisualizer
from .utils import EasyExploreUtils, Log, StatsUtils
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering, Birch, DBSCAN, FeatureAgglomeration, KMeans, MeanShift, OPTICS, SpectralClustering
from sklearn.decomposition import FactorAnalysis, FastICA, PCA, NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, kneighbors_graph
from sklearn.manifold import Isomap, LocallyLinearEmbedding, MDS, SpectralEmbedding, TSNE
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from typing import List


CLUSTERING_ALGORITHMS: List[str] = ['affinity_propagation',
                                    'agglo_cluster',
                                    'birch',
                                    'dbscan',
                                    'factor',
                                    'feature_agglo',
                                    'isomap',
                                    'kmeans',
                                    'lda',
                                    'lle',
                                    'mds',
                                    'nmf',
                                    'optics',
                                    'pca',
                                    'spectral_cluster',
                                    'spectral_embedding',
                                    'tsne',
                                    'tsvd'
                                    ]


class Clustering:
    """
    Class for handling clustering or dimensionality reduction algorithms
    """
    def __init__(self, cl_params: dict = None, cpu_cores: int = 0, seed: int = 1234):
        """
        :param cl_params: dict
            Clustering hyperparameters

        :param cpu_cores: int
            Number of cpu cores

        :param seed: int
            Seed
        """
        self.cl_params: dict = {} if cl_params is None else cl_params
        self.seed: int = 1234 if seed <= 0 else seed
        if cpu_cores <= 0:
            self.cpu_cores: int = os.cpu_count() - 1 if os.cpu_count() > 1 else os.cpu_count()
        else:
            if cpu_cores <= os.cpu_count():
                self.cpu_cores: int = cpu_cores
            else:
                self.cpu_cores: int = os.cpu_count() - 1 if os.cpu_count() > 1 else os.cpu_count()

    def affinity_propagation(self) -> AffinityPropagation:
        """
        Config affinity propagation

        :return: AffinityPropagation
            Sklearn object containing the affinity propagation configuration
        """
        return AffinityPropagation(damping=0.5 if self.cl_params.get('damping') is None else self.cl_params.get('damping'),
                                   max_iter=200 if self.cl_params.get('max_iter') is None else self.cl_params.get('max_iter'),
                                   convergence_iter=15 if self.cl_params.get('convergence_iter') is None else self.cl_params.get('convergence_iter'),
                                   copy=True if self.cl_params.get('copy') is None else self.cl_params.get('copy'),
                                   preference=self.cl_params.get('preference'),
                                   affinity='euclidean' if self.cl_params.get('affinity') is None else self.cl_params.get('affinity'),
                                   verbose=False if self.cl_params.get('verbose') is None else self.cl_params.get('verbose')
                                   )

    def agglomerative_clustering(self) -> AgglomerativeClustering:
        """
        Config agglomerative clustering

        :return: AgglomerativeClustering
            Sklearn object containing the agglomerative clustering configuration
        """
        return AgglomerativeClustering(n_clusters=3 if self.cl_params.get('n_clusters') is None else self.cl_params.get('n_clusters'),
                                       affinity='euclidean' if self.cl_params.get('affinity') is None else self.cl_params.get('affinity'),
                                       compute_full_tree='auto' if self.cl_params.get('compute_full_tree') is None else self.cl_params.get('compute_full_tree'),
                                       connectivity=self.cl_params.get('connectivity'),
                                       distance_threshold=self.cl_params.get('distance_threshold'),
                                       linkage='ward' if self.cl_params.get('linkage') is None else self.cl_params.get('linkage'),
                                       memory=self.cl_params.get('memory')
                                       )

    def birch(self) -> Birch:
        """
        Config birch clustering

        :return: Birch
            Sklearn object containing the birch clustering configuration
        """
        return Birch(threshold=0.5 if self.cl_params.get('threshold') is None else self.cl_params.get('threshold'),
                     branching_factor=50 if self.cl_params.get('branching_factor') is None else self.cl_params.get('branching_factor'),
                     n_clusters=3 if self.cl_params.get('n_clusters') is None else self.cl_params.get('n_clusters'),
                     compute_labels=True if self.cl_params.get('compute_labels') is None else self.cl_params.get('compute_labels'),
                     copy=True if self.cl_params.get('copy') is None else self.cl_params.get('copy'),
                     )

    def dbscan(self) -> DBSCAN:
        """
        Config density-based algorithm for discovering clusters in large spatial databases with noise

        :return: DBSCAN
            Sklearn object containing the dbscan clustering configuration
        """
        return DBSCAN(eps=0.5 if self.cl_params.get('eps') is None else self.cl_params.get('eps'),
                      min_samples=5 if self.cl_params.get('min_samples') is None else self.cl_params.get('min_samples'),
                      metric='euclidean' if self.cl_params.get('metric') is None else self.cl_params.get('metric'),
                      metric_params=self.cl_params.get('metric_params'),
                      algorithm='auto' if self.cl_params.get('algorithm') is None else self.cl_params.get('algorithm'),
                      leaf_size=30 if self.cl_params.get('leaf_size') is None else self.cl_params.get('leaf_size'),
                      p=self.cl_params.get('p'),
                      n_jobs=self.cpu_cores
                      )

    def factor_analysis(self) -> FactorAnalysis:
        """
        Config factor analysis

        :return: FactorAnalysis
            Sklearn object containing the factor analysis configuration
        """
        return FactorAnalysis(n_components=None if self.cl_params.get('n_components') is None else self.cl_params.get('n_components'),
                              tol=0.01 if self.cl_params.get('tol') is None else self.cl_params.get('tol'),
                              copy=True if self.cl_params.get('copy') is None else self.cl_params.get('copy'),
                              max_iter=1000 if self.cl_params.get('max_iter') is None else self.cl_params.get('max_iter'),
                              noise_variance_init=None if self.cl_params.get('noise_variance_init') is None else self.cl_params.get('noise_variance_init'),
                              svd_method='randomized' if self.cl_params.get('svd_method') is None else self.cl_params.get('svd_method'),
                              iterated_power=3 if self.cl_params.get('iterated_power') is None else self.cl_params.get('iterated_power'),
                              random_state=self.seed
                              )

    def feature_agglomeration(self) -> FeatureAgglomeration:
        """
        Config feature agglomeration clustering

        :return: FeatureAgglomeration
            Sklearn object containing the feature agglomeration configuration
        """
        return FeatureAgglomeration(n_clusters=2 if self.cl_params.get('n_clusters') is None else self.cl_params.get('n_clusters'),
                                    affinity='euclidean' if self.cl_params.get('affinity') is None else self.cl_params.get('affinity'),
                                    memory=None if self.cl_params.get('memory') is None else self.cl_params.get('memory'),
                                    connectivity=None if self.cl_params.get('connectivity') is None else self.cl_params.get('connectivity'),
                                    compute_full_tree='auto' if self.cl_params.get('compute_full_tree') is None else self.cl_params.get('compute_full_tree'),
                                    linkage='ward' if self.cl_params.get('linkage') is None else self.cl_params.get('linkage'),
                                    pooling_func=np.mean if self.cl_params.get('pooling_func') is None else self.cl_params.get('pooling_func'),
                                    distance_threshold=self.cl_params.get('distance_threshold')
                                    )

    def independent_component_analysis(self) -> FastICA:
        """
        Config independent component analysis

        :return: FastICA
            Sklearn object containing the independent component analysis configuration
        """
        return FastICA(n_components=2 if self.cl_params.get('n_components') is None else self.cl_params.get('n_components'),
                       algorithm='parallel' if self.cl_params.get('algorithm') is None else self.cl_params.get('algorithm'),
                       whiten=True if self.cl_params.get('whiten') is None else self.cl_params.get('whiten'),
                       fun='logcosh' if self.cl_params.get('fun') is None else self.cl_params.get('fun'),
                       fun_args=None if self.cl_params.get('fun_args') is None else self.cl_params.get('fun_args'),
                       max_iter=200 if self.cl_params.get('max_iter') is None else self.cl_params.get('max_iter'),
                       tol=0.0001 if self.cl_params.get('tol') is None else self.cl_params.get('tol'),
                       w_init=None if self.cl_params.get('w_init') is None else self.cl_params.get('w_init'),
                       random_state=self.seed
                       )

    def isometric_mapping(self) -> Isomap:
        """
        Config isometric mapping

        :return: Isomap
            Sklearn object containing the isometric mapping configuration
        """
        return Isomap(n_neighbors=5 if self.cl_params.get('n_neighbors') is None else self.cl_params.get('n_neighbors'),
                      n_components=2 if self.cl_params.get('n_components') is None else self.cl_params.get('n_components'),
                      eigen_solver='auto' if self.cl_params.get('eigen_solver') is None else self.cl_params.get('eigen_solver'),
                      tol=0 if self.cl_params.get('tol') is None else self.cl_params.get('tol'),
                      max_iter=self.cl_params.get('max_iter'),
                      path_method='auto' if self.cl_params.get('path_method') is None else self.cl_params.get('path_method'),
                      neighbors_algorithm='auto' if self.cl_params.get('neighbors_algorithm') is None else self.cl_params.get('neighbors_algorithm'),
                      n_jobs=self.cpu_cores
                      )

    def kmeans(self):
        """
        Config k-means clustering

        :return KMeans
            Sklearn object containing the k-means clustering configuration
        """
        return KMeans(n_clusters=2 if self.cl_params.get('n_clusters') is None else self.cl_params.get('n_clusters'),
                      init='random' if self.cl_params.get('init') is None else self.cl_params.get('init'),
                      n_init=10 if self.cl_params.get('n_init') is None else self.cl_params.get('n_init'),
                      max_iter=300 if self.cl_params.get('max_iter') is None else self.cl_params.get('max_iter'),
                      tol=1e-04 if self.cl_params.get('tol') is None else self.cl_params.get('tol'),
                      random_state=self.seed
                      )

    def latent_dirichlet_allocation(self) -> LatentDirichletAllocation:
        """
        Config latent dirichlet allocation

        :return: LatentDirichletAllocation
            Sklearn object containing the latent dirichlet allocation configuration
        """
        return LatentDirichletAllocation(n_components=10 if self.cl_params.get('n_components') is None else self.cl_params.get('n_components'),
                                         doc_topic_prior=None if self.cl_params.get('doc_topic_prior') is None else self.cl_params.get('doc_topic_prior'),
                                         topic_word_prior=None if self.cl_params.get('topic_word_prior') is None else self.cl_params.get('topic_word_prior'),
                                         learning_method='batch' if self.cl_params.get('learning_method') is None else self.cl_params.get('learning_method'),
                                         learning_decay=0.7 if self.cl_params.get('learning_decay') is None else self.cl_params.get('learning_decay'),
                                         learning_offset=10 if self.cl_params.get('learning_offset') is None else self.cl_params.get('learning_offset'),
                                         max_iter=10 if self.cl_params.get('max_iter') is None else self.cl_params.get('max_iter'),
                                         batch_size=128 if self.cl_params.get('batch_size') is None else self.cl_params.get('batch_size'),
                                         evaluate_every=-1 if self.cl_params.get('evaluate_every') is None else self.cl_params.get('evaluate_every'),
                                         total_samples=1e6 if self.cl_params.get('total_samples') is None else self.cl_params.get('total_samples'),
                                         perp_tol=0.1 if self.cl_params.get('perp_tol') is None else self.cl_params.get('perp_tol'),
                                         mean_change_tol=0.001 if self.cl_params.get('mean_change_tol') is None else self.cl_params.get('mean_change_tol'),
                                         max_doc_update_iter=100 if self.cl_params.get('max_doc_update_iter') is None else self.cl_params.get('max_doc_update_iter'),
                                         n_jobs=self.cpu_cores,
                                         verbose=0 if self.cl_params.get('verbose') is None else self.cl_params.get('verbose'),
                                         random_state=self.seed
                                         )

    def locally_linear_embedding(self) -> LocallyLinearEmbedding:
        """
        Config locally linear embedding

        :return: LocallyLinearEmbedding
            Sklearn object containing the locally linear embedding configuration
        """
        return LocallyLinearEmbedding(n_neighbors=5 if self.cl_params.get('n_neighbors') is None else self.cl_params.get('n_neighbors'),
                                      n_components=2 if self.cl_params.get('n_components') is None else self.cl_params.get('n_components'),
                                      reg=0.001 if self.cl_params.get('reg') is None else self.cl_params.get('reg'),
                                      eigen_solver='auto' if self.cl_params.get('eigen_solver') is None else self.cl_params.get('eigen_solver'),
                                      tol=0.000001 if self.cl_params.get('tol') is None else self.cl_params.get('tol'),
                                      max_iter=100 if self.cl_params.get('max_iter') is None else self.cl_params.get('max_iter'),
                                      method='standard' if self.cl_params.get('method') is None else self.cl_params.get('method'),
                                      hessian_tol=0.0001 if self.cl_params.get('hessian_tol') is None else self.cl_params.get('hessian_tol'),
                                      modified_tol=1e-12 if self.cl_params.get('modified_tol') is None else self.cl_params.get('modified_tol'),
                                      neighbors_algorithm='auto' if self.cl_params.get('neighbors_algorithm') is None else self.cl_params.get('neighbors_algorithm'),
                                      random_state=self.seed,
                                      n_jobs=self.cpu_cores
                                      )

    def multi_dimensional_scaling(self) -> MDS:
        """
        Config multi dimensional scaling

        :return: MDS
            Sklearn object containing the multi dimensional scaling configuration
        """
        return MDS(n_components=2 if self.cl_params.get('n_components') is None else self.cl_params.get('n_components'),
                   metric=True if self.cl_params.get('metric') is None else self.cl_params.get('metric'),
                   n_init=4 if self.cl_params.get('n_init') is None else self.cl_params.get('n_init'),
                   max_iter=300 if self.cl_params.get('max_iter') is None else self.cl_params.get('max_iter'),
                   verbose=0 if self.cl_params.get('verbose') is None else self.cl_params.get('verbose'),
                   eps=0.001 if self.cl_params.get('eps') is None else self.cl_params.get('eps'),
                   n_jobs=self.cpu_cores,
                   random_state=self.seed,
                   dissimilarity='euclidean' if self.cl_params.get('dissimilarity') is None else self.cl_params.get('dissimilarity')
                   )

    def non_negative_matrix_factorization(self) -> NMF:
        """
        Config non-negative matrix factorization

        :return NMF
            Sklearn object containing the non-negative matrix factorization clustering configuration
        """
        return NMF(n_components=10 if self.cl_params.get('n_components') is None else self.cl_params.get('n_components'),
                   init=None if self.cl_params.get('init') is None else self.cl_params.get('init'),
                   solver='cd' if self.cl_params.get('solver') is None else self.cl_params.get('solver'),
                   beta_loss='frobenius' if self.cl_params.get('beta_loss') is None else self.cl_params.get('beta_loss'),
                   tol=0.0001 if self.cl_params.get('tol') is None else self.cl_params.get('tol'),
                   max_iter=200 if self.cl_params.get('max_iter') is None else self.cl_params.get('max_iter'),
                   random_state=self.seed,
                   alpha=0 if self.cl_params.get('alpha') is None else self.cl_params.get('alpha'),
                   l1_ratio=0 if self.cl_params.get('l1_ratio') is None else self.cl_params.get('l1_ratio'),
                   verbose=0 if self.cl_params.get('verbose') is None else self.cl_params.get('verbose'),
                   shuffle=False if self.cl_params.get('shuffle') is None else self.cl_params.get('shuffle'),
                   )

    def optics(self) -> OPTICS:
        """
        Config ordering points to identify clustering structure

        :return: OPTICS
            Sklearn object containing the optics configuration
        """
        return OPTICS(min_samples=5 if self.cl_params.get('min_samples') is None else self.cl_params.get('min_samples'),
                      max_eps=np.inf if self.cl_params.get('max_eps') is None else self.cl_params.get('max_eps'),
                      metric='minkowski' if self.cl_params.get('metric') is None else self.cl_params.get('metric'),
                      p=2 if self.cl_params.get('p') is None else self.cl_params.get('p'),
                      metric_params=self.cl_params.get('metric_params'),
                      cluster_method='xi' if self.cl_params.get('cluster_method') is None else self.cl_params.get('cluster_method'),
                      eps=self.cl_params.get('eps'),
                      xi=0.05 if self.cl_params.get('xi') is None else self.cl_params.get('xi'),
                      predecessor_correction=True if self.cl_params.get('predecessor_correction') is None else self.cl_params.get('predecessor_correction'),
                      min_cluster_size=self.cl_params.get('min_cluster_size'),
                      algorithm='auto' if self.cl_params.get('algorithm') is None else self.cl_params.get('algorithm'),
                      leaf_size=30 if self.cl_params.get('leaf_size') is None else self.cl_params.get('leaf_size'),
                      n_jobs=self.cpu_cores
                      )

    def principal_component_analysis(self) -> PCA:
        """
        Config principal component analysis

        :return: PCA
            Sklearn object containing the principal component analysis configuration
        """
        return PCA(n_components=self.cl_params.get('n_components'),
                   copy=True if self.cl_params.get('copy') is None else self.cl_params.get('copy'),
                   whiten=False if self.cl_params.get('whiten') is None else self.cl_params.get('whiten'),
                   svd_solver='auto' if self.cl_params.get('svd_solver') is None else self.cl_params.get('svd_solver'),
                   tol=0.0 if self.cl_params.get('tol') is None else self.cl_params.get('tol'),
                   iterated_power='auto' if self.cl_params.get('iterated_power') is None else self.cl_params.get('iterated_power'),
                   random_state=self.seed
                   )

    def spectral_clustering(self) -> SpectralClustering:
        """
        Config spectral clustering

        :return: SpectralClustering
            Sklearn object containing the spectral clustering configuration
        """
        return SpectralClustering(n_clusters=8 if self.cl_params.get('n_clusters') is None else self.cl_params.get('n_clusters'),
                                  eigen_solver=self.cl_params.get('eigen_solver'),
                                  random_state=self.seed,
                                  n_init=10 if self.cl_params.get('n_init') is None else self.cl_params.get('n_init'),
                                  gamma=1.0 if self.cl_params.get('gamma') is None else self.cl_params.get('gamma'),
                                  affinity='rbf' if self.cl_params.get('affinity') is None else self.cl_params.get('affinity'),
                                  n_neighbors=10 if self.cl_params.get('n_neighbors') is None else self.cl_params.get('n_neighbors'),
                                  eigen_tol=0.0 if self.cl_params.get('eigen_tol') is None else self.cl_params.get('eigen_tol'),
                                  assign_labels='kmeans' if self.cl_params.get('assign_labels') is None else self.cl_params.get('assign_labels'),
                                  degree=3 if self.cl_params.get('degree') is None else self.cl_params.get('degree'),
                                  coef0=1 if self.cl_params.get('coef0') is None else self.cl_params.get('coef0'),
                                  kernel_params=self.cl_params.get('kernel_params'),
                                  n_jobs=self.cpu_cores
                                  )

    def spectral_embedding(self) -> SpectralEmbedding:
        """
        Config spectral embedding

        :return: SpectralEmbedding
            Sklearn object containing the spectral embedding configuration
        """
        return SpectralEmbedding(n_components=2 if self.cl_params.get('n_components') is None else self.cl_params.get('n_components'),
                                 affinity='nearest_neighbors' if self.cl_params.get('affinity') is None else self.cl_params.get('affinity'),
                                 gamma=self.cl_params.get('gamma'),
                                 random_state=self.seed,
                                 eigen_solver=self.cl_params.get('eigen_solver'),
                                 n_neighbors=self.cl_params.get('n_neighbors'),
                                 n_jobs=self.cpu_cores
                                 )

    def t_distributed_stochastic_neighbor_embedding(self) -> TSNE:
        """
        Config t-distributed stochastic neighbor embedding

        :return: TSNE
            Sklearn object containing the t-distributed stochastic neighbor embedding configuration
        """
        return TSNE(n_components=2 if self.cl_params.get('n_components') is None else self.cl_params.get('n_components'),
                    perplexity=30.0 if self.cl_params.get('perplexity') is None else self.cl_params.get('perplexity'),
                    early_exaggeration=12.0 if self.cl_params.get('early_exaggeration') is None else self.cl_params.get('early_exaggeration'),
                    learning_rate=200.0 if self.cl_params.get('learning_rate') is None else self.cl_params.get('learning_rate'),
                    n_iter=1000 if self.cl_params.get('n_iter') is None else self.cl_params.get('n_iter'),
                    n_iter_without_progress=300 if self.cl_params.get('n_iter_without_progress') is None else self.cl_params.get('n_iter_without_progress'),
                    min_grad_norm=1e-7 if self.cl_params.get('min_grad_norm') is None else self.cl_params.get('min_grad_norm'),
                    metric='euclidean' if self.cl_params.get('metric') is None else self.cl_params.get('metric'),
                    init='random' if self.cl_params.get('init') is None else self.cl_params.get('init'),
                    verbose=0 if self.cl_params.get('verbose') is None else self.cl_params.get('verbose'),
                    random_state=self.seed,
                    method='barnes_hut' if self.cl_params.get('method') is None else self.cl_params.get('method'),
                    angle=0.5 if self.cl_params.get('angle') is None else self.cl_params.get('angle')
                    )

    def truncated_single_value_decomp(self) -> TruncatedSVD:
        """
        Config latent semantic analysis using truncated single value decomposition

        :return: TruncatedSVD
            Sklearn object containing the latent truncated single value decomposition configuration
        """
        return TruncatedSVD(n_components=2 if self.cl_params.get('n_components') is None else self.cl_params.get('n_components'),
                            algorithm='randomized' if self.cl_params.get('algorithm') is None else self.cl_params.get('algorithm'),
                            n_iter=5 if self.cl_params.get('n_iter') is None else self.cl_params.get('n_iter'),
                            random_state=self.seed,
                            tol=0.0 if self.cl_params.get('tol') is None else self.cl_params.get('tol')
                            )


class UnsupervisedMLException(Exception):
    """
    Class for handling exceptions for class UnsupervisedML
    """
    pass


class UnsupervisedML:
    """
    Class for applying unsupervised learning algorithms
    """
    def __init__(self,
                 df: pd.DataFrame,
                 features: List[str] = None,
                 find_optimum: bool = True,
                 silhouette_analysis: bool = True,
                 n_cluster_components: int = None,
                 n_neighbors: int = None,
                 n_iter: int = None,
                 metric: List[str] = None,
                 affinity: List[str] = None,
                 connectivity: List[str] = None,
                 linkage: List[str] = None,
                 plot: bool = True,
                 cpu_cores: int = 0,
                 **kwargs
                 ):
        """
        :param df: pd.DataFrame
            Data set

        :param features: List[str]
            List of strings containing the features to cluster

        :param target: str
            Name of the target features

        :param find_optimum: bool
            Whether to Find optimum number of components or clusters or not

        :param n_cluster_components: int
            Amount of clusters for partitioning clustering

        :param n_neighbors: int
            Amount of neighbors

        :param n_iter: int
            Amount of iterations

        :param metric: List[str]
            Names of the metric for each clustering

        :param affinity: List[str]
            Names of the affinity metric for each clustering

        :param connectivity: List[str]
            Names of the connectivity structure for each clustering

        :param linkage: List[str]
            Names of the linkage function for each clustering

        :param silhouette_analysis: bool
            Run silhouette analysis to evaluate clustering or not

        :param plot: bool
            Plot clustering or not

        :param kwargs: dict
            Key-word arguments regarding the machine learning algorithms
        """
        self.df: pd.DataFrame = df
        self.features: List[str] = list(self.df.keys()) if features is None else features
        if len(self.features) == 0:
            self.features = list(self.df.keys())
        self.cluster: dict = {}
        self.cluster_plot: dict = {}
        self.ml_algorithm: str = None
        self.find_optimum: bool = find_optimum
        self.n_cluster_components: int = n_cluster_components
        self.n_neighbors: int = n_neighbors
        self.n_iter: int = n_iter
        self.metric: List[str] = metric
        self.affinity: List[str] = affinity
        self.connectivity: List[str] = connectivity
        self.linkage: List[str] = linkage
        self.silhouette: bool = silhouette_analysis
        self.plot: bool = plot
        self.seed: int = 1234
        self.eigen_value = None
        self.eigen_vector = None
        if cpu_cores <= 0:
            self.cpu_cores: int = os.cpu_count() - 1 if os.cpu_count() > 1 else os.cpu_count()
        else:
            if cpu_cores <= os.cpu_count():
                self.cpu_cores: int = cpu_cores
            else:
                self.cpu_cores: int = os.cpu_count() - 1 if os.cpu_count() > 1 else os.cpu_count()
        self.kwargs: dict = {} if kwargs is None else kwargs
        if self.kwargs.get('file_path') is None:
            self.to_export: bool = False
        else:
            if len(self.kwargs.get('file_path')) > 0:
                self.to_export: bool = True
            else:
                self.to_export: bool = False

    def _affinity_propagation(self):
        """
        Affinity propagation for graph based cluster without pre-defined partitions
        """
        _clustering: AffinityPropagation = Clustering(cl_params=self.kwargs).affinity_propagation()
        _clustering.fit(X=self.df[self.features])
        _labels: np.array = _clustering.predict(self.df[self.features])
        self.cluster[self.ml_algorithm].update({'fit': _clustering,
                                                'n_clusters': len(list(set(_labels)))
                                                })
        self.kwargs.update({'n_clusters': self.cluster[self.ml_algorithm].get('n_clusters')})
        if self.to_export:
            _file_path_silhouette: str = os.path.join(self.kwargs.get('file_path'), 'affinity_propagation_silhouette.html')
            _file_path_cluster_partition: str = os.path.join(self.kwargs.get('file_path'), 'affinity_propagation_cluster_partition.html')
        else:
            _file_path_silhouette: str = None
            _file_path_cluster_partition: str = None
        if self.find_optimum:
            if self.silhouette:
                _silhouette: dict = self.silhouette_analysis(labels=_labels)
                self.cluster[self.ml_algorithm].update({'silhouette': _silhouette})
                self.cluster_plot.update({'Affinity Propagation: Silhouette Analysis': dict(data=self.df,
                                                                                            features=None,
                                                                                            plot_type='silhouette',
                                                                                            use_auto_extensions=False if self.kwargs.get(
                                                                                                'use_auto_extensions') is None else self.kwargs.get(
                                                                                                'use_auto_extensions'),
                                                                                            file_path=_file_path_silhouette,
                                                                                            kwargs=dict(layout={},
                                                                                                        n_clusters=self.cluster[self.ml_algorithm].get('n_clusters'),
                                                                                                        silhouette=_silhouette
                                                                                                        )
                                                                                            )
                                          })
        if 'silhouette' not in self.cluster[self.ml_algorithm].keys():
            self.cluster[self.ml_algorithm].update({'silhouette': None})
        self.cluster[self.ml_algorithm].update({'cluster_centers': _clustering.cluster_centers_,
                                                'affinity_matrix': _clustering.affinity_matrix_,
                                                'labels': _clustering.labels_,
                                                'cluster': _clustering.predict(X=self.df[self.features])
                                                })
        _df: pd.DataFrame = self.df
        _df['cluster'] = self.cluster[self.ml_algorithm].get('cluster')
        self.cluster_plot.update({'Affinity Propagation: Cluster Partition': dict(data=_df,
                                                                                  features=self.features,
                                                                                  group_by=['cluster'],
                                                                                  melt=True if self.kwargs.get('melt') is None else self.kwargs.get('melt'),
                                                                                  plot_type='scatter',
                                                                                  use_auto_extensions=False if self.kwargs.get(
                                                                                                'use_auto_extensions') is None else self.kwargs.get(
                                                                                                'use_auto_extensions'),
                                                                                  file_path=_file_path_cluster_partition,
                                                                                  kwargs=dict(layout={})
                                                                                  )
                                  })

    def _agglomerative_clustering(self):
        """
        Agglomerative clustering for hierarchical clustering using similarities
        """
        if self.ml_algorithm.find('unstruc') < 0:
            if self.kwargs.get('connectivity') is None:
                self.kwargs.update({'connectivity': kneighbors_graph(X=self.df[self.features],
                                                                     n_neighbors=self.n_neighbors,
                                                                     mode='connectivity' if self.kwargs.get(
                                                                         'connectivity') is None else self.kwargs.get(
                                                                         'connectivity'),
                                                                     metric='minkowski' if self.kwargs.get(
                                                                         'metric') is None else self.kwargs.get(
                                                                         'metric'),
                                                                     p=2 if self.kwargs.get(
                                                                         'p') is None else self.kwargs.get('p'),
                                                                     metric_params=self.kwargs.get('metric_params'),
                                                                     include_self=False if self.kwargs.get(
                                                                         'include_self') is None else self.kwargs.get(
                                                                         'include_self'),
                                                                     n_jobs=self.cpu_cores
                                                                     )
                                    })
        _clustering: AgglomerativeClustering = Clustering(cl_params=self.kwargs).agglomerative_clustering()
        _clustering.fit(X=self.df[self.features])
        self.cluster[self.ml_algorithm].update({'fit': _clustering,
                                                'n_clusters': _clustering.n_clusters_
                                                })
        self.kwargs.update({'n_clusters': self.cluster[self.ml_algorithm].get('n_clusters')})
        if self.to_export:
            _file_path_silhouette: str = os.path.join(self.kwargs.get('file_path'), 'agglomerative_clustering_silhouette.html')
            _file_path_cluster_partition: str = os.path.join(self.kwargs.get('file_path'), 'agglomerative_clustering_partition.html')
            _file_path_hierarchical: str = os.path.join(self.kwargs.get('file_path'), 'agglomerative_clustering_hierarchical.html')
        else:
            _file_path_silhouette: str = None
            _file_path_cluster_partition: str = None
            _file_path_hierarchical: str = None
        if self.find_optimum:
            if self.silhouette:
                _silhouette: dict = self.silhouette_analysis(labels=_clustering.labels_)
                self.cluster[self.ml_algorithm].update({'silhouette': _silhouette})
                self.cluster_plot.update({'Agglomerative Clustering: Silhouette Analysis': dict(data=self.df,
                                                                                                features=None,
                                                                                                plot_type='silhouette',
                                                                                                use_auto_extensions=False if self.kwargs.get(
                                                                                                'use_auto_extensions') is None else self.kwargs.get(
                                                                                                'use_auto_extensions'),
                                                                                                file_path=_file_path_silhouette,
                                                                                                kwargs=dict(layout={},
                                                                                                            n_clusters=self.cluster[self.ml_algorithm].get('n_clusters'),
                                                                                                            silhouette=_silhouette
                                                                                                            )
                                                                                                )
                                          })
        if 'silhouette' not in self.cluster[self.ml_algorithm].keys():
            self.cluster[self.ml_algorithm].update({'silhouette': None})
        self.cluster[self.ml_algorithm].update({'connectivity': self.kwargs.get('connectivity'),
                                                'n_clusters': _clustering.n_clusters_,
                                                'n_leaves': _clustering.n_leaves_,
                                                'n_components': _clustering.n_connected_components_,
                                                'children': _clustering.children_,
                                                'labels': _clustering.labels_
                                                })
        _df: pd.DataFrame = self.df
        _df['cluster'] = self.cluster[self.ml_algorithm].get('labels')
        self.cluster_plot.update({'Agglomerative Clustering: Partition': dict(data=_df,
                                                                              features=self.features,
                                                                              group_by=['cluster'],
                                                                              melt=True if self.kwargs.get('melt') is None else self.kwargs.get('melt'),
                                                                              plot_type='scatter',
                                                                              use_auto_extensions=False if self.kwargs.get('use_auto_extensions') is None else self.kwargs.get('use_auto_extensions'),
                                                                              file_path=_file_path_cluster_partition,
                                                                              kwargs=dict(layout={})
                                                                              ),
                                  'Agglomerative Clustering: Hierarchical': dict(data=_df,
                                                                                 features=self.features,
                                                                                 plot_type='dendro',
                                                                                 use_auto_extensions=False if self.kwargs.get(
                                                                                     'use_auto_extensions') is None else self.kwargs.get(
                                                                                     'use_auto_extensions'),
                                                                                 file_path=_file_path_hierarchical,
                                                                                 kwargs=dict(layout={})
                                                                                 ),
                                  })

    def _birch_clustering(self):
        """
        Balanced iterative reducing and clustering using hierarchies (Birch) for generating efficient cluster partitions on big data sets
        """
        _clustering: Birch = Clustering(cl_params=self.kwargs).birch()
        _clustering.fit(X=self.df[self.features])
        _labels: np.array = _clustering.predict(self.df[self.features])
        self.cluster[self.ml_algorithm].update({'fit': _clustering,
                                                'n_clusters': len(list(set(_labels)))
                                                })
        self.kwargs.update({'n_clusters': self.cluster[self.ml_algorithm].get('n_clusters')})
        if self.to_export:
            _file_path_silhouette: str = os.path.join(self.kwargs.get('file_path'), 'birch_silhouette.html')
            _file_path_cluster_partition: str = os.path.join(self.kwargs.get('file_path'), 'birch_cluster_partition.html')
        else:
            _file_path_silhouette: str = None
            _file_path_cluster_partition: str = None
        if self.find_optimum:
            if self.silhouette:
                _silhouette: dict = self.silhouette_analysis(labels=_labels)
                self.cluster[self.ml_algorithm].update({'n_clusters': int((len(list(_silhouette.keys())) - 1) / 2),
                                                        'silhouette': _silhouette
                                                        })
                self.cluster_plot.update({'Birch: Silhouette Analysis': dict(data=self.df,
                                                                             features=None,
                                                                             plot_type='silhouette',
                                                                             use_auto_extensions=False if self.kwargs.get(
                                                                                                'use_auto_extensions') is None else self.kwargs.get(
                                                                                                'use_auto_extensions'),
                                                                             file_path=_file_path_silhouette,
                                                                             kwargs=dict(layout={},
                                                                                         n_clusters=self.cluster[self.ml_algorithm].get('n_clusters'),
                                                                                         silhouette=_silhouette
                                                                                         )
                                                                             )
                                          })
        if 'silhouette' not in self.cluster[self.ml_algorithm].keys():
            self.cluster[self.ml_algorithm].update({'silhouette': None})
        self.cluster[self.ml_algorithm].update({'partial_fit': _clustering.partial_fit_,
                                                'root': _clustering.root_,
                                                'centroids': _clustering.subcluster_centers_,
                                                'cluster': _clustering.transform(X=self.df[self.features]),
                                                'cluster_labels': _clustering.subcluster_labels_,
                                                'dummy_leaf': _clustering.dummy_leaf_,
                                                'labels': _clustering.labels_
                                                })
        _df: pd.DataFrame = self.df
        _df['cluster'] = self.cluster[self.ml_algorithm].get('labels')
        self.cluster_plot.update({'Birch: Cluster Partition': dict(data=_df,
                                                                   features=self.features,
                                                                   group_by=['cluster'],
                                                                   melt=True if self.kwargs.get('melt') is None else self.kwargs.get('melt'),
                                                                   plot_type='scatter',
                                                                   use_auto_extensions=False if self.kwargs.get(
                                                                                                'use_auto_extensions') is None else self.kwargs.get(
                                                                                                'use_auto_extensions'),
                                                                   file_path=_file_path_cluster_partition,
                                                                   kwargs=dict(layout={})
                                                                   )
                                  })

    def _clean_missing_data(self):
        """
        Clean cases containing missing data
        """
        Log(write=False, level='info').log(msg='Clean cases containing missing values...')
        self.df = self.df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
        if self.df.shape[0] == 0:
            raise UnsupervisedMLException('No cases containing valid observations left')

    def _cumulative_explained_variance_ratio(self, explained_variance_ratio: np.ndarray) -> int:
        """
        Calculate optimal amount of components to be used for principal component analysis based on the explained variance ratio

        :return: int
            Optimal amount of components
        """
        _threshold: float = 0.75 if self.kwargs.get('cev') is None else self.kwargs.get('cev')
        for i, ratio in enumerate(np.cumsum(explained_variance_ratio)):
            if ratio >= _threshold:
                return i + 1

    def _density_based_spatial_clustering_applications_with_noise(self):
        """
        Density-based spatial clustering applications with noise (DBSCAN) for clustering complex structures like dense regions in space
        """
        _clustering: DBSCAN = Clustering(cl_params=self.kwargs).dbscan()
        _clustering.fit(X=self.df[self.features])
        self.cluster[self.ml_algorithm].update({'fit': _clustering,
                                                'n_clusters': len(list(set(_clustering.labels_))),
                                                'core_sample_indices': _clustering.core_sample_indices_,
                                                'labels': _clustering.labels_
                                                })
        _df: pd.DataFrame = self.df
        _df['cluster'] = self.cluster[self.ml_algorithm].get('labels')
        if self.to_export:
            _file_path_cluster_partition: str = os.path.join(self.kwargs.get('file_path'), 'dbscan_cluster_partition.html')
        else:
            _file_path_cluster_partition: str = None
        self.cluster_plot.update({'DBSCAN: Cluster Partition': dict(data=_df,
                                                                    features=self.features,
                                                                    group_by=['cluster'],
                                                                    melt=True if self.kwargs.get('melt') is None else self.kwargs.get('melt'),
                                                                    plot_type='scatter',
                                                                    use_auto_extensions=False if self.kwargs.get(
                                                                                                'use_auto_extensions') is None else self.kwargs.get(
                                                                                                'use_auto_extensions'),
                                                                    file_path=_file_path_cluster_partition,
                                                                    kwargs=dict(layout={})
                                                                    )
                                  })

    def _factor_analysis(self):
        """
        Factor analysis
        """
        _kmo: dict = StatsUtils(data=self.df).factoriability_test(meth='kmo')
        self.cluster[self.ml_algorithm].update({'kmo': _kmo})
        if _kmo.get('kmo') < 0.6:
            Log(write=False, level='info').log(
                msg='Data set not suitable for running factor analysis since KMO coefficient ({}) is lower than 0.6'.format(
                    _kmo.get('kmo')))
        else:
            if self.n_cluster_components is None:
                self.kwargs.update({'n_factors': 2})
            else:
                if self.n_cluster_components >= len(self.features):
                    self.kwargs.update({'n_factors': 2})
                    Log(write=False, level='info').log(
                        msg='Number of factors are greater than or equal to number of features. Number of factors set to 2')
                else:
                    self.kwargs.update({'n_components': self.n_cluster_components})
            _clustering: FactorAnalysis = Clustering(cl_params=self.kwargs).factor_analysis()
            _clustering.fit(X=self.df[self.features])
            self.cluster[self.ml_algorithm].update({'fit': _clustering,
                                                    'n_factors': self.kwargs.get('n_factors')
                                                    })
            if self.find_optimum:
                if self.silhouette:
                    _silhouette: dict = self.silhouette_analysis(labels=_clustering.transform(self.df[self.features]))
                    self.cluster[self.ml_algorithm].update({'silhouette': _silhouette})
                    self.cluster_plot.update({'Silhouette Analysis (FA)': dict(data=None,
                                                                               features=None,
                                                                               plot_type='silhouette',
                                                                               use_auto_extensions=False if self.kwargs.get(
                                                                                                'use_auto_extensions') is None else self.kwargs.get(
                                                                                                'use_auto_extensions'),
                                                                               file_path=self.kwargs.get('file_path'),
                                                                               kwargs=dict(layout={},
                                                                                           n_clusters=self.kwargs.get(
                                                                                               'n_clusters'),
                                                                                           silhouette=_silhouette
                                                                                           )
                                                                               )
                                              })
                else:
                    self.kwargs.update({'n_factors': self._estimate_optimal_factors(factors=_clustering.transform(X=self.df[self.features]))})
                    self.cluster[self.ml_algorithm].update({'n_factors': self.kwargs.get('n_factors')})
                    self.cluster_plot.update({'Optimal Number of Factors': dict(data=self.eigen_value,
                                                                                features=None,
                                                                                plot_type='line',
                                                                                use_auto_extensions=False if self.kwargs.get(
                                                                                                'use_auto_extensions') is None else self.kwargs.get(
                                                                                                'use_auto_extensions'),
                                                                                file_path=self.kwargs.get('file_path'),
                                                                                kwargs=dict(layout={})
                                                                                )
                                              })
            if 'silhouette' not in self.cluster[self.ml_algorithm].keys():
                self.cluster[self.ml_algorithm].update({'silhouette': None})
            _factors: np.array = _clustering.transform(X=self.df[self.features])
            self.cluster[self.ml_algorithm].update({'factors': _clustering.components_,
                                                    'explained_variance': _clustering.explained_variance_,
                                                    'fa': _factors
                                                    })
            _components: pd.DataFrame = pd.DataFrame(data=np.array(_clustering.components_),
                                                     columns=self.features,
                                                     index=['fa{}'.format(fa) for fa in
                                                            range(0, self.kwargs.get('n_factors'), 1)]
                                                     ).transpose()
            _feature_importance: pd.DataFrame = abs(_components)
            for fa in range(0, self.kwargs.get('n_factors'), 1):
                self.cluster_plot.update({'Feature Importance FA{}'.format(fa): dict(data=_feature_importance,
                                                                                     features=None,
                                                                                     plot_type='bar',
                                                                                     use_auto_extensions=False if self.kwargs.get(
                                                                                                'use_auto_extensions') is None else self.kwargs.get(
                                                                                                'use_auto_extensions'),
                                                                                     file_path=self.kwargs.get(
                                                                                         'file_path'),
                                                                                     kwargs=dict(layout={},
                                                                                                 x=self.features,
                                                                                                 y=_feature_importance[
                                                                                                     'fa{}'.format(fa)],
                                                                                                 marker=dict(color=
                                                                                                             _feature_importance[
                                                                                                                 'fa{}'.format(
                                                                                                                     fa)],
                                                                                                             colorscale='rdylgn',
                                                                                                             autocolorscale=True
                                                                                                             )
                                                                                                 )
                                                                                     )
                                          })
            self.cluster_plot.update({'Explained Variance': dict(data=pd.DataFrame(),
                                                                 features=None,
                                                                 plot_type='bar',
                                                                 use_auto_extensions=False if self.kwargs.get(
                                                                                                'use_auto_extensions') is None else self.kwargs.get(
                                                                                                'use_auto_extensions'),
                                                                 file_path=self.kwargs.get('file_path'),
                                                                 kwargs=dict(layout={},
                                                                             x=list(_feature_importance.keys()),
                                                                             y=self.cluster[self.ml_algorithm].get('explained_variance_ratio')
                                                                             )
                                                                 ),
                                      'Factor Loadings': dict(data=pd.DataFrame(data=self.cluster[self.ml_algorithm].get('fa'),
                                                                                columns=list(_feature_importance.keys())
                                                                                ),
                                                              features=list(_feature_importance.keys()),
                                                              plot_type='scatter',
                                                              use_auto_extensions=False if self.kwargs.get(
                                                                                                'use_auto_extensions') is None else self.kwargs.get(
                                                                                                'use_auto_extensions'),
                                                              file_path=self.kwargs.get('file_path'),
                                                              melt=True if self.kwargs.get('melt') is None else self.kwargs.get('melt'),
                                                              kwargs=dict(layout={},
                                                                          marker=dict(color=self.cluster[self.ml_algorithm].get('fa'),
                                                                                      colorscale='rdylgn',
                                                                                      autocolorscale=True
                                                                                      )
                                                                          )
                                                              )
                                      })

    def _feature_agglomeration(self):
        """
        Feature agglomeration for reducing features into grouped clusters (hierarchical clustering)
        """
        if self.n_cluster_components is None:
            self.kwargs.update({'n_clusters': 2})
        else:
            if self.n_cluster_components < 2:
                Log(write=False,
                    level='info'
                    ).log(msg=f"It makes no sense to reduce feature dimensionality into less than 2 groups ({self.kwargs.get('n_clusters')}). Number of clusters are set to 2")
                self.kwargs.update({'n_clusters': 2})
            else:
                self.kwargs.update({'n_clusters': self.n_cluster_components})
        _clustering: FeatureAgglomeration = Clustering(cl_params=self.kwargs).feature_agglomeration()
        _clustering.fit(X=self.df[self.features])
        self.cluster[self.ml_algorithm].update({'fit': _clustering,
                                                'n_clusters': _clustering.n_clusters_,
                                                'n_leaves': _clustering.n_leaves_,
                                                'n_components': _clustering.n_connected_components_,
                                                'children': _clustering.children_,
                                                'reduced_data_set': _clustering.transform(X=self.df[self.features]),
                                                'labels': _clustering.labels_
                                                })

    def _estimate_optimal_factors(self, factors: np.array) -> int:
        """
        Calculate optimal amount of factors to be used in factor analysis based on the eigenvalues

        :param factors: np.array
            Factor loadings

        :return: int
            Optimal amount of factors
        """
        _diff: List[float] = []
        self.eigen_value, self.eigen_vector = np.linalg.eig(factors)
        for eigen_value in self.eigen_value:
            _diff.append(1 - eigen_value)
        return _diff.index(factors.min())

    def _elbow(self) -> int:
        """
        Calculate optimal number of clusters for partitioning clustering

        :return: int
            Optimal amount of clusters
        """
        _distortions: list = []
        _max_clusters: int = 10 if self.kwargs.get('max_clusters') is None else self.kwargs.get('max_clusters')
        for i in range(1, _max_clusters, 1):
            self.kwargs.update({'n_clusters': i})
            _distortions.append(Clustering(cl_params=self.kwargs).kmeans().fit(self.df[self.features]).inertia_)
        return 1

    def _isometric_mapping(self):
        """
        Isometric mapping for non-linear dimensionality reduction (manifold learning)
        """
        if self.n_cluster_components is None:
            self.kwargs.update({'n_components': 3})
        else:
            self.kwargs.update({'n_components': self.n_cluster_components})
        _clustering: Isomap = Clustering(cl_params=self.kwargs).isometric_mapping()
        _clustering.fit(X=self.df[self.features])
        self.cluster[self.ml_algorithm].update({'fit': _clustering,
                                                'n_components': self.kwargs.get('n_components'),
                                                'embeddings': _clustering.embedding_,
                                                'transformed_embeddings': _clustering.transform(X=self.df[self.features]),
                                                'distance_matrix': _clustering.dist_matrix_,
                                                'kernel_pca': _clustering.kernel_pca_,
                                                'reconstruction_error': _clustering.reconstruction_error()
                                                })

    def _k_means(self):
        """
        K-Means clustering (partitioning clustering) for graphical data, which is spherical about the cluster centre
        """
        if self.n_cluster_components is None:
            self.kwargs.update({'n_clusters': 2})
        else:
            if self.n_cluster_components < 2:
                Log(write=False,
                    level='info'
                    ).log(
                    msg=f"It makes no sense to run cluster analysis with less than 2 clusters ({self.kwargs.get('n_clusters')}). Number of components are set to 2")
                self.kwargs.update({'n_clusters': 2})
            else:
                self.kwargs.update({'n_clusters': self.n_cluster_components})
        _clustering: KMeans = Clustering(cl_params=self.kwargs).kmeans()
        _clustering.fit(X=self.df[self.features])
        self.cluster[self.ml_algorithm].update({'fit': _clustering,
                                                'n_clusters': self.kwargs.get('n_clusters')
                                                })
        if self.to_export:
            _file_path_silhouette: str = os.path.join(self.kwargs.get('file_path'), 'kmeans_silhouette.html')
            _file_path_cluster_partition: str = os.path.join(self.kwargs.get('file_path'), 'kmeans_cluster_partition.html')
        else:
            _file_path_silhouette: str = None
            _file_path_cluster_partition: str = None
        if self.find_optimum:
            if self.silhouette:
                _silhouette: dict = self.silhouette_analysis(labels=_clustering.predict(self.df[self.features]))
                self.cluster[self.ml_algorithm].update({'silhouette': _silhouette})
                self.cluster_plot.update({'K-Means: Silhouette Analysis': dict(data=self.df,
                                                                               features=None,
                                                                               plot_type='silhouette',
                                                                               use_auto_extensions=False if self.kwargs.get(
                                                                                                'use_auto_extensions') is None else self.kwargs.get(
                                                                                                'use_auto_extensions'),
                                                                               file_path=_file_path_silhouette,
                                                                               kwargs=dict(layout={},
                                                                                           n_clusters=self.kwargs.get(
                                                                                               'n_clusters'),
                                                                                           silhouette=_silhouette
                                                                                           )
                                                                               )
                                          })
        if 'silhouette' not in self.cluster[self.ml_algorithm].keys():
            self.cluster[self.ml_algorithm].update({'silhouette': None})
        self.cluster[self.ml_algorithm].update({'inertia': _clustering.inertia_,
                                                'cluster': _clustering.predict(X=self.df[self.features]),
                                                'cluster_distance_space': _clustering.transform(X=self.df[self.features]),
                                                'centroids': _clustering.cluster_centers_,
                                                'labels': _clustering.labels_
                                                })
        _df: pd.DataFrame = self.df
        _df['cluster'] = self.cluster[self.ml_algorithm].get('cluster')
        self.cluster_plot.update({'KMeans: Cluster Partition': dict(data=_df,
                                                                    features=self.features,
                                                                    group_by=['cluster'],
                                                                    melt=True if self.kwargs.get('melt') is None else self.kwargs.get('melt'),
                                                                    plot_type='scatter',
                                                                    use_auto_extensions=False if self.kwargs.get(
                                                                                                'use_auto_extensions') is None else self.kwargs.get(
                                                                                                'use_auto_extensions'),
                                                                    file_path=_file_path_cluster_partition,
                                                                    kwargs=dict(layout={})
                                                                    )
                                  })

    def _latent_dirichlet_allocation(self):
        """
        Latent Dirichlet Allocation for text clustering
        """
        _clustering: LatentDirichletAllocation = Clustering(cl_params=self.kwargs).latent_dirichlet_allocation()
        _clustering.fit(X=self.df[self.features])
        self.cluster[self.ml_algorithm].update({'fit': _clustering,
                                                'components': _clustering.transform(X=self.df[self.features]),
                                                'em_iter': _clustering.n_batch_iter_,
                                                'passes_iter': _clustering.n_iter_,
                                                'perplexity_score': _clustering.bound_,
                                                'doc_topic_prior': _clustering.doc_topic_prior_,
                                                'topic_word_prior': _clustering.topic_word_prior_,
                                                })

    def _locally_linear_embedding(self):
        """
        Locally linear embedding for non-linear dimensionality reduction (manifold learning)
        """
        if self.kwargs.get('n_components') is None:
            self.kwargs.update({'n_components': 2})
        _clustering: LocallyLinearEmbedding = Clustering(cl_params=self.kwargs).locally_linear_embedding()
        _clustering.fit(X=self.df[self.features])
        self.cluster[self.ml_algorithm].update({'fit': _clustering,
                                                'n_components': self.kwargs.get('n_components'),
                                                'embeddings': _clustering.embedding_,
                                                'transformed_embeddings': _clustering.transform(
                                                    X=self.df[self.features]),
                                                'reconstruction_error': _clustering.reconstruction_error_
                                                })

    def _multi_dimensional_scaling(self):
        """
        Multi-dimensional scaling (MDS)
        """
        if self.n_cluster_components is None:
            self.kwargs.update({'n_components': 3})
        else:
            self.kwargs.update({'n_components': self.n_cluster_components})
        _clustering: MDS = Clustering(cl_params=self.kwargs).multi_dimensional_scaling()
        _clustering.fit(X=self.df[self.features])
        self.cluster[self.ml_algorithm].update({'fit': _clustering,
                                                'n_components': self.kwargs.get('n_components'),
                                                'embeddings': _clustering.embedding_,
                                                'dissimilarity_matrix': _clustering.dissimilarity_matrix_,
                                                'stress': _clustering.stress_,
                                                'n_iter': _clustering.n_iter_
                                                })

    def _non_negative_matrix_factorization(self):
        """
        Non-Negative Matrix Factorization for text clustering
        """
        _clustering: NMF = Clustering(cl_params=self.kwargs).non_negative_matrix_factorization()
        _clustering.fit(X=self.df[self.features])
        self.cluster[self.ml_algorithm].update({'fit': _clustering,
                                                'factorization_matrix_w': _clustering.transform(X=self.df[self.features]),
                                                'factorization_matrix_h': _clustering.components_,
                                                'reconstruction_error': _clustering.reconstruction_err_,
                                                'n_iter': _clustering.n_iter_
                                                })

    def _ordering_points_to_identify_clustering_structure(self):
        """
        Ordering points to identify the clustering structure (OPTICS) for clustering complex structures like dense regions in space
        """
        _clustering: OPTICS = Clustering(cl_params=self.kwargs).optics()
        _clustering.fit(X=self.df[self.features])
        self.cluster[self.ml_algorithm].update({'fit': _clustering,
                                                'n_clusters': len(list(set(_clustering.labels_)))
                                                })
        self.kwargs.update({'n_clusters': self.cluster[self.ml_algorithm].get('n_clusters')})
        if self.to_export:
            _file_path_silhouette: str = os.path.join(self.kwargs.get('file_path'), 'optics_silhouette.html')
            _file_path_cluster_partition: str = os.path.join(self.kwargs.get('file_path'),
                                                             'optics_cluster_partition.html')
            _file_path_cluster_reachability: str = os.path.join(self.kwargs.get('file_path'),
                                                                'optics_cluster_reachability.html')
        else:
            _file_path_silhouette: str = None
            _file_path_cluster_partition: str = None
            _file_path_cluster_reachability: str = None
        if self.find_optimum:
            if self.silhouette:
                _silhouette: dict = self.silhouette_analysis(labels=_clustering.labels_)
                self.cluster[self.ml_algorithm].update({'silhouette': _silhouette})
                self.cluster_plot.update({'OPTICS: Silhouette Analysis': dict(data=self.df,
                                                                              features=None,
                                                                              plot_type='silhouette',
                                                                              use_auto_extensions=False if self.kwargs.get(
                                                                                                'use_auto_extensions') is None else self.kwargs.get(
                                                                                                'use_auto_extensions'),
                                                                              file_path=_file_path_silhouette,
                                                                              kwargs=dict(layout={},
                                                                                          n_clusters=self.cluster[self.ml_algorithm].get('n_clusters'),
                                                                                          silhouette=_silhouette
                                                                                          )
                                                                              )
                                          })
        if 'silhouette' not in self.cluster[self.ml_algorithm].keys():
            self.cluster[self.ml_algorithm].update({'silhouette': None})
        self.cluster[self.ml_algorithm].update({'reachability': _clustering.reachability_,
                                                'ordering': _clustering.ordering_,
                                                'core_distances': _clustering.core_distances_,
                                                'predecessor': _clustering.predecessor_,
                                                'cluster_hierarchy': _clustering.cluster_hierarchy_,
                                                'labels': _clustering.labels_
                                                })
        _reachability: pd.DataFrame = pd.DataFrame(
            data={'reachability': self.cluster[self.ml_algorithm].get('reachability')[self.cluster[self.ml_algorithm].get('ordering')],
                  'labels': self.cluster[self.ml_algorithm].get('labels')[self.cluster[self.ml_algorithm].get('ordering')]
                  }
            )
        _reachability = _reachability.replace(to_replace=[np.inf, -np.inf], value=np.nan, inplace=False)
        _reachability = _reachability.dropna(axis=0, how='any', inplace=False)
        _df: pd.DataFrame = self.df
        _df['cluster'] = self.cluster[self.ml_algorithm].get('labels')
        self.cluster_plot.update({'OPTICS: Cluster Partition': dict(data=_df,
                                                                    features=self.features,
                                                                    group_by=['cluster'],
                                                                    melt=True if self.kwargs.get('melt') is None else self.kwargs.get('melt'),
                                                                    plot_type='scatter',
                                                                    use_auto_extensions=False if self.kwargs.get(
                                                                                                'use_auto_extensions') is None else self.kwargs.get(
                                                                                                'use_auto_extensions'),
                                                                    file_path=_file_path_cluster_partition,
                                                                    kwargs=dict(layout={})
                                                                    ),
                                  'OPTICS: Reachability': dict(data=_reachability,
                                                               features=['reachability'],
                                                               group_by=['labels'],
                                                               melt=True if self.kwargs.get('melt') is None else self.kwargs.get('melt'),
                                                               plot_type='hist',
                                                               use_auto_extensions=False if self.kwargs.get(
                                                                                                'use_auto_extensions') is None else self.kwargs.get(
                                                                                                'use_auto_extensions'),
                                                               file_path=_file_path_cluster_reachability,
                                                               kwargs=dict(layout={})
                                                               )
                                  })

    def _principal_component_analysis(self):
        """
        Principal component analysis (PCA)
        """
        if self.n_cluster_components is None:
            self.kwargs.update({'n_components': 2})
        else:
            if self.n_cluster_components >= len(self.features):
                self.kwargs.update({'n_components': 2})
                Log(write=False, level='info').log(
                    msg='Number of components are greater than or equal to number of features. Number of components are set to 2')
            else:
                self.kwargs.update({'n_components': self.n_cluster_components})
        _clustering: PCA = Clustering(cl_params=self.kwargs).principal_component_analysis()
        _clustering.fit(X=self.df[self.features])
        self.cluster[self.ml_algorithm].update({'fit': _clustering,
                                                'n_components': self.kwargs.get('n_components'),
                                                'explained_variance_ratio': None,
                                                'cumulative_explained_variance_ratio': None
                                                })
        if self.to_export:
            _file_path_onc: str = os.path.join(self.kwargs.get('file_path'), 'pca_optimal_number_of_components.html')
            _file_path_explained_variance: str = os.path.join(self.kwargs.get('file_path'), 'pca_explained_variance.html')
            _file_path_pca: str = os.path.join(self.kwargs.get('file_path'), 'pca_components.html')
        else:
            _file_path_onc: str = None
            _file_path_explained_variance: str = None
            _file_path_pca: str = None
        if self.find_optimum:
            _cumulative_explained_variance_ratio: np.ndarray = np.cumsum(_clustering.explained_variance_ratio_)
            _cumulative_variance: pd.DataFrame = pd.DataFrame(data=_cumulative_explained_variance_ratio,
                                                              columns=['cumulative_explained_variance'],
                                                              index=[i for i in
                                                                     range(0, self.kwargs.get('n_components'), 1)]
                                                              )
            _cumulative_variance['component'] = _cumulative_variance.index.values.tolist()
            self.cluster[self.ml_algorithm].update({'explained_variance_ratio': _clustering.explained_variance_ratio_})
            self.cluster[self.ml_algorithm].update({'cumulative_explained_variance_ratio': _cumulative_explained_variance_ratio})
            self.kwargs.update({'n_components': self._cumulative_explained_variance_ratio(
                explained_variance_ratio=_cumulative_explained_variance_ratio)})
            self.cluster[self.ml_algorithm].update({'n_components': self.kwargs.get('n_components')})
            self.cluster_plot.update({'PCA: Optimal Number of Components': dict(data=_cumulative_variance,
                                                                                features=['cumulative_explained_variance'],
                                                                                time_features=['component'],
                                                                                plot_type='line',
                                                                                use_auto_extensions=False if self.kwargs.get(
                                                                                                'use_auto_extensions') is None else self.kwargs.get(
                                                                                                'use_auto_extensions'),
                                                                                file_path=_file_path_onc,
                                                                                kwargs=dict(layout={})
                                                                                )
                                      })
            _clustering: PCA = Clustering(cl_params=self.kwargs).principal_component_analysis()
            _clustering.fit(X=self.df[self.features])
            self.cluster[self.ml_algorithm].update({'fit': _clustering,
                                                    'n_components': self.kwargs.get('n_components')
                                                    })
        _components: pd.DataFrame = pd.DataFrame(data=np.array(_clustering.components_),
                                                 columns=self.features,
                                                 index=['pc{}'.format(pc) for pc in
                                                        range(0, self.kwargs.get('n_components'), 1)]
                                                 ).transpose()
        _feature_importance: pd.DataFrame = abs(_components)
        self.cluster[self.ml_algorithm].update({'components': _clustering.components_,
                                                'explained_variance': list(_clustering.explained_variance_),
                                                'explained_variance_ratio': list(_clustering.explained_variance_ratio_),
                                                'pc': _clustering.transform(X=self.df[self.features]),
                                                'feature_importance': dict(names={pca: _feature_importance[pca].sort_values(axis=0, ascending=False).index.values[0] for pca in _feature_importance.keys()},
                                                                           scores=_feature_importance
                                                                           )
                                                })
        for pca in range(0, self.kwargs.get('n_components'), 1):
            self.cluster_plot.update({'PCA: Feature Importance PC{}'.format(pca): dict(data=_feature_importance,
                                                                                       features=None,
                                                                                       plot_type='bar',
                                                                                       use_auto_extensions=False if self.kwargs.get(
                                                                                                'use_auto_extensions') is None else self.kwargs.get(
                                                                                                'use_auto_extensions'),
                                                                                       file_path=os.path.join(self.kwargs.get('file_path'), f'pca_feature_importance_{pca}.html') if self.to_export else None,
                                                                                       kwargs=dict(layout={},
                                                                                                   x=self.features,
                                                                                                   y=_feature_importance[f'pc{pca}'],
                                                                                                   marker=dict(
                                                                                                       color=_feature_importance[f'pc{pca}'],
                                                                                                       colorscale='rdylgn',
                                                                                                       autocolorscale=True
                                                                                                   )
                                                                                                   )
                                                                                       )
                                      })
        self.cluster_plot.update({'PCA: Explained Variance': dict(data=pd.DataFrame(),
                                                                  features=None,
                                                                  plot_type='bar',
                                                                  use_auto_extensions=False if self.kwargs.get(
                                                                                                'use_auto_extensions') is None else self.kwargs.get(
                                                                                                'use_auto_extensions'),
                                                                  file_path=_file_path_explained_variance,
                                                                  kwargs=dict(layout={},
                                                                              x=list(_feature_importance.keys()),
                                                                              y=self.cluster[self.ml_algorithm].get('explained_variance_ratio')
                                                                              )
                                                                  ),
                                  'PCA: Principal Components': dict(data=pd.DataFrame(data=self.cluster[self.ml_algorithm].get('pc'),
                                                                                      columns=list(_feature_importance.keys())
                                                                                      ),
                                                                    features=list(_feature_importance.keys()),
                                                                    plot_type='scatter',
                                                                    melt=True if self.kwargs.get('melt') is None else self.kwargs.get('melt'),
                                                                    use_auto_extensions=False if self.kwargs.get(
                                                                                                'use_auto_extensions') is None else self.kwargs.get(
                                                                                                'use_auto_extensions'),
                                                                    file_path=_file_path_pca,
                                                                    kwargs=dict(layout={},
                                                                                marker=dict(color=self.cluster[self.ml_algorithm].get('pc'),
                                                                                            colorscale='rdylgn',
                                                                                            autocolorscale=True
                                                                                            )
                                                                                )
                                                                    )
                                  })

    def _spectral_clustering(self):
        """
        Spectral clustering for dimensionality reduction of graphical and non-graphical data
        """
        _clustering: SpectralClustering = Clustering(cl_params=self.kwargs).spectral_clustering()
        _clustering.fit(X=self.df[self.features])
        self.cluster[self.ml_algorithm].update({'fit': _clustering,
                                                'n_clusters': len(list(set(_clustering.labels_)))
                                                })
        self.kwargs.update({'n_clusters': self.cluster[self.ml_algorithm].get('n_clusters')})
        if self.to_export:
            _file_path_silhouette: str = os.path.join(self.kwargs.get('file_path'), 'spectral_clustering_silhouette.html')
            _file_path_cluster_partition: str = os.path.join(self.kwargs.get('file_path'), 'spectral_clustering_cluster_partition.html')
        else:
            _file_path_silhouette: str = None
            _file_path_cluster_partition: str = None
        if self.find_optimum:
            if self.silhouette:
                _silhouette: dict = self.silhouette_analysis(labels=_clustering.labels_)
                self.cluster[self.ml_algorithm].update({'silhouette': _silhouette})
                self.cluster_plot.update({'Spectral Clustering: Silhouette Analysis': dict(data=self.df,
                                                                                           features=None,
                                                                                           plot_type='silhouette',
                                                                                           use_auto_extensions=False if self.kwargs.get(
                                                                                                'use_auto_extensions') is None else self.kwargs.get(
                                                                                                'use_auto_extensions'),
                                                                                           file_path=_file_path_silhouette,
                                                                                           kwargs=dict(layout={},
                                                                                                       n_clusters=self.cluster[self.ml_algorithm].get('n_clusters'),
                                                                                                       silhouette=_silhouette
                                                                                                       )
                                                                                           )
                                          })
        if 'silhouette' not in self.cluster[self.ml_algorithm].keys():
            self.cluster[self.ml_algorithm].update({'silhouette': None})
        self.cluster[self.ml_algorithm].update({'fit': _clustering,
                                                'affinity_matrix': _clustering.affinity_matrix_,
                                                'labels': _clustering.labels_
                                                })
        _df: pd.DataFrame = self.df
        _df['cluster'] = self.cluster[self.ml_algorithm].get('labels')
        self.cluster_plot.update({'Spectral Clustering: Partition': dict(data=_df,
                                                                         features=self.features,
                                                                         group_by=['cluster'],
                                                                         melt=True if self.kwargs.get('melt') is None else self.kwargs.get('melt'),
                                                                         plot_type='scatter',
                                                                         use_auto_extensions=False if self.kwargs.get(
                                                                                                'use_auto_extensions') is None else self.kwargs.get(
                                                                                                'use_auto_extensions'),
                                                                         file_path=_file_path_cluster_partition,
                                                                         kwargs=dict(layout={})
                                                                         )
                                  })

    def _spectral_embedding(self):
        """
        Spectral embedding
        """
        if self.kwargs.get('n_components') is None:
            self.kwargs.update({'n_components': 2})
        _clustering: SpectralEmbedding = Clustering(cl_params=self.kwargs).spectral_embedding()
        _clustering.fit(X=self.df[self.features])
        self.cluster[self.ml_algorithm].update({'fit': _clustering,
                                                'n_components': self.kwargs.get('n_components'),
                                                'embeddings': _clustering.embedding_,
                                                'affinity_matrix': _clustering.affinity_matrix_
                                                })

    def _t_distributed_stochastic_neighbor_embedding(self):
        """
        T-distributed stochastic neighbor embedding (TSNE)
        """
        if self.kwargs.get('n_components') is None:
            self.kwargs.update({'n_components': 2})
        _clustering: TSNE = Clustering(cl_params=self.kwargs).t_distributed_stochastic_neighbor_embedding()
        _clustering.fit(X=self.df[self.features])
        self.cluster[self.ml_algorithm].update({'fit': _clustering,
                                                'n_components': self.kwargs.get('n_components'),
                                                'embeddings': _clustering.embedding_
                                                })

    def _truncated_single_value_decomposition(self):
        """
        Truncated single value decomposition (TSVD / SVD)
        """
        if self.n_cluster_components is None:
            self.kwargs.update({'n_components': 2})
        else:
            if self.n_cluster_components >= len(self.features):
                self.kwargs.update({'n_components': 2})
                Log(write=False, level='info').log(
                    msg='Number of components are greater than or equal to number of features. Number of components set to 2')
            else:
                self.kwargs.update({'n_components': self.n_cluster_components})
        _clustering: TruncatedSVD = Clustering(cl_params=self.kwargs).truncated_single_value_decomp()
        _clustering.fit(X=self.df[self.features])
        self.cluster[self.ml_algorithm].update({'fit': _clustering,
                                                'n_components': self.kwargs.get('n_components'),
                                                'explained_variance_ratio': None,
                                                'cumulative_explained_variance_ratio': None
                                                })
        if self.to_export:
            _file_path_onc: str = os.path.join(self.kwargs.get('file_path'), 'svd_optimal_number_of_components.html')
            _file_path_explained_variance: str = os.path.join(self.kwargs.get('file_path'), 'svd_explained_variance.html')
            _file_path_pca: str = os.path.join(self.kwargs.get('file_path'), 'svd_components.html')
        else:
            _file_path_onc: str = None
            _file_path_explained_variance: str = None
            _file_path_pca: str = None
        if self.find_optimum:
            _cumulative_explained_variance_ratio: np.ndarray = np.cumsum(_clustering.explained_variance_ratio_)
            _cumulative_variance: pd.DataFrame = pd.DataFrame(data=_cumulative_explained_variance_ratio,
                                                              columns=['cumulative_explained_variance'],
                                                              index=[i for i in
                                                                     range(0, self.kwargs.get('n_components'), 1)]
                                                              )
            _cumulative_variance['component'] = _cumulative_variance.index.values.tolist()
            self.cluster[self.ml_algorithm].update(
                {'explained_variance_ratio': _clustering.explained_variance_ratio_})
            self.cluster[self.ml_algorithm].update(
                {'cumulative_explained_variance_ratio': _cumulative_explained_variance_ratio})
            self.kwargs.update({'n_components': self._cumulative_explained_variance_ratio(
                explained_variance_ratio=_cumulative_explained_variance_ratio)})
            self.cluster[self.ml_algorithm].update({'n_components': self.kwargs.get('n_components')})
            self.cluster_plot.update({'SVD: Optimal Number of Components': dict(data=_cumulative_variance,
                                                                                features=['cumulative_explained_variance'],
                                                                                time_features=['component'],
                                                                                plot_type='line',
                                                                                use_auto_extensions=False if self.kwargs.get(
                                                                                                'use_auto_extensions') is None else self.kwargs.get(
                                                                                                'use_auto_extensions'),
                                                                                file_path=_file_path_pca,
                                                                                kwargs=dict(layout={})
                                                                                )
                                      })
            _clustering: TruncatedSVD = Clustering(cl_params=self.kwargs).truncated_single_value_decomp()
            _clustering.fit(X=self.df[self.features])
            self.cluster[self.ml_algorithm].update({'fit': _clustering,
                                                    'n_components': self.kwargs.get('n_components')
                                                    })
        _components: pd.DataFrame = pd.DataFrame(data=np.array(_clustering.components_),
                                                 columns=self.features,
                                                 index=['svd{}'.format(svd) for svd in
                                                        range(0, self.kwargs.get('n_components'), 1)]
                                                 ).transpose()
        _feature_importance: pd.DataFrame = abs(_components)
        self.cluster[self.ml_algorithm].update({'components': _clustering.components_,
                                                'explained_variance': list(_clustering.explained_variance_),
                                                'explained_variance_ratio': list(_clustering.explained_variance_ratio_),
                                                'pc': _clustering.transform(X=self.df[self.features]),
                                                'feature_importance': dict(
                                                    names={c: _feature_importance[c].sort_values(axis=0, ascending=False).index.values[0]
                                                           for c in _feature_importance.keys()},
                                                    scores=_feature_importance
                                                )
                                                })
        for svd in range(0, self.kwargs.get('n_components'), 1):
            self.cluster_plot.update({f'SVD: Feature Importance PC{svd}': dict(data=_feature_importance,
                                                                               features=None,
                                                                               plot_type='bar',
                                                                               use_auto_extensions=False if self.kwargs.get(
                                                                                                'use_auto_extensions') is None else self.kwargs.get(
                                                                                                'use_auto_extensions'),
                                                                               file_path=os.path.join(self.kwargs.get('file_path'), f'svd_feature_importance_{svd}.html') if self.to_export else None,
                                                                               kwargs=dict(layout={},
                                                                                           x=self.features,
                                                                                           y=_feature_importance[
                                                                                               'svd{}'.format(svd)],
                                                                                           marker=dict(
                                                                                               color=_feature_importance[
                                                                                                   'svd{}'.format(svd)],
                                                                                               colorscale='rdylgn',
                                                                                               autocolorscale=True
                                                                                           )
                                                                                           )
                                                                               )
                                      })
        self.cluster_plot.update({'SVD: Explained Variance': dict(data=pd.DataFrame(),
                                                                  features=None,
                                                                  plot_type='bar',
                                                                  use_auto_extensions=False if self.kwargs.get(
                                                                                                'use_auto_extensions') is None else self.kwargs.get(
                                                                                                'use_auto_extensions'),
                                                                  file_path=_file_path_explained_variance,
                                                                  kwargs=dict(layout={},
                                                                              x=list(_feature_importance.keys()),
                                                                              y=self.cluster[self.ml_algorithm].get('explained_variance_ratio')
                                                                              )
                                                                  ),
                                  'SVD: Principal Components': dict(data=pd.DataFrame(data=self.cluster[self.ml_algorithm].get('pc'),
                                                                                 columns=list(_feature_importance.keys())
                                                                                 ),
                                                                    features=list(_feature_importance.keys()),
                                                                    melt=True if self.kwargs.get('melt') is None else self.kwargs.get('melt'),
                                                                    plot_type='scatter',
                                                                    use_auto_extensions=False if self.kwargs.get(
                                                                                                'use_auto_extensions') is None else self.kwargs.get(
                                                                                                'use_auto_extensions'),
                                                                    file_path=_file_path_pca,
                                                                    kwargs=dict(layout={},
                                                                                marker=dict(color=self.cluster[self.ml_algorithm].get('pc'),
                                                                                            colorscale='rdylgn',
                                                                                            autocolorscale=True
                                                                                            )
                                                                                )
                                                                    )
                                  })

    def run_clustering(self, cluster_algorithms: List[str], clean_missing_data: bool = False):
        """
        Run clustering algorithms

        :param cluster_algorithms: List[str]
            Names of the cluster algorithms

        :param clean_missing_data: bool
            Whether to clean cases containing missing data or not
        """
        _cluster: dict = {}
        _cluster_plot: dict = {}
        if clean_missing_data:
            self._clean_missing_data()
        for cl in cluster_algorithms:
            self.ml_algorithm = cl
            self.cluster.update({cl: {}})
            ################################
            # Principal Component Analysis #
            ################################
            if cl == 'pca':
                self._principal_component_analysis()
            ###################
            # Factor Analysis #
            ###################
            elif cl in ['fa', 'factor']:
                self._factor_analysis()
            ########################################
            # Truncated Single Value Decomposition #
            ########################################
            elif cl in ['svd', 'tsvd']:
                self._truncated_single_value_decomposition()
            ###############################################
            # t-Distributed Stochastic Neighbor Embedding #
            ###############################################
            elif cl == 'tsne':
                self._t_distributed_stochastic_neighbor_embedding()
            #############################
            # Multi Dimensional Scaling #
            #############################
            elif cl == 'mds':
                self._multi_dimensional_scaling()
            #####################
            # Isometric Mapping #
            #####################
            elif cl == 'isomap':
                self._isometric_mapping()
            ######################
            # Spectral Embedding #
            ######################
            elif cl in ['spectral_emb', 'spectral_embedding']:
                self._spectral_embedding()
            ############################
            # Locally Linear Embedding #
            ############################
            elif cl in ['lle', 'locally_emb', 'locally_linear', 'locally_embedding']:
                self._locally_linear_embedding()
            ###########
            # K-Means #
            ###########
            elif cl == 'kmeans':
                self._k_means()
            #####################################
            # Non-Negative Matrix Factorization #
            #####################################
            elif cl == 'nmf':
                self._non_negative_matrix_factorization()
            ###############################
            # Latent Dirichlet Allocation #
            ###############################
            elif cl == 'lda':
                self._latent_dirichlet_allocation()
            ########################################################
            # Ordering Points To Identify the Clustering Structure #
            ########################################################
            elif cl == 'optics':
                self._ordering_points_to_identify_clustering_structure()
            ###############################################################
            # Density-Based Spatial Clustering of Applications with Noise #
            ###############################################################
            elif cl == 'dbscan':
                self._density_based_spatial_clustering_applications_with_noise()
            #######################
            # Spectral Clustering #
            #######################
            elif cl in ['spectral_cl', 'spectral_cluster']:
                self._spectral_clustering()
            #########################
            # Feature Agglomeration #
            #########################
            elif cl in ['feature_agglo', 'feature_agglomeration']:
                self._feature_agglomeration()
            ############################
            # Agglomerative Clustering #
            ############################
            elif cl in ['agglo_cl', 'agglo_cluster', 'struc_agglo_cl', 'struc_agglo_cluster', 'unstruc_agglo_cl', 'unstruc_agglo_cluster']:
                self._agglomerative_clustering()
            ####################
            # Birch Clustering #
            ####################
            elif cl == 'birch':
                self._birch_clustering()
            ########################
            # Affinity Propagation #
            ########################
            elif cl in ['affinity_prop', 'affinity_propagation']:
                self._affinity_propagation()
            else:
                raise UnsupervisedMLException(f'Clustering algorithm ({cl}) not supported')
        if self.plot:
            DataVisualizer(subplots=self.cluster_plot,
                           interactive=True,
                           height=500,
                           width=500,
                           unit='px'
                           ).run()

    def silhouette_analysis(self, labels: List[int]) -> dict:
        """
        Calculate silhouette scores to evaluate optimal amount of clusters for most cluster analysis algorithms

        :param labels: List[int]
            Predicted cluster labels by any cluster algorithm

        :return: dict
            Optimal clusters and the average silhouette score as well as silhouette score for each sample
        """
        _lower: int = 10
        _silhouette: dict = {}
        _avg_silhoutte_score: List[float] = []
        if self.kwargs.get('n_clusters') is None:
            if self.n_cluster_components is None:
                self.kwargs.update({'n_clusters': 2})
            else:
                if self.n_cluster_components < 2:
                    Log(write=False, level='info').log(
                        msg='It makes no sense to run cluster analysis with less than 2 clusters ({}). Run analysis with more than 1 cluster instead'.format(
                            self.kwargs.get('n_clusters')))
                    self.kwargs.update({'n_clusters': 2})
                else:
                    self.kwargs.update({'n_clusters': self.n_cluster_components})
        _clusters: List[int] = [n for n in range(0, self.kwargs.get('n_clusters'), 1)]
        for cl in _clusters:
            _avg_silhoutte_score.append(silhouette_score(X=self.df[self.features],
                                                         labels=labels,
                                                         metric='euclidean' if self.metric is None else self.metric,
                                                         sample_size=self.kwargs.get('sample_size'),
                                                         random_state=self.seed
                                                         )
                                        )
            _silhouette.update({'cluster_{}_avg'.format(cl): _avg_silhoutte_score[-1]})
            _silhouette_samples: np.array = silhouette_samples(X=self.df[self.features],
                                                               labels=labels,
                                                               metric='euclidean' if self.metric is None else self.metric,
                                                               )
            for s in range(0, cl + 1, 1):
                _s: np.array = _silhouette_samples[labels == s]
                _s.sort()
                _upper: int = _lower + _s.shape[0]
                _silhouette.update({'cluster_{}_samples'.format(cl): dict(y=np.arange(_lower, _upper), scores=_s)})
                _lower = _upper + 10
        _max_avg_score: float = max(_avg_silhoutte_score)
        _silhouette.update({'best': dict(cluster=_avg_silhoutte_score.index(_max_avg_score) + 1, avg_score=_max_avg_score)})
        return _silhouette
