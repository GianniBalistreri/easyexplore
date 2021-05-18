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
        return FactorAnalysis(n_components=None if self.cl_params.get('n_components') else self.cl_params.get('n_components'),
                              tol=0.01 if self.cl_params.get('tol') else self.cl_params.get('tol'),
                              copy=True if self.cl_params.get('copy') else self.cl_params.get('copy'),
                              max_iter=1000 if self.cl_params.get('max_iter') else self.cl_params.get('max_iter'),
                              noise_variance_init=None if self.cl_params.get('noise_variance_init') else self.cl_params.get('noise_variance_init'),
                              svd_method='randomized' if self.cl_params.get('svd_method') else self.cl_params.get('svd_method'),
                              iterated_power=3 if self.cl_params.get('iterated_power') else self.cl_params.get('iterated_power'),
                              random_state=self.seed
                              )

    def feature_agglomeration(self) -> FeatureAgglomeration:
        """
        Config feature agglomeration clustering

        :return: FeatureAgglomeration
            Sklearn object containing the feature agglomeration configuration
        """
        return FeatureAgglomeration(n_clusters=2 if self.cl_params.get('n_clusters') else self.cl_params.get('n_clusters'),
                                    affinity='euclidean' if self.cl_params.get('affinity') else self.cl_params.get('affinity'),
                                    memory=None if self.cl_params.get('memory') else self.cl_params.get('memory'),
                                    connectivity=None if self.cl_params.get('connectivity') else self.cl_params.get('connectivity'),
                                    compute_full_tree='auto' if self.cl_params.get('compute_full_tree') else self.cl_params.get('compute_full_tree'),
                                    linkage='ward' if self.cl_params.get('linkage') else self.cl_params.get('linkage'),
                                    pooling_func=np.mean if self.cl_params.get('pooling_func') else self.cl_params.get('pooling_func'),
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
                 cluster_algorithms: List[str],
                 features: List[str] = None,
                 find_optimum: bool = False,
                 silhouette_analysis: bool = True,
                 n_cluster_components: int = None,
                 n_neighbors: int = None,
                 n_iter: int = None,
                 metric: List[str] = None,
                 affinity: List[str] = None,
                 connectivity: List[str] = None,
                 linkage: List[str] = None,
                 target: str = None,
                 plot: bool = True,
                 cpu_cores: int = 0,
                 log_path: str = None,
                 **kwargs
                 ):
        """
        :param df: pd.DataFrame
            Data set

        :param features: List[str]
            List of strings containing the features to cluster

        :param target: str
            Name of the target features

        :param cluster_algorithms: List[str]
            Names of the cluster algorithms

        :param find_optimum: bool
            Find optimum number of components or clusters or not

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
        self.target: str = target
        self.cluster_algorithms = cluster_algorithms
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

    def _cummulative_explained_variance_ratio(self, explained_variance_ratio: np.ndarray) -> int:
        """
        Calculate optimal amount of components to be used in principal component analysis based on the explained variance ratio

        :return: int
            Optimal amount of components
        """
        _threshold: float = 0.75 if self.kwargs.get('cev') is None else self.kwargs.get('cev')
        for i, ratio in enumerate(np.cumsum(explained_variance_ratio)):
            if ratio >= _threshold:
                return i + 1

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
        for i in range(1, 10, 1):
            self.kwargs.update({'n_clusters': i})
            _distortions.append(Clustering(cl_params=self.kwargs).kmeans().fit(self.df[self.features]).inertia_)
        return 1

    def _silhoutte_analysis(self, labels: List[int]) -> dict:
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

    def ml_pipeline(self) -> dict:
        """
        Run clustering algorithms
        """
        _cluster: dict = {}
        _cluster_plot: dict = {}
        for cl in self.cluster_algorithms:
            _cluster.update({cl: {}})
            ################################
            # Principal Component Analysis #
            ################################
            if cl is 'pca':
                #if self.df.loc[self.df.isnull(), self.features].shape[0] > 0:
                #    Log(write=False, level='info').log(msg='Clean cases containing missing values...')
                #    self.df = self.df[~self.df.isnull()]
                #    if self.df.shape[0] == 0:
                #        raise UnsupervisedMLException('No cases containing valid observations left')
                if self.n_cluster_components is None:
                    self.kwargs.update({'n_components': 2})
                else:
                    if self.n_cluster_components >= len(self.features):
                        self.kwargs.update({'n_components': 2})
                        Log(write=False, level='info').log(msg='Number of components are greater then or equal to number of features. Number of components set to 2')
                    else:
                        self.kwargs.update({'n_components': self.n_cluster_components})
                if self.find_optimum:
                    if self.silhouette:
                        _try_run = Clustering(cl_params=self.kwargs).principal_component_analysis().fit(X=self.df[self.features])
                        _silhouette: dict = self._silhoutte_analysis(labels=_try_run.transform(self.df[self.features]))
                        _cluster[cl].update({'silhouette': _silhouette})
                        _cluster_plot.update({'Silhouette Analysis (PCA)': dict(data=None,
                                                                                features=None,
                                                                                plot_type='silhouette',
                                                                                kwargs=dict(layout={},
                                                                                            n_clusters=self.kwargs.get('n_clusters'),
                                                                                            silhouette=_silhouette
                                                                                            )
                                                                                )
                                              })
                    else:
                        _try_run = Clustering(cl_params=self.kwargs).principal_component_analysis().fit(X=self.df[self.features])
                        _cummulative_explained_variance_ratio: np.ndarray = np.cumsum(_try_run.explained_variance_ratio_)
                        self.kwargs.update({'n_components': self._cummulative_explained_variance_ratio(explained_variance_ratio=_cummulative_explained_variance_ratio)})
                        _cluster_plot.update({'Optimal Number of Components': dict(data=_cummulative_explained_variance_ratio,
                                                                                   features=None,
                                                                                   plot_type='line',
                                                                                   kwargs=dict(layout={})
                                                                                   )
                                              })
                _cluster[cl].update({'fit': Clustering(cl_params=self.kwargs).principal_component_analysis().fit(X=self.df[self.features])})
                _components: pd.DataFrame = pd.DataFrame(data=np.array(_cluster[cl].get('fit').components_),
                                                         columns=self.features,
                                                         index=['pc{}'.format(pc) for pc in range(0, self.kwargs.get('n_components'), 1)]
                                                         ).transpose()
                _feature_importance: pd.DataFrame = abs(_components)
                _cluster[cl].update({'components': _cluster[cl].get('fit').components_,
                                     'explained_variance': list(_cluster[cl].get('fit').explained_variance_),
                                     'explained_variance_ratio': list(_cluster[cl].get('fit').explained_variance_ratio_),
                                     'pc': _cluster[cl].get('fit').transform(X=self.df[self.features]),
                                     'feature_importance': dict(names={pca: _feature_importance[pca].sort_values(axis=0, ascending=False).index.values[0] for pca in _feature_importance.keys()},
                                                                scores=_feature_importance
                                                                )
                                     })
                for pca in range(0, self.kwargs.get('n_components'), 1):
                    _cluster_plot.update({'Feature Importance PC{}'.format(pca): dict(data=_feature_importance,
                                                                                      features=None,
                                                                                      plot_type='bar',
                                                                                      kwargs=dict(layout={},
                                                                                                  x=self.features,
                                                                                                  y=_feature_importance['pc{}'.format(pca)],
                                                                                                  marker=dict(color=_feature_importance['pc{}'.format(pca)],
                                                                                                              colorscale='rdylgn',
                                                                                                              autocolorscale=True
                                                                                                              )
                                                                                                  )
                                                                                      )
                                          })
                _cluster_plot.update({'Explained Variance': dict(data=pd.DataFrame(),
                                                                 features=None,
                                                                 plot_type='bar',
                                                                 kwargs=dict(layout={},
                                                                             x=list(_feature_importance.keys()),
                                                                             y=_cluster[cl].get('explained_variance_ratio')
                                                                             )
                                                                 ),
                                      'Principal Components': dict(data=pd.DataFrame(data=_cluster[cl].get('pc'),
                                                                                     columns=list(_feature_importance.keys())
                                                                                     ),
                                                                   features=list(_feature_importance.keys()),
                                                                   plot_type='scatter',
                                                                   melt=True,
                                                                   kwargs=dict(layout={},
                                                                               marker=dict(color=_cluster[cl].get('pc'),
                                                                                           colorscale='rdylgn',
                                                                                           autocolorscale=True
                                                                                           )
                                                                               )
                                                                   )
                                      })
            ###################
            # Factor Analysis #
            ###################
            elif cl in ['fa', 'factor']:
                _kmo: dict = StatsUtils(data=self.df, features=self.features).factoriability_test(meth='kmo')
                if _kmo.get('kmo') < 0.6:
                    Log(write=False, level='info').log(msg='Data set not suitable for running factor analysis since KMO coefficient ({}) is lower than 0.6'.format(_kmo.get('kmo')))
                else:
                    if self.n_cluster_components is None:
                        self.kwargs.update({'n_factors': 2})
                    else:
                        if self.n_cluster_components >= len(self.features):
                            self.kwargs.update({'n_components': 2})
                            Log(write=False, level='info').log(msg='Number of factors are greater then or equal to number of features. Number of factors set to 2')
                        else:
                            self.kwargs.update({'n_components': self.n_cluster_components})
                    if self.find_optimum:
                        if self.silhouette:
                            _try_run = Clustering(cl_params=self.kwargs).factor_analysis().fit(X=self.df[self.features])
                            _silhouette: dict = self._silhoutte_analysis(labels=_try_run.transform(self.df[self.features]))
                            _cluster[cl].update({'silhouette': _silhouette})
                            _cluster_plot.update({'Silhouette Analysis (FA)': dict(data=None,
                                                                                   features=None,
                                                                                   plot_type='silhouette',
                                                                                   kwargs=dict(layout={},
                                                                                               n_clusters=self.kwargs.get('n_clusters'),
                                                                                               silhouette=_silhouette
                                                                                               )
                                                                                   )
                                                  })
                        else:
                            _try_run = Clustering(cl_params=self.kwargs).factor_analysis().fit(X=self.df[self.features])
                            self.kwargs.update({'n_factors': self._estimate_optimal_factors(factors=_try_run.transform(X=self.df[self.features]))})
                            _cluster_plot.update({'Optimal Number of Factors': dict(data=self.eigen_value,
                                                                                    features=None,
                                                                                    plot_type='line',
                                                                                    kwargs=dict(layout={})
                                                                                    )
                                                  })
                    _cluster[cl].update({'fit': Clustering(cl_params=self.kwargs).factor_analysis().fit(X=self.df[self.features], y=self.target)})
                    _factors: np.array = _cluster[cl].get('fit').transform(X=self.df[self.features])
                    _cluster[cl].update({'factors': _cluster[cl].get('fit').components_,
                                         'explained_variance': _cluster[cl].get('fit').explained_variance_,
                                         'fa': _factors
                                         })
                    _components: pd.DataFrame = pd.DataFrame(data=np.array(_cluster[cl].get('fit').components_),
                                                             columns=self.features,
                                                             index=['fa{}'.format(fa) for fa in range(0, self.kwargs.get('n_factors'), 1)]
                                                             ).transpose()
                    _feature_importance: pd.DataFrame = abs(_components)
                    for fa in range(0, self.kwargs.get('n_factors'), 1):
                        _cluster_plot.update({'Feature Importance FA{}'.format(fa): dict(data=_feature_importance,
                                                                                         features=None,
                                                                                         plot_type='bar',
                                                                                         kwargs=dict(layout={},
                                                                                                     x=self.features,
                                                                                                     y=_feature_importance['fa{}'.format(fa)],
                                                                                                     marker=dict(color=_feature_importance['fa{}'.format(fa)],
                                                                                                                 colorscale='rdylgn',
                                                                                                                 autocolorscale=True
                                                                                                                 )
                                                                                                     )
                                                                                         )
                                              })
                    _cluster_plot.update({'Explained Variance': dict(data=pd.DataFrame(),
                                                                     features=None,
                                                                     plot_type='bar',
                                                                     kwargs=dict(layout={},
                                                                                 x=list(_feature_importance.keys()),
                                                                                 y=_cluster[cl].get('explained_variance_ratio')
                                                                                 )
                                                                     ),
                                          'Factor Loadings': dict(data=pd.DataFrame(data=_cluster[cl].get('fa'),
                                                                                    columns=list(_feature_importance.keys())
                                                                                    ),
                                                                  features=list(_feature_importance.keys()),
                                                                  plot_type='scatter',
                                                                  melt=True,
                                                                  kwargs=dict(layout={},
                                                                              marker=dict(color=_cluster[cl].get('fa'),
                                                                                          colorscale='rdylgn',
                                                                                          autocolorscale=True
                                                                                          )
                                                                              )
                                                                  )
                                          })
            ########################################
            # Truncated Single Value Decomposition #
            ########################################
            elif cl in ['svd', 'tsvd']:
                if self.n_cluster_components is None:
                    self.kwargs.update({'n_components': 2})
                else:
                    if self.n_cluster_components >= len(self.features):
                        self.kwargs.update({'n_components': 2})
                        Log(write=False, level='info').log(msg='Number of components are greater then or equal to number of features. Number of components set to 2')
                    else:
                        self.kwargs.update({'n_components': self.n_cluster_components})
                if self.find_optimum:
                    if self.silhouette:
                        _try_run = Clustering(cl_params=self.kwargs).truncated_single_value_decomp().fit(X=self.df[self.features], y=None)
                        _silhouette: dict = self._silhoutte_analysis(labels=_try_run.transform(self.df[self.features]))
                        _cluster[cl].update({'silhouette': _silhouette})
                        _cluster_plot.update({'Silhouette Analysis (SVD)': dict(data=None,
                                                                                features=None,
                                                                                plot_type='silhouette',
                                                                                kwargs=dict(layout={},
                                                                                            n_clusters=self.kwargs.get('n_clusters'),
                                                                                            silhouette=_silhouette
                                                                                            )
                                                                                )
                                              })
                    else:
                        _try_run = Clustering(cl_params=self.kwargs).truncated_single_value_decomp().fit(X=self.df[self.features], y=None)
                        _cummulative_explained_variance_ratio: np.ndarray = np.cumsum(_try_run.explained_variance_ratio_)
                        self.kwargs.update({'n_components': self._cummulative_explained_variance_ratio(explained_variance_ratio=_cummulative_explained_variance_ratio)})
                        _cluster_plot.update({'Optimal Number of Components': dict(data=_cummulative_explained_variance_ratio,
                                                                                   features=None,
                                                                                   plot_type='line',
                                                                                   kwargs=dict(layout={})
                                                                                   )
                                              })
                _cluster[cl].update({'fit': Clustering(cl_params=self.kwargs).truncated_single_value_decomp().fit(X=self.df[self.features], y=self.target)})
                _components: pd.DataFrame = pd.DataFrame(data=np.array(_cluster[cl].get('fit').components_),
                                                         columns=self.features,
                                                         index=['c{}'.format(pc) for pc in range(0, self.kwargs.get('n_components'), 1)]
                                                         ).transpose()
                _feature_importance: pd.DataFrame = abs(_components)
                _cluster[cl].update({'components': _cluster[cl].get('fit').components_,
                                     'explained_variance': list(_cluster[cl].get('fit').explained_variance_),
                                     'explained_variance_ratio': list(_cluster[cl].get('fit').explained_variance_ratio_),
                                     'pc': _cluster[cl].get('fit').transform(X=self.df[self.features]),
                                     'feature_importance': dict(names={c: _feature_importance[c].sort_values(axis=0, ascending=False).index.values[0] for c in _feature_importance.keys()},
                                                                scores=_feature_importance
                                                                )
                                     })
                for svd in range(0, self.kwargs.get('n_components'), 1):
                    _cluster_plot.update({'Feature Importance PC{}'.format(svd): dict(data=_feature_importance,
                                                                                      features=None,
                                                                                      plot_type='bar',
                                                                                      kwargs=dict(layout={},
                                                                                                  x=self.features,
                                                                                                  y=_feature_importance['svd{}'.format(svd)],
                                                                                                  marker=dict(color=_feature_importance['svd{}'.format(pca)],
                                                                                                              colorscale='rdylgn',
                                                                                                              autocolorscale=True
                                                                                                              )
                                                                                                  )
                                                                                      )
                                          })
                _cluster_plot.update({'Explained Variance': dict(data=pd.DataFrame(),
                                                                 features=None,
                                                                 plot_type='bar',
                                                                 kwargs=dict(layout={},
                                                                             x=list(_feature_importance.keys()),
                                                                             y=_cluster[cl].get('explained_variance_ratio')
                                                                             )
                                                                 ),
                                      'Principal Components': dict(data=pd.DataFrame(data=_cluster[cl].get('pc'),
                                                                                     columns=list(_feature_importance.keys())
                                                                                     ),
                                                                   features=list(_feature_importance.keys()),
                                                                   plot_type='scatter',
                                                                   melt=True,
                                                                   kwargs=dict(layout={},
                                                                               marker=dict(color=_cluster[cl].get('pc'),
                                                                                           colorscale='rdylgn',
                                                                                           autocolorscale=True
                                                                                           )
                                                                               )
                                                                   )
                                      })
            ###############################################
            # t-Distributed Stochastic Neighbor Embedding #
            ###############################################
            elif cl is 'tsne':
                if self.kwargs.get('n_components') is None:
                    self.kwargs.update({'n_components': 2})
                if self.find_optimum:
                    if self.silhouette:
                        _try_run = Clustering(cl_params=self.kwargs).t_distributed_stochastic_neighbor_embedding().fit(X=self.df[self.features], y=self.target)
                        _silhouette: dict = self._silhoutte_analysis(labels=_try_run.transform(self.df[self.features]))
                        _cluster[cl].update({'silhouette': _silhouette})
                        _cluster_plot.update({'Silhouette Analysis (TSNE)': dict(data=None,
                                                                                 features=None,
                                                                                 plot_type='silhouette',
                                                                                 kwargs=dict(layout={},
                                                                                             n_clusters=self.kwargs.get('n_clusters'),
                                                                                             silhouette=_silhouette
                                                                                             )
                                                                                 )
                                              })
                _cluster[cl].update({'fit': Clustering(cl_params=self.kwargs).t_distributed_stochastic_neighbor_embedding().fit(X=self.df[self.features], y=self.target)})
                _embeddings: pd.DataFrame = pd.DataFrame(data=np.array(_cluster[cl].get('fit').embedding_),
                                                         columns=self.features,
                                                         index=['emb{}'.format(emb) for emb in range(0, self.kwargs.get('n_components'), 1)]
                                                         ).transpose()
                _feature_importance: pd.DataFrame = abs(_embeddings)
                _cluster[cl].update({'embeddings': _cluster[cl].get('fit').embedding_,
                                     'transformed_embeddings': _cluster[cl].get('fit').transform(X=self.df[self.features]),
                                     'feature_importance': dict(names={c: _feature_importance[c].sort_values(axis=0, ascending=False).index.values[0] for c in _feature_importance.keys()}, scores=_feature_importance)
                                     })
                for tsne in range(0, self.kwargs.get('n_components'), 1):
                    _cluster_plot.update({'Feature Importance TSNE{}'.format(tsne): dict(data=_feature_importance,
                                                                                         features=None,
                                                                                         plot_type='bar',
                                                                                         kwargs=dict(layout={},
                                                                                                     x=self.features,
                                                                                                     y=_feature_importance['tsne{}'.format(pca)],
                                                                                                     marker=dict(color=_feature_importance['tsne{}'.format(pca)],
                                                                                                                 colorscale='rdylgn',
                                                                                                                 autocolorscale=True
                                                                                                                 )
                                                                                                     )
                                                                                         )
                                          })
                _cluster_plot.update({'TSNE Embeddings': dict(data=pd.DataFrame(data=_cluster[cl].get('transformed_embeddings'),
                                                                                columns=list(_feature_importance.keys())
                                                                                ),
                                                              features=list(_feature_importance.keys()),
                                                              plot_type='scatter',
                                                              melt=True,
                                                              kwargs=dict(layout={},
                                                                          marker=dict(color=_cluster[cl].get('transformed_embeddings'),
                                                                                      colorscale='rdylgn',
                                                                                      autocolorscale=True
                                                                                      )
                                                                          )
                                                              )
                                      })
            #############################
            # Multi Dimensional Scaling #
            #############################
            elif cl is 'mds':
                if self.kwargs.get('n_components') is None:
                    self.kwargs.update({'n_components': 2})
                if self.find_optimum:
                    if self.silhouette:
                        _try_run = Clustering(cl_params=self.kwargs).multi_dimensional_scaling().fit(X=self.df[self.features], y=self.target)
                        _silhouette: dict = self._silhoutte_analysis(labels=_try_run.transform(self.df[self.features]))
                        _cluster[cl].update({'silhouette': _silhouette})
                        _cluster_plot.update({'Silhouette Analysis (MDS)': dict(data=None,
                                                                                features=None,
                                                                                plot_type='silhouette',
                                                                                kwargs=dict(layout={},
                                                                                            n_clusters=self.kwargs.get('n_clusters'),
                                                                                            silhouette=_silhouette
                                                                                            )
                                                                                )
                                              })
                _cluster[cl].update({'fit': Clustering(cl_params=self.kwargs).multi_dimensional_scaling().fit(X=self.df[self.features], y=self.target)})
                _embeddings: pd.DataFrame = pd.DataFrame(data=np.array(_cluster[cl].get('fit').embedding_),
                                                         columns=self.features,
                                                         index=['emb{}'.format(emb) for emb in
                                                                range(0, self.kwargs.get('n_components'), 1)]
                                                         ).transpose()
                _feature_importance: pd.DataFrame = abs(_embeddings)
                _cluster[cl].update({'embeddings': _cluster[cl].get('fit').embedding_,
                                     'transformed_embeddings': _cluster[cl].get('fit').transform(X=self.df[self.features]),
                                     'dissimilarity_matrix': _cluster[cl].get('fit').dissimilarity_matrix_,
                                     'feature_importance': dict(names={c: _feature_importance[c].sort_values(axis=0, ascending=False).index.values[0] for c in _feature_importance.keys()}, scores=_feature_importance)
                                     })
                for mds in range(0, self.kwargs.get('n_components'), 1):
                    _cluster_plot.update({'Feature Importance MDS{}'.format(mds): dict(data=_feature_importance,
                                                                                       features=None,
                                                                                       plot_type='bar',
                                                                                       kwargs=dict(layout={},
                                                                                                   x=self.features,
                                                                                                   y=_feature_importance['mds{}'.format(mds)],
                                                                                                   marker=dict(color=_feature_importance['mds{}'.format(mds)],
                                                                                                               colorscale='rdylgn',
                                                                                                               autocolorscale=True
                                                                                                               )
                                                                                                   )
                                                                                       )
                                          })
                _cluster_plot.update({'MDS Embeddings': dict(data=pd.DataFrame(data=_cluster[cl].get('transformed_embeddings'),
                                                                               columns=list(_feature_importance.keys())
                                                                               ),
                                                             features=list(_feature_importance.keys()),
                                                             plot_type='scatter',
                                                             melt=True,
                                                             kwargs=dict(layout={},
                                                                         marker=dict(color=_cluster[cl].get('transformed_embeddings'),
                                                                                     colorscale='rdylgn',
                                                                                     autocolorscale=True
                                                                                     )
                                                                         )
                                                             )
                                      })
            #####################
            # Isometric Mapping #
            #####################
            elif cl is 'isomap':
                if self.kwargs.get('n_components') is None:
                    self.kwargs.update({'n_components': 2})
                if self.find_optimum:
                    if self.silhouette:
                        _try_run = Clustering(cl_params=self.kwargs).isometric_mapping().fit(X=self.df[self.features], y=self.target)
                        _silhouette: dict = self._silhoutte_analysis(labels=_try_run.transform(self.df[self.features]))
                        _cluster[cl].update({'silhouette': _silhouette})
                        _cluster_plot.update({'Silhouette Analysis (Isomap)': dict(data=None,
                                                                                   features=None,
                                                                                   plot_type='silhouette',
                                                                                   kwargs=dict(layout={},
                                                                                               n_clusters=self.kwargs.get('n_clusters'),
                                                                                               silhouette=_silhouette
                                                                                               )
                                                                                   )
                                              })
                _cluster[cl].update({'fit': Clustering(cl_params=self.kwargs).isometric_mapping().fit(X=self.df[self.features], y=self.target)})
                _embeddings: pd.DataFrame = pd.DataFrame(data=np.array(_cluster[cl].get('fit').embedding_),
                                                         columns=self.features,
                                                         index=['emb{}'.format(emb) for emb in
                                                                range(0, self.kwargs.get('n_components'), 1)]
                                                         ).transpose()
                _feature_importance: pd.DataFrame = abs(_embeddings)
                _cluster[cl].update({'embeddings': _cluster[cl].get('fit').embedding_,
                                     'transformed_embeddings': _cluster[cl].get('fit').transform(X=self.df[self.features]),
                                     'distance_matrix': _cluster[cl].get('fit').dist_matrix_,
                                     'kernel_pca': _cluster[cl].get('fit').kernel_pca_,
                                     'reconstruction_error': _cluster[cl].get('fit').reconstruction_error(),
                                     'feature_importance': dict(names={
                                         c: _feature_importance[c].sort_values(axis=0, ascending=False).index.values[0]
                                         for c in _feature_importance.keys()}, scores=_feature_importance)
                                     })
                for iso in range(0, self.kwargs.get('n_components'), 1):
                    _cluster_plot.update({'Feature Importance MDS{}'.format(iso): dict(data=_feature_importance,
                                                                                       features=None,
                                                                                       plot_type='bar',
                                                                                       kwargs=dict(layout={},
                                                                                                   x=self.features,
                                                                                                   y=_feature_importance['iso{}'.format(iso)],
                                                                                                   marker=dict(color=_feature_importance['iso{}'.format(iso)],
                                                                                                               colorscale='rdylgn',
                                                                                                               autocolorscale=True
                                                                                                               )
                                                                                                   )
                                                                                       )
                                          })
                _cluster_plot.update({'Isomap Embeddings': dict(data=pd.DataFrame(data=_cluster[cl].get('transformed_embeddings'),
                                                                                  columns=list(_feature_importance.keys())
                                                                                  ),
                                                                features=list(_feature_importance.keys()),
                                                                plot_type='scatter',
                                                                melt=True,
                                                                kwargs=dict(layout={},
                                                                            marker=dict(color=_cluster[cl].get('transformed_embeddings'),
                                                                                        colorscale='rdylgn',
                                                                                        autocolorscale=True
                                                                                        )
                                                                            )
                                                                )
                                      })
            ######################
            # Spectral Embedding #
            ######################
            elif cl in ['spectral_emb', 'spectral_embedding']:
                if self.kwargs.get('n_components') is None:
                    self.kwargs.update({'n_components': 2})
                if self.find_optimum:
                    if self.silhouette:
                        _try_run = Clustering(cl_params=self.kwargs).spectral_embedding().fit(X=self.df[self.features], y=self.target)
                        _silhouette: dict = self._silhoutte_analysis(labels=_try_run.transform(self.df[self.features]))
                        _cluster[cl].update({'silhouette': _silhouette})
                        _cluster_plot.update({'Silhouette Analysis (SPE)': dict(data=None,
                                                                                features=None,
                                                                                plot_type='silhouette',
                                                                                kwargs=dict(layout={},
                                                                                            n_clusters=self.kwargs.get('n_clusters'),
                                                                                            silhouette=_silhouette
                                                                                            )
                                                                                )
                                              })
                _cluster[cl].update({'fit': Clustering(cl_params=self.kwargs).spectral_embedding().fit(X=self.df[self.features], y=self.target)})
                _embeddings: pd.DataFrame = pd.DataFrame(data=np.array(_cluster[cl].get('fit').embedding_),
                                                         columns=self.features,
                                                         index=['emb{}'.format(emb) for emb in
                                                                range(0, self.kwargs.get('n_components'), 1)]
                                                         ).transpose()
                _feature_importance: pd.DataFrame = abs(_embeddings)
                _cluster[cl].update({'embeddings': _cluster[cl].get('fit').embedding_,
                                     'transformed_embeddings': _cluster[cl].get('fit').transform(X=self.df[self.features]),
                                     'affinity_matrix': _cluster[cl].get('fit').affinity_matrix_,
                                     'feature_importance': dict(names={
                                         c: _feature_importance[c].sort_values(axis=0, ascending=False).index.values[0]
                                         for c in _feature_importance.keys()}, scores=_feature_importance)
                                     })
                for spe in range(0, self.kwargs.get('n_components'), 1):
                    _cluster_plot.update({'Feature Importance SPE{}'.format(spe): dict(data=_feature_importance,
                                                                                       features=None,
                                                                                       plot_type='bar',
                                                                                       kwargs=dict(layout={},
                                                                                                   x=self.features,
                                                                                                   y=_feature_importance['spe{}'.format(spe)],
                                                                                                   marker=dict(color=_feature_importance['spe{}'.format(spe)],
                                                                                                               colorscale='rdylgn',
                                                                                                               autocolorscale=True
                                                                                                               )
                                                                                                   )
                                                                                       )
                                          })
                _cluster_plot.update({'Spectral Embeddings': dict(data=pd.DataFrame(data=_cluster[cl].get('transformed_embeddings'),
                                                                                    columns=list(_feature_importance.keys())
                                                                                    ),
                                                                  features=list(_feature_importance.keys()),
                                                                  plot_type='scatter',
                                                                  melt=True,
                                                                  kwargs=dict(layout={},
                                                                              marker=dict(color=_cluster[cl].get('transformed_embeddings'),
                                                                                          colorscale='rdylgn',
                                                                                          autocolorscale=True
                                                                                          )
                                                                              )
                                                                  )
                                      })
            ############################
            # Locally Linear Embedding #
            ############################
            elif cl in ['lle', 'locally_emb', 'locally_linear', 'locally_embedding']:
                if self.kwargs.get('n_components') is None:
                    self.kwargs.update({'n_components': 2})
                if self.find_optimum:
                    if self.silhouette:
                        _try_run = Clustering(cl_params=self.kwargs).locally_linear_embedding().fit(X=self.df[self.features], y=self.target)
                        _silhouette: dict = self._silhoutte_analysis(labels=_try_run.transform(self.df[self.features]))
                        _cluster[cl].update({'silhouette': _silhouette})
                        _cluster_plot.update({'Silhouette Analysis (LLE)': dict(data=None,
                                                                                features=None,
                                                                                plot_type='silhouette',
                                                                                kwargs=dict(layout={},
                                                                                            n_clusters=self.kwargs.get('n_clusters'),
                                                                                            silhouette=_silhouette
                                                                                            )
                                                                                )
                                              })
                _cluster[cl].update({'fit': Clustering(cl_params=self.kwargs).locally_linear_embedding().fit(X=self.df[self.features], y=self.target)})
                _embeddings: pd.DataFrame = pd.DataFrame(data=np.array(_cluster[cl].get('fit').embedding_),
                                                         columns=self.features,
                                                         index=['emb{}'.format(emb) for emb in
                                                                range(0, self.kwargs.get('n_components'), 1)]
                                                         ).transpose()
                _feature_importance: pd.DataFrame = abs(_embeddings)
                _cluster[cl].update({'embeddings': _cluster[cl].get('fit').embedding_,
                                     'transformed_embeddings': _cluster[cl].get('fit').transform(X=self.df[self.features]),
                                     'reconstruction_error': _cluster[cl].get('fit').reconstruction_error(),
                                     'feature_importance': dict(names={
                                         c: _feature_importance[c].sort_values(axis=0, ascending=False).index.values[0]
                                         for c in _feature_importance.keys()}, scores=_feature_importance)
                                     })
                for lle in range(0, self.kwargs.get('n_components'), 1):
                    _cluster_plot.update({'Feature Importance LLE{}'.format(lle): dict(data=_feature_importance,
                                                                                       features=None,
                                                                                       plot_type='bar',
                                                                                       kwargs=dict(layout={},
                                                                                                   x=self.features,
                                                                                                   y=_feature_importance['lle{}'.format(lle)],
                                                                                                   marker=dict(color=_feature_importance['lle{}'.format(lle)],
                                                                                                               colorscale='rdylgn',
                                                                                                               autocolorscale=True
                                                                                                               )
                                                                                                   )
                                                                                       )
                                          })
                _cluster_plot.update({'Locally Linear Embeddings': dict(data=pd.DataFrame(data=_cluster[cl].get('transformed_embeddings'),
                                                                                          columns=list(_feature_importance.keys())
                                                                                          ),
                                                                        features=list(_feature_importance.keys()),
                                                                        plot_type='scatter',
                                                                        melt=True,
                                                                        kwargs=dict(layout={},
                                                                                    marker=dict(color=_cluster[cl].get('transformed_embeddings'),
                                                                                                colorscale='rdylgn',
                                                                                                autocolorscale=True
                                                                                                )
                                                                                    )
                                                                        )
                                      })
            ###########
            # K-Means #
            ###########
            elif cl is 'kmeans':
                if self.n_cluster_components is None:
                    self.kwargs.update({'n_clusters': 7})
                else:
                    if self.n_cluster_components < 2:
                        Log(write=False, level='info').log(msg='It makes no sense to run cluster analysis with less than 2 clusters ({}). Run analysis with 7 clusters instead'.format(self.kwargs.get('n_clusters')))
                        self.kwargs.update({'n_clusters': 7})
                    else:
                        self.kwargs.update({'n_clusters': self.n_cluster_components})
                if self.find_optimum:
                    if self.silhouette:
                        _try_run = Clustering(cl_params=self.kwargs).kmeans().fit(X=self.df[self.features], y=self.target)
                        _silhouette: dict = self._silhoutte_analysis(labels=_try_run.predict(self.df[self.features]))
                        _cluster[cl].update({'silhouette': _silhouette})
                        _cluster_plot.update({'Silhouette Analysis (KMeans)': dict(data=None,
                                                                                   features=None,
                                                                                   plot_type='silhouette',
                                                                                   kwargs=dict(layout={},
                                                                                               n_clusters=self.kwargs.get('n_clusters'),
                                                                                               silhouette=_silhouette
                                                                                               )
                                                                                   )
                                              })
                    else:
                        _try_run = Clustering(cl_params=self.kwargs).kmeans().fit(X=self.df[self.features], y=self.target)
                    self.kwargs.update({'n_clusters': 7})
                _cluster[cl].update({'fit': Clustering(cl_params=self.kwargs).kmeans().fit(X=self.df[self.features], y=self.target)})
                _cluster[cl].update({'inertia': _cluster[cl].get('fit').inertia_,
                                     'cluster': _cluster[cl].get('fit').predict(X=self.df[self.features]),
                                     'cluster_distance_space': _cluster[cl].get('fit').transform(X=self.df[self.features]),
                                     'centroids': _cluster[cl].get('fit').cluster_centers_,
                                     'labels': _cluster[cl].get('fit').labels_
                                     })
                _cluster_plot.update({'Partitioning Clustering: KMeans': dict(data=self.df,
                                                                              features=self.features,
                                                                              plot_type='scatter',
                                                                              melt=True,
                                                                              kwargs=dict(layout={},
                                                                                          marker=dict(color=_cluster[cl].get('fit').labels_.astype(float))
                                                                                          )
                                                                              )
                                      })
            #####################################
            # Non-Negative Matrix Factorization #
            #####################################
            elif cl is 'nmf':
                _cluster[cl].update({'fit': Clustering(cl_params=self.kwargs).non_negative_matrix_factorization().fit(X=self.df)})
                _cluster[cl].update({'factorization_matrix_w': _cluster[cl].get('fit').transform(X=self.df),
                                     'factorization_matrix_h': _cluster[cl].get('fit').components_,
                                     'reconstruction_error': _cluster[cl].get('fit').reconstruction_err_,
                                     'n_iter': _cluster[cl].get('fit').n_iter_
                                     })
            ###############################
            # Latent Dirichlet Allocation #
            ###############################
            elif cl is 'lda':
                _cluster[cl].update({'fit': Clustering(cl_params=self.kwargs).latent_dirichlet_allocation().fit(X=self.df)})
                _cluster[cl].update({'components': _cluster[cl].get('fit').transform(X=self.df),
                                     'em_iter': _cluster[cl].get('fit').n_batch_iter_,
                                     'passes_iter': _cluster[cl].get('fit').n_iter_,
                                     'perplexity_score': _cluster[cl].get('fit').bound_,
                                     'doc_topic_prior': _cluster[cl].get('fit').doc_topic_prior_,
                                     'topic_word_prior': _cluster[cl].get('fit').topic_word_prior_,
                                     })
            ##################################
            # Latent Single Value Allocation #
            ##################################
            elif cl is 'lsa':
                _cluster[cl].update({'fit': Clustering(cl_params=self.kwargs).truncated_single_value_decomp().fit(X=self.df)})
                _cluster[cl].update({'components': _cluster[cl].get('fit').transform(X=self.df),
                                     'explained_variance': _cluster[cl].get('fit').explained_variance_,
                                     'explained_variance_ration': _cluster[cl].get('fit').explained_variance_ratio_,
                                     'n_iter': _cluster[cl].get('fit').n_iter_
                                     })
            ########################################################
            # Ordering Points To Identify the Clustering Structure #
            ########################################################
            elif cl is 'optics':
                _cluster[cl].update({'fit': Clustering(cl_params=self.kwargs).optics().fit(X=self.df)})
                _cluster[cl].update({'reachability': _cluster[cl].get('fit').reachability_,
                                     'ordering': _cluster[cl].get('fit').ordering_,
                                     'core_distances': _cluster[cl].get('fit').core_distances_,
                                     'predecessor': _cluster[cl].get('fit').predecessor_,
                                     'cluster': _cluster[cl].get('fit').transform(X=self.df[self.features]),
                                     'cluster_hierarchy': _cluster[cl].get('fit').cluster_hierarchy_,
                                     'labels': _cluster[cl].get('fit').labels_
                                     })
                _reachability: pd.DataFrame = pd.DataFrame(data={'reachability': _cluster[cl].get('reachability')[_cluster[cl].get('ordering')],
                                                                 'labels': _cluster[cl].get('labels')[_cluster[cl].get('ordering')]
                                                                 }
                                                           )
                _cluster_plot.update({'OPTICS': dict(data=self.df,
                                                     features=self.features,
                                                     plot_type='scatter',
                                                     melt=True,
                                                     kwargs=dict(layout={},
                                                                 marker=dict(color=_cluster[cl].get('fit').labels_.astype(float))
                                                                 )
                                                     ),
                                      'Reachability': dict(data=_reachability,
                                                           features=['reachability'],
                                                           group_by=['labels'],
                                                           plot_type='scatter',
                                                           melt=True,
                                                           kwargs=dict(layout={},
                                                                       marker=dict(color=_cluster[cl].get('fit').labels_.astype(float))
                                                                       )
                                                           )
                                      })
            ###############################################################
            # Density-Based Spatial Clustering of Applications with Noise #
            ###############################################################
            elif cl is 'dbscan':
                _cluster[cl].update({'fit': Clustering(cl_params=self.kwargs).dbscan().fit(X=self.df)})
                _cluster[cl].update({'core_samples': _cluster[cl].get('fit').core_samples_,
                                     'cluster': _cluster[cl].get('fit').transform(X=self.df[self.features]),
                                     'lables': _cluster[cl].get('fit').labels_
                                     })
                _cluster_plot.update({'DBSCAN': dict(data=self.df,
                                                     features=self.features,
                                                     plot_type='scatter',
                                                     melt=True,
                                                     kwargs=dict(layout={},
                                                                 marker=dict(color=_cluster[cl].get('fit').labels_.astype(float))
                                                                 )
                                                     )
                                      })
            #######################
            # Spectral Clustering #
            #######################
            elif cl in ['spectral_cl', 'spectral_cluster']:
                _cluster[cl].update({'fit': Clustering(cl_params=self.kwargs).spectral_clustering().fit(X=self.df)})
                _cluster[cl].update({'affinity_matrix': _cluster[cl].get('fit').affinity_matrix_,
                                     'cluster': _cluster[cl].get('fit').transform(X=self.df[self.features]),
                                     'lables': _cluster[cl].get('fit').labels_
                                     })
                _cluster_plot.update({'Spectral Clustering': dict(data=self.df,
                                                                  features=self.features,
                                                                  plot_type='scatter',
                                                                  melt=True,
                                                                  kwargs=dict(layout={},
                                                                              marker=dict(color=_cluster[cl].get('fit').labels_.astype(float))
                                                                              )
                                                                  )
                                      })
            #########################
            # Feature Agglomeration #
            #########################
            elif cl in ['feature_agglo', 'feature_agglomeration']:
                _cluster[cl].update({'fit': Clustering(cl_params=self.kwargs).feature_agglomeration().fit(X=self.df[self.features], y=self.target)})
                _cluster[cl].update({'clusters': _cluster[cl].clusters_,
                                     'within_cluster_error': _cluster[cl].get('fit').explained_variance_,
                                     'cluster': _cluster[cl].get('fit').transform(X=self.df[self.features])
                                     })
                _cluster_plot.update({'Feature Agglomeration': dict(data=self.df,
                                                                    features=self.features,
                                                                    plot_type='scatter',
                                                                    melt=True,
                                                                    kwargs=dict(layout={},
                                                                                marker=dict(color=_cluster[cl].get('fit').labels_.astype(float))
                                                                                )
                                                                    )
                                      })
            ############################
            # Agglomerative Clustering #
            ############################
            elif cl in ['agglo_cl', 'agglo_cluster', 'struc_agglo_cl', 'struc_agglo_cluster', 'unstruc_agglo_cl', 'unstruc_agglo_cluster']:
                if cl.find('struc') >= 0:
                    if self.kwargs.get('connectivity') is None:
                        self.kwargs.update({'connectivity': kneighbors_graph(X=self.df[self.features],
                                                                             n_neighbors=self.n_neighbors,
                                                                             mode='connectivity' if self.kwargs.get('connectivity') is None else self.kwargs.get('connectivity'),
                                                                             metric='minkowski' if self.kwargs.get('metric') is None else self.kwargs.get('metric'),
                                                                             p=2 if self.kwargs.get('p') is None else self.kwargs.get('p'),
                                                                             metric_params=self.kwargs.get('metric_params'),
                                                                             include_self=False if self.kwargs.get('include_self') is None else self.kwargs.get('include_self'),
                                                                             n_jobs=self.cpu_cores
                                                                             )})
                _cluster[cl].update({'fit': Clustering(cl_params=self.kwargs).agglomerative_clustering().fit(X=self.df[self.features], y=self.target)})
                _cluster[cl].update({'clusters': _cluster[cl].get('fit').clusters_,
                                     'cluster': _cluster[cl].get('fit').transform(X=self.df[self.features]),
                                     'components': _cluster[cl].get('fit').n_components_,
                                     'labels': _cluster[cl].get('fit').labels_
                                     })
                _cluster_plot.update({'Agglomerative Clustering': dict(data=self.df,
                                                                       features=self.features,
                                                                       plot_type='scatter',
                                                                       melt=True,
                                                                       kwargs=dict(layout={},
                                                                                   marker=dict(color=_cluster[cl].get('fit').labels_.astype(float))
                                                                                   )
                                                                       )
                                      })
            ####################
            # Birch Clustering #
            ####################
            elif cl is 'birch':
                _cluster[cl].update({'fit': Clustering(cl_params=self.kwargs).birch().fit(X=self.df[self.features])})
                _cluster[cl].update({'partial_fit': _cluster[cl].get('fit').partial_fit_,
                                     'root': _cluster[cl].get('fit').root_,
                                     'centroids': _cluster[cl].get('fit').subcluster_centers_,
                                     'cluster': _cluster[cl].get('fit').transform(X=self.df[self.features]),
                                     'cluster_labels': _cluster[cl].get('fit').subcluster_labels_,
                                     'dummy_leaf': _cluster[cl].get('fit').dummy_leaf_,
                                     'labels': _cluster[cl].get('fit').labels_
                                     })
                _cluster_plot.update({'Birch': dict(data=self.df,
                                                    features=self.features,
                                                    plot_type='scatter',
                                                    melt=True,
                                                    kwargs=dict(layout={},
                                                                marker=dict(color=_cluster[cl].get('fit').labels_.astype(float))
                                                                )
                                                    )
                                      })
            ########################
            # Affinity Propagation #
            ########################
            elif cl in ['affinity_prop', 'affinity_propagation']:
                _cluster[cl].update({'fit': Clustering(cl_params=self.kwargs).affinity_propagation().fit(X=self.df[self.features], y=self.target)})
                _cluster[cl].update({'cluster_centers': _cluster[cl].get('fit').cluster_centers_,
                                     'affinity_matrix': _cluster[cl].get('fit').affinity_matrix_,
                                     'labels': _cluster[cl].get('fit').labels_,
                                     'predict': _cluster[cl].get('fit').predict(X=self.df[self.features])
                                     })
                _cluster_plot.update({'Affinity Propagation': dict(data=self.df,
                                                                   features=self.features,
                                                                   plot_type='scatter',
                                                                   melt=True,
                                                                   kwargs=dict(layout={},
                                                                               marker=dict(color=_cluster[cl].get('fit').labels_.astype(float))
                                                                               )
                                                                   )
                                      })
            else:
                raise UnsupervisedMLException('Clustering method ({}) not supported'.format(cl))
            if self.plot:
                DataVisualizer(subplots=_cluster_plot,
                               interactive=True,
                               height=500,
                               width=500,
                               unit='px'
                               ).run()
        return _cluster
