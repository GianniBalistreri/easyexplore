import pandas as pd
import unittest

from easyexplore.unsupervised_machine_learning import (
    AffinityPropagation, AgglomerativeClustering, Birch, Clustering, DBSCAN, FactorAnalysis, FeatureAgglomeration,
    FastICA, Isomap, KMeans, LatentDirichletAllocation, LocallyLinearEmbedding, MDS, NMF, OPTICS, PCA,
    SpectralClustering, SpectralEmbedding, TSNE, TruncatedSVD, UnsupervisedML
)
from typing import List

DATA_SET: pd.DataFrame = pd.read_csv(filepath_or_buffer='wine.csv', sep=',')
FEATURES: List[str] = ['alcohol',
                       'malic_acid',
                       'ash',
                       'alcalinity_of_ash',
                       'magnesium',
                       'total_phenols',
                       'flavanoids',
                       'non_flavanoid_phenols',
                       'proanthocyanins',
                       'color_intensity',
                       'hue',
                       'OD280/OD315_of_diluted_wines',
                       'proline'
                       ]


class ClusterTest(unittest.TestCase):
    """
    Unit test for class Clustering
    """
    def test_affinity_propagation(self):
        _clustering: Clustering = Clustering(cl_params=None)
        self.assertTrue(expr=isinstance(_clustering.affinity_propagation(), AffinityPropagation))

    def test_agglomerative_clustering(self):
        _clustering: Clustering = Clustering(cl_params=None)
        self.assertTrue(expr=isinstance(_clustering.agglomerative_clustering(), AgglomerativeClustering))

    def test_birch(self):
        _clustering: Clustering = Clustering(cl_params=None)
        self.assertTrue(expr=isinstance(_clustering.birch(), Birch))

    def test_dbscan(self):
        _clustering: Clustering = Clustering(cl_params=None)
        self.assertTrue(expr=isinstance(_clustering.dbscan(), DBSCAN))

    def test_factor_analysis(self):
        _clustering: Clustering = Clustering(cl_params=None)
        self.assertTrue(expr=isinstance(_clustering.factor_analysis(), FactorAnalysis))

    def test_feature_agglomeration(self):
        _clustering: Clustering = Clustering(cl_params=None)
        self.assertTrue(expr=isinstance(_clustering.feature_agglomeration(), FeatureAgglomeration))

    def test_independent_component_analysis(self):
        _clustering: Clustering = Clustering(cl_params=None)
        self.assertTrue(expr=isinstance(_clustering.independent_component_analysis(), FastICA))

    def test_isometric_mapping(self):
        _clustering: Clustering = Clustering(cl_params=None)
        self.assertTrue(expr=isinstance(_clustering.isometric_mapping(), Isomap))

    def test_kmeans(self):
        _clustering: Clustering = Clustering(cl_params=None)
        self.assertTrue(expr=isinstance(_clustering.kmeans(), KMeans))

    def test_latent_dirichlet_allocation(self):
        _clustering: Clustering = Clustering(cl_params=None)
        self.assertTrue(expr=isinstance(_clustering.latent_dirichlet_allocation(), LatentDirichletAllocation))

    def test_locally_linear_embedding(self):
        _clustering: Clustering = Clustering(cl_params=None)
        self.assertTrue(expr=isinstance(_clustering.locally_linear_embedding(), LocallyLinearEmbedding))

    def test_multi_dimensional_scaling(self):
        _clustering: Clustering = Clustering(cl_params=None)
        self.assertTrue(expr=isinstance(_clustering.multi_dimensional_scaling(), MDS))

    def test_non_negative_matrix_factorization(self):
        _clustering: Clustering = Clustering(cl_params=None)
        self.assertTrue(expr=isinstance(_clustering.non_negative_matrix_factorization(), NMF))

    def test_optics(self):
        _clustering: Clustering = Clustering(cl_params=None)
        self.assertTrue(expr=isinstance(_clustering.optics(), OPTICS))

    def test_principal_component_analysis(self):
        _clustering: Clustering = Clustering(cl_params=None)
        self.assertTrue(expr=isinstance(_clustering.principal_component_analysis(), PCA))

    def test_spectral_clustering(self):
        _clustering: Clustering = Clustering(cl_params=None)
        self.assertTrue(expr=isinstance(_clustering.spectral_clustering(), SpectralClustering))

    def test_spectral_embedding(self):
        _clustering: Clustering = Clustering(cl_params=None)
        self.assertTrue(expr=isinstance(_clustering.spectral_embedding(), SpectralEmbedding))

    def test_t_distributed_stochastic_neighbor_embedding(self):
        _clustering: Clustering = Clustering(cl_params=None)
        self.assertTrue(expr=isinstance(_clustering.t_distributed_stochastic_neighbor_embedding(), TSNE))

    def test_truncated_single_value_decomp(self):
        _clustering: Clustering = Clustering(cl_params=None)
        self.assertTrue(expr=isinstance(_clustering.truncated_single_value_decomp(), TruncatedSVD))


class UnsupervisedMLTest(unittest.TestCase):
    """
    Unit test for class UnsupervisedML
    """
    def test_run_clustering_pca(self):
        _clustering: UnsupervisedML = UnsupervisedML(df=DATA_SET,
                                                     features=FEATURES,
                                                     find_optimum=True,
                                                     silhouette_analysis=True,
                                                     n_cluster_components=3,
                                                     n_neighbors=None,
                                                     n_iter=None,
                                                     metric=None,
                                                     affinity=None,
                                                     connectivity=None,
                                                     linkage=None,
                                                     target=None,
                                                     plot=False,
                                                     log_path=None
                                                     )
        _clustering.run_clustering(cluster_algorithms=['pca'])
        _meth: str = list(_clustering.cluster.keys())[0]
        _pca_keys: List[str] = ['fit',
                                'n_components',
                                'explained_variance_ratio',
                                'cumulative_explained_variance_ratio',
                                'components',
                                'explained_variance',
                                'pc',
                                'feature_importance'
                                ]
        self.assertTrue(expr=_meth == 'pca' and list(_clustering.cluster[_meth].keys()) == _pca_keys)

    def test_run_clustering_factor(self):
        _clustering: UnsupervisedML = UnsupervisedML(df=DATA_SET,
                                                     features=FEATURES,
                                                     find_optimum=True,
                                                     silhouette_analysis=True,
                                                     n_cluster_components=3,
                                                     n_neighbors=None,
                                                     n_iter=None,
                                                     metric=None,
                                                     affinity=None,
                                                     connectivity=None,
                                                     linkage=None,
                                                     target=None,
                                                     plot=False,
                                                     log_path=None
                                                     )
        _clustering.run_clustering(cluster_algorithms=['factor'])
        _meth: str = list(_clustering.cluster.keys())[0]
        _factor_keys: List[str] = []
        self.assertTrue(expr=_meth == 'factor' and list(_clustering.cluster[_meth].keys()) == _factor_keys)

    def test_run_clustering_svd(self):
        _clustering: UnsupervisedML = UnsupervisedML(df=DATA_SET,
                                                     features=FEATURES,
                                                     find_optimum=True,
                                                     silhouette_analysis=True,
                                                     n_cluster_components=3,
                                                     n_neighbors=None,
                                                     n_iter=None,
                                                     metric=None,
                                                     affinity=None,
                                                     connectivity=None,
                                                     linkage=None,
                                                     target=None,
                                                     plot=False,
                                                     log_path=None
                                                     )
        _clustering.run_clustering(cluster_algorithms=['svd'])
        _meth: str = list(_clustering.cluster.keys())[0]
        _svd_keys: List[str] = ['fit',
                                'n_components',
                                'explained_variance_ratio',
                                'cumulative_explained_variance_ratio',
                                'components',
                                'explained_variance',
                                'pc',
                                'feature_importance'
                                ]
        self.assertTrue(expr=_meth == 'svd' and list(_clustering.cluster[_meth].keys()) == _svd_keys)

    def test_run_clustering_tsne(self):
        _clustering: UnsupervisedML = UnsupervisedML(df=DATA_SET,
                                                     features=FEATURES,
                                                     find_optimum=True,
                                                     silhouette_analysis=True,
                                                     n_cluster_components=3,
                                                     n_neighbors=None,
                                                     n_iter=None,
                                                     metric=None,
                                                     affinity=None,
                                                     connectivity=None,
                                                     linkage=None,
                                                     target=None,
                                                     plot=False,
                                                     log_path=None
                                                     )
        _clustering.run_clustering(cluster_algorithms=['tsne'])
        _meth: str = list(_clustering.cluster.keys())[0]
        _tsne_keys: List[str] = ['fit',
                                 'n_components',
                                 'embeddings'
                                 ]
        self.assertTrue(expr=_meth == 'tsne' and list(_clustering.cluster[_meth].keys()) == _tsne_keys)

    def test_run_clustering_mds(self):
        _clustering: UnsupervisedML = UnsupervisedML(df=DATA_SET,
                                                     features=FEATURES,
                                                     find_optimum=True,
                                                     silhouette_analysis=True,
                                                     n_cluster_components=3,
                                                     n_neighbors=None,
                                                     n_iter=None,
                                                     metric=None,
                                                     affinity=None,
                                                     connectivity=None,
                                                     linkage=None,
                                                     target=None,
                                                     plot=False,
                                                     log_path=None
                                                     )
        _clustering.run_clustering(cluster_algorithms=['mds'])
        _meth: str = list(_clustering.cluster.keys())[0]
        _mds_keys: List[str] = ['fit',
                                'n_components',
                                'embeddings',
                                'dissimilarity_matrix',
                                'stress',
                                'n_iter'
                                ]
        self.assertTrue(expr=_meth == 'mds' and list(_clustering.cluster[_meth].keys()) == _mds_keys)

    def test_run_clustering_isomap(self):
        _clustering: UnsupervisedML = UnsupervisedML(df=DATA_SET,
                                                     features=FEATURES,
                                                     find_optimum=True,
                                                     silhouette_analysis=True,
                                                     n_cluster_components=3,
                                                     n_neighbors=None,
                                                     n_iter=None,
                                                     metric=None,
                                                     affinity=None,
                                                     connectivity=None,
                                                     linkage=None,
                                                     target=None,
                                                     plot=False,
                                                     log_path=None
                                                     )
        _clustering.run_clustering(cluster_algorithms=['isomap'])
        _meth: str = list(_clustering.cluster.keys())[0]
        _isomap_keys: List[str] = ['fit',
                                   'n_components',
                                   'embeddings',
                                   'transformed_embeddings',
                                   'distance_matrix',
                                   'kernel_pca',
                                   'reconstruction_error'
                                   ]
        self.assertTrue(expr=_meth == 'isomap' and list(_clustering.cluster[_meth].keys()) == _isomap_keys)

    def test_run_clustering_spectral_embedding(self):
        _clustering: UnsupervisedML = UnsupervisedML(df=DATA_SET,
                                                     features=FEATURES,
                                                     find_optimum=True,
                                                     silhouette_analysis=True,
                                                     n_cluster_components=3,
                                                     n_neighbors=None,
                                                     n_iter=None,
                                                     metric=None,
                                                     affinity=None,
                                                     connectivity=None,
                                                     linkage=None,
                                                     target=None,
                                                     plot=False,
                                                     log_path=None
                                                     )
        _clustering.run_clustering(cluster_algorithms=['spectral_embedding'])
        _meth: str = list(_clustering.cluster.keys())[0]
        _spectral_embedding_keys: List[str] = ['fit',
                                               'n_components',
                                               'embeddings',
                                               'affinity_matrix',
                                               ]
        self.assertTrue(expr=_meth == 'spectral_embedding' and list(_clustering.cluster[_meth].keys()) == _spectral_embedding_keys)

    def test_run_clustering_locally_embedding(self):
        _clustering: UnsupervisedML = UnsupervisedML(df=DATA_SET,
                                                     features=FEATURES,
                                                     find_optimum=True,
                                                     silhouette_analysis=True,
                                                     n_cluster_components=3,
                                                     n_neighbors=None,
                                                     n_iter=None,
                                                     metric=None,
                                                     affinity=None,
                                                     connectivity=None,
                                                     linkage=None,
                                                     target=None,
                                                     plot=False,
                                                     log_path=None
                                                     )
        _clustering.run_clustering(cluster_algorithms=['locally_embedding'])
        _meth: str = list(_clustering.cluster.keys())[0]
        _locally_embedding_keys: List[str] = ['fit',
                                              'n_components',
                                              'embeddings',
                                              'transformed_embeddings',
                                              'reconstruction_error'
                                              ]
        self.assertTrue(expr=_meth == 'locally_embedding' and list(_clustering.cluster[_meth].keys()) == _locally_embedding_keys)

    def test_run_clustering_kmeans(self):
        _clustering: UnsupervisedML = UnsupervisedML(df=DATA_SET,
                                                     features=FEATURES,
                                                     find_optimum=True,
                                                     silhouette_analysis=True,
                                                     n_cluster_components=3,
                                                     n_neighbors=None,
                                                     n_iter=None,
                                                     metric=None,
                                                     affinity=None,
                                                     connectivity=None,
                                                     linkage=None,
                                                     target=None,
                                                     plot=False,
                                                     log_path=None
                                                     )
        _clustering.run_clustering(cluster_algorithms=['kmeans'])
        _meth: str = list(_clustering.cluster.keys())[0]
        _kmeans_keys: List[str] = ['fit',
                                   'n_clusters',
                                   'silhouette',
                                   'inertia',
                                   'cluster',
                                   'cluster_distance_space',
                                   'centroids',
                                   'labels'
                                   ]
        self.assertTrue(expr=_meth == 'kmeans' and list(_clustering.cluster[_meth].keys()) == _kmeans_keys)

    def test_run_clustering_nmf(self):
        _clustering: UnsupervisedML = UnsupervisedML(df=DATA_SET,
                                                     features=FEATURES,
                                                     find_optimum=True,
                                                     silhouette_analysis=True,
                                                     n_cluster_components=3,
                                                     n_neighbors=None,
                                                     n_iter=None,
                                                     metric=None,
                                                     affinity=None,
                                                     connectivity=None,
                                                     linkage=None,
                                                     target=None,
                                                     plot=False,
                                                     log_path=None
                                                     )
        _clustering.run_clustering(cluster_algorithms=['nmf'])
        _meth: str = list(_clustering.cluster.keys())[0]
        _nmf_keys: List[str] = ['fit',
                                'factorization_matrix_w',
                                'factorization_matrix_h',
                                'reconstruction_error',
                                'n_iter'
                                ]
        self.assertTrue(expr=_meth == 'nmf' and list(_clustering.cluster[_meth].keys()) == _nmf_keys)

    def test_run_clustering_lda(self):
        _clustering: UnsupervisedML = UnsupervisedML(df=DATA_SET,
                                                     features=FEATURES,
                                                     find_optimum=True,
                                                     silhouette_analysis=True,
                                                     n_cluster_components=3,
                                                     n_neighbors=None,
                                                     n_iter=None,
                                                     metric=None,
                                                     affinity=None,
                                                     connectivity=None,
                                                     linkage=None,
                                                     target=None,
                                                     plot=False,
                                                     log_path=None
                                                     )
        _clustering.run_clustering(cluster_algorithms=['lda'])
        _meth: str = list(_clustering.cluster.keys())[0]
        _lda_keys: List[str] = ['fit',
                                'components',
                                'em_iter',
                                'passes_iter',
                                'perplexity_score',
                                'doc_topic_prior',
                                'topic_word_prior'
                                ]
        self.assertTrue(expr=_meth == 'lda' and list(_clustering.cluster[_meth].keys()) == _lda_keys)

    def test_run_clustering_optics(self):
        _clustering: UnsupervisedML = UnsupervisedML(df=DATA_SET,
                                                     features=FEATURES,
                                                     find_optimum=True,
                                                     silhouette_analysis=True,
                                                     n_cluster_components=3,
                                                     n_neighbors=None,
                                                     n_iter=None,
                                                     metric=None,
                                                     affinity=None,
                                                     connectivity=None,
                                                     linkage=None,
                                                     target=None,
                                                     plot=False,
                                                     log_path=None
                                                     )
        _clustering.run_clustering(cluster_algorithms=['optics'])
        _meth: str = list(_clustering.cluster.keys())[0]
        _optics_keys: List[str] = ['fit',
                                   'n_clusters',
                                   'silhouette',
                                   'reachability',
                                   'ordering',
                                   'core_distances',
                                   'predecessor',
                                   'cluster_hierarchy',
                                   'labels'
                                   ]
        self.assertTrue(expr=_meth == 'optics' and list(_clustering.cluster[_meth].keys()) == _optics_keys)

    def test_run_clustering_dbscan(self):
        _clustering: UnsupervisedML = UnsupervisedML(df=DATA_SET,
                                                     features=FEATURES,
                                                     find_optimum=True,
                                                     silhouette_analysis=True,
                                                     n_cluster_components=3,
                                                     n_neighbors=None,
                                                     n_iter=None,
                                                     metric=None,
                                                     affinity=None,
                                                     connectivity=None,
                                                     linkage=None,
                                                     target=None,
                                                     plot=False,
                                                     log_path=None
                                                     )
        _clustering.run_clustering(cluster_algorithms=['dbscan'])
        _meth: str = list(_clustering.cluster.keys())[0]
        _dbscan_keys: List[str] = ['fit',
                                   'n_clusters',
                                   'core_sample_indices',
                                   'labels'
                                   ]
        self.assertTrue(expr=_meth == 'dbscan' and list(_clustering.cluster[_meth].keys()) == _dbscan_keys)

    def test_run_clustering_spectral_cluster(self):
        _clustering: UnsupervisedML = UnsupervisedML(df=DATA_SET,
                                                     features=FEATURES,
                                                     find_optimum=True,
                                                     silhouette_analysis=True,
                                                     n_cluster_components=3,
                                                     n_neighbors=None,
                                                     n_iter=None,
                                                     metric=None,
                                                     affinity=None,
                                                     connectivity=None,
                                                     linkage=None,
                                                     target=None,
                                                     plot=False,
                                                     log_path=None
                                                     )
        _clustering.run_clustering(cluster_algorithms=['spectral_cluster'])
        _meth: str = list(_clustering.cluster.keys())[0]
        _spectral_cluster_keys: List[str] = ['fit',
                                             'n_clusters',
                                             'silhouette',
                                             'affinity_matrix',
                                             'labels'
                                             ]
        self.assertTrue(expr=_meth == 'spectral_cluster' and list(_clustering.cluster[_meth].keys()) == _spectral_cluster_keys)

    def test_run_clustering_feature_agglomeration(self):
        _clustering: UnsupervisedML = UnsupervisedML(df=DATA_SET,
                                                     features=FEATURES,
                                                     find_optimum=True,
                                                     silhouette_analysis=True,
                                                     n_cluster_components=3,
                                                     n_neighbors=None,
                                                     n_iter=None,
                                                     metric=None,
                                                     affinity=None,
                                                     connectivity=None,
                                                     linkage=None,
                                                     target=None,
                                                     plot=False,
                                                     log_path=None
                                                     )
        _clustering.run_clustering(cluster_algorithms=['feature_agglomeration'])
        _meth: str = list(_clustering.cluster.keys())[0]
        _feature_agglomeration_keys: List[str] = ['fit',
                                                  'n_clusters',
                                                  'n_leaves',
                                                  'n_components',
                                                  'children',
                                                  'reduced_data_set',
                                                  'labels'
                                                  ]
        self.assertTrue(expr=_meth == 'feature_agglomeration' and list(_clustering.cluster[_meth].keys()) == _feature_agglomeration_keys)

    def test_run_clustering_agglo_cluster(self):
        _clustering: UnsupervisedML = UnsupervisedML(df=DATA_SET,
                                                     features=FEATURES,
                                                     find_optimum=True,
                                                     silhouette_analysis=True,
                                                     n_cluster_components=3,
                                                     n_neighbors=3,
                                                     n_iter=None,
                                                     metric=None,
                                                     affinity=None,
                                                     connectivity=None,
                                                     linkage=None,
                                                     target=None,
                                                     plot=False,
                                                     log_path=None
                                                     )
        _clustering.run_clustering(cluster_algorithms=['agglo_cluster'])
        _meth: str = list(_clustering.cluster.keys())[0]
        _agglo_cluster_keys: List[str] = ['fit',
                                          'silhouette',
                                          'connectivity',
                                          'n_clusters',
                                          'n_leaves',
                                          'n_components',
                                          'children',
                                          'labels'
                                          ]
        self.assertTrue(expr=_meth == 'agglo_cluster' and list(_clustering.cluster[_meth].keys()) == _agglo_cluster_keys)

    def test_run_clustering_unstruc_agglo_cluster(self):
        _clustering: UnsupervisedML = UnsupervisedML(df=DATA_SET,
                                                     features=FEATURES,
                                                     find_optimum=True,
                                                     silhouette_analysis=True,
                                                     n_cluster_components=3,
                                                     n_neighbors=None,
                                                     n_iter=None,
                                                     metric=None,
                                                     affinity=None,
                                                     connectivity=None,
                                                     linkage=None,
                                                     target=None,
                                                     plot=False,
                                                     log_path=None
                                                     )
        _clustering.run_clustering(cluster_algorithms=['unstruc_agglo_cluster'])
        _meth: str = list(_clustering.cluster.keys())[0]
        _unstruc_agglo_cluster_keys: List[str] = ['fit',
                                                  'silhouette',
                                                  'connectivity',
                                                  'n_clusters',
                                                  'n_leaves',
                                                  'n_components',
                                                  'children',
                                                  'labels'
                                                  ]
        self.assertTrue(expr=_meth == 'unstruc_agglo_cluster' and list(_clustering.cluster[_meth].keys()) == _unstruc_agglo_cluster_keys)

    def test_run_clustering_birch(self):
        _clustering: UnsupervisedML = UnsupervisedML(df=DATA_SET,
                                                     features=FEATURES,
                                                     find_optimum=True,
                                                     silhouette_analysis=True,
                                                     n_cluster_components=3,
                                                     n_neighbors=None,
                                                     n_iter=None,
                                                     metric=None,
                                                     affinity=None,
                                                     connectivity=None,
                                                     linkage=None,
                                                     target=None,
                                                     plot=False,
                                                     log_path=None
                                                     )
        _clustering.run_clustering(cluster_algorithms=['birch'])
        _meth: str = list(_clustering.cluster.keys())[0]
        _birch_keys: List[str] = ['fit',
                                  'n_clusters',
                                  'silhouette',
                                  'partial_fit',
                                  'root',
                                  'centroids',
                                  'cluster',
                                  'cluster_labels',
                                  'dummy_leaf',
                                  'labels'
                                  ]
        self.assertTrue(expr=_meth == 'birch' and list(_clustering.cluster[_meth].keys()) == _birch_keys)

    def test_run_clustering_affinity_propagation(self):
        _clustering: UnsupervisedML = UnsupervisedML(df=DATA_SET,
                                                     features=FEATURES,
                                                     find_optimum=True,
                                                     silhouette_analysis=True,
                                                     n_cluster_components=3,
                                                     n_neighbors=None,
                                                     n_iter=None,
                                                     metric=None,
                                                     affinity=None,
                                                     connectivity=None,
                                                     linkage=None,
                                                     target=None,
                                                     plot=False,
                                                     log_path=None
                                                     )
        _clustering.run_clustering(cluster_algorithms=['affinity_propagation'])
        _meth: str = list(_clustering.cluster.keys())[0]
        _affinity_propagation_keys: List[str] = ['fit',
                                                 'n_clusters',
                                                 'silhouette',
                                                 'cluster_centers',
                                                 'affinity_matrix',
                                                 'labels',
                                                 'cluster'
                                                 ]
        self.assertTrue(expr=_meth == 'affinity_propagation' and list(_clustering.cluster[_meth].keys()) == _affinity_propagation_keys)

    def test_silhouette_analysis(self):
        _n_clusters: int = 3
        _clustering: UnsupervisedML = UnsupervisedML(df=DATA_SET,
                                                     features=FEATURES,
                                                     find_optimum=False,
                                                     silhouette_analysis=True,
                                                     n_cluster_components=_n_clusters,
                                                     n_neighbors=None,
                                                     n_iter=None,
                                                     metric=None,
                                                     affinity=None,
                                                     connectivity=None,
                                                     linkage=None,
                                                     target=None,
                                                     plot=False,
                                                     log_path=None
                                                     )
        _silhouette_results: dict = _clustering.silhouette_analysis(labels=DATA_SET['class'].values.tolist())
        _silhouette_results_keys: List[str] = []
        for c in range(0, _n_clusters, 1):
            _silhouette_results_keys.append(f'cluster_{c}_avg')
            _silhouette_results_keys.append(f'cluster_{c}_samples')
        _silhouette_results_keys.append('best')
        self.assertTrue(expr=_silhouette_results_keys == list(_silhouette_results.keys()))


if __name__ == '__main__':
    unittest.main()
