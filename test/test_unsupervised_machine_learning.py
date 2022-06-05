import pandas as pd
import unittest

from easyexplore.unsupervised_machine_learning import (
    AffinityPropagation, AgglomerativeClustering, Birch, Clustering, DBSCAN, FactorAnalysis, FeatureAgglomeration,
    FastICA, Isomap, KMeans, LatentDirichletAllocation, LocallyLinearEmbedding, MDS, NMF, OPTICS, PCA,
    SpectralClustering, SpectralEmbedding, TSNE, TruncatedSVD, UnsupervisedML
)
from typing import List

DATA_SET: pd.DataFrame = pd.read_csv(filepath_or_buffer='wine.csv', sep=',')


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


if __name__ == '__main__':
    unittest.main()
