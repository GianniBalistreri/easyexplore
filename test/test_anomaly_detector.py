import pandas as pd
import unittest

from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.cof import COF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.loci import LOCI
from pyod.models.lof import LOF
from pyod.models.lmdd import LMDD
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.sod import SOD
from pyod.models.sos import SOS
from easyexplore.anomaly_detector import AnomalyDetector

DATA_SET: pd.DataFrame = pd.read_csv(filepath_or_buffer='test_data.csv', sep=',')


class AnomalyDetectorTest(unittest.TestCase):
    """
    Unit test for class AnomalyDetector
    """
    def test_angle_based_outlier_detection(self):
        self.assertTrue(expr=isinstance(AnomalyDetector(df=DATA_SET).angle_based_outlier_detection(), ABOD))

    def test_cluster_based_local_outlier_factor(self):
        self.assertTrue(expr=isinstance(AnomalyDetector(df=DATA_SET).cluster_based_local_outlier_factor(), CBLOF))

    def test_connectivity_based_outlier_fraction(self):
        self.assertTrue(expr=isinstance(AnomalyDetector(df=DATA_SET).connectivity_based_outlier_fraction(), COF))

    def test_deviation_based_outlier_detection(self):
        self.assertTrue(expr=isinstance(AnomalyDetector(df=DATA_SET).deviation_based_outlier_detection(), LMDD))

    def test_feature_bagging(self):
        self.assertTrue(expr=isinstance(AnomalyDetector(df=DATA_SET).feature_bagging(), FeatureBagging))

    def test_histogram_based_outlier_detection(self):
        self.assertTrue(expr=isinstance(AnomalyDetector(df=DATA_SET).histogram_based_outlier_detection(), HBOS))

    def test_isolation_forest(self):
        self.assertTrue(expr=isinstance(AnomalyDetector(df=DATA_SET).isolation_forest(), IForest))

    def test_k_nearest_neighbor(self):
        self.assertTrue(expr=isinstance(AnomalyDetector(df=DATA_SET).k_nearest_neighbor(), KNN))

    def test_local_correlated_integral(self):
        self.assertTrue(expr=isinstance(AnomalyDetector(df=DATA_SET).local_correlated_integral(), LOCI))

    def test_local_outlier_fraction(self):
        self.assertTrue(expr=isinstance(AnomalyDetector(df=DATA_SET).local_outlier_fraction(), LOF))

    def test_locally_selective_combination_of_parallel_outlier_ensambles(self):
        pass
        #self.assertTrue(expr=isinstance(AnomalyDetector(df=DATA_SET).locally_selective_combination_of_parallel_outlier_ensambles(), LSCP))

    def test_minimum_covariance_determinant(self):
        self.assertTrue(expr=isinstance(AnomalyDetector(df=DATA_SET).minimum_covariance_determinant(), MCD))

    def test_one_class_support_vector_machine(self):
        self.assertTrue(expr=isinstance(AnomalyDetector(df=DATA_SET).one_class_support_vector_machine(), OCSVM))

    def test_principal_component_analysis(self):
        self.assertTrue(expr=isinstance(AnomalyDetector(df=DATA_SET).principal_component_analysis(), PCA))

    def test_subspace_outlier_detection(self):
        self.assertTrue(expr=isinstance(AnomalyDetector(df=DATA_SET).subspace_outlier_detection(), SOD))

    def test_stochastic_outlier_selection(self):
        self.assertTrue(expr=isinstance(AnomalyDetector(df=DATA_SET).stochastic_outlier_selection(), SOS))

    def test_univariate(self):
        pass

    def test_multivariate(self):
        pass


if __name__ == '__main__':
    unittest.main()
