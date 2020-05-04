import numpy as np
import pandas as pd

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
from pyod.models.lscp import LSCP
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.sod import SOD
from pyod.models.sos import SOS
from typing import Dict, List


ANOMALY_METH: dict = dict(abod='angle_based_outlier_detection',
                          cblof='cluster_based_local_outlier_factor',
                          cof='connectivity_based_outlier_fraction',
                          ftbag='feature_bagging',
                          hbos='histogram_based_outlier_detection',
                          iforest='isolation_forest',
                          knn='k_nearest_neighbor',
                          loci='local_correlated_integral',
                          lof='local_outlier_fraction',
                          lmdd='deviation_based_outlier_detection',
                          lscb='locally_selective_combination_of_parallel_outlier_ensambles',
                          mcd='minimum_covariance_determinant',
                          ocsvm='one_class_support_vector_machine',
                          pca='principal_component_analysis',
                          sod='subspace_outlier_detection',
                          sos='stochastic_outlier_selection'
                          )

DETECTOR_TYPES: Dict[str, List[str]] = dict(linear=['lmdd', 'mcd', 'pca', 'ocsvm'],
                                            prox=['cblof', 'cof', 'hbos', 'loci', 'lof', 'knn', 'sod'],
                                            prob=['abod', 'sos'],
                                            ensamble=['ftbag', 'iforest', 'lscp', 'xgbod'],
                                            neural=['autoencoder', 'mo_gaal', 'so_gaal']
                                            )


class AnomalyDetectorException(Exception):
    """

    Class for handling exceptions for class AnomalyDetector

    """
    pass


class AnomalyDetector:
    """

    Class for detecting and analyzing anomalies / outliers

    """
    def __init__(self,
                 df: pd.DataFrame,
                 features: List[str] = None,
                 meth: List[str] = None,
                 params: dict = None,
                 outlier_threshold: float = 0.15,
                 detector_type: Dict[str, List[str]] = None,
                 seed: int = 1234
                 ):
        """
        Parameters
        ----------
        df: pd.DataFrame
            Data set

        features: List[str]
            Names of the features

        meth:
            Names of the supervised machine learning method to use
                -> None: K-Nearest-Neighbor
                -> abod: Angle-Based Outlier Detector
                -> abod: Angle-Based Outlier Detector
                -> abod: Angle-Based Outlier Detector
                -> abod: Angle-Based Outlier Detector
                -> abod: Angle-Based Outlier Detector
                -> abod: Angle-Based Outlier Detector
                -> abod: Angle-Based Outlier Detector
                -> abod: Angle-Based Outlier Detector
                -> abod: Angle-Based Outlier Detector
        params: dict
            Parameters of the supervised machine learning algorithm

        outlier_threshold: float
            Threshold to define case as outlier (univariate / multivariate)

        detector_type: Dict[str, List[str]]
            Names of the detector type (group of supervised machine learning algorithm)
                -> linear: Linear models (PCA, MCD, OCSVM, LMDD)
                -> prox: Proximity based models (LOF, COF, CBLOF, LOCI, HBOS, KNN, SOD)
                -> prob: Probabilistic models (ABOD, SOS)
                -> ensamble: Ensamble based models (IFOREST, FTBAG, LSCP, XGBOD)
                -> neural: Neural networks (AutoEncoder, SO_GAAL, MO_GAAL)

        seed: int
            Seed
        """
        self.seed: int = seed if seed > 0 else 1234
        self.df: pd.DataFrame = df
        self.features: List[str] = features
        self.cases: list = []
        self.outlier_threshold: float = outlier_threshold if outlier_threshold > 0 else 0.15
        self.params: dict = params
        self.meth: List[str] = []
        if detector_type is None:
            if meth is None:
                if self.params is None:
                    self.meth = ['knn']
                else:
                    for m in self.params.keys():
                        if m in ANOMALY_METH.keys():
                            self.meth.append(m)
                    if len(self.meth) == 0:
                        self.meth = ['knn']
            else:
                if len(meth) > 0:
                    for m in meth:
                        if m in ANOMALY_METH.keys():
                            self.meth.append(m)
                    if len(self.meth) == 0:
                        self.meth = ['knn']
                else:
                    self.meth = ['knn']
        else:
            for d in detector_type.keys():
                if d in DETECTOR_TYPES.keys():
                    self.meth.extend(DETECTOR_TYPES.get(d))
            if len(self.meth) == 0:
                self.meth = ['knn']

    def angle_based_outlier_detection(self) -> ABOD:
        """
        Config Angle-Based Outlier Detection

        Returns
        -------
        ABOD:
        """
        return ABOD(contamination=self.outlier_threshold,
                    n_neighbors=5,
                    method='fast'
                    )

    def cluster_based_local_outlier_factor(self) -> CBLOF:
        """
        Config Cluster-Based Local Outlier Factor

        Returns
        -------
        CBLOF:
        """
        return CBLOF(n_clusters=8,
                     contamination=self.outlier_threshold,
                     clustering_estimator=None,
                     alpha=0.9,
                     beta=5,
                     use_weights=False,
                     check_estimator=False,
                     random_state=self.seed,
                     n_jobs=1
                     )

    def connectivity_based_outlier_fraction(self) -> COF:
        """
        Config Angle-Based Outlier Detection

        Returns
        -------
        COF:
        """
        return COF(contamination=self.outlier_threshold,
                   n_neighbors=20
                   )

    def deviation_based_outlier_detection(self) -> LMDD:
        """
        Config Deviation-Based Outlier Detection

        Returns
        -------
        LMDD:
        """
        return LMDD(contamination=self.outlier_threshold,
                    n_iter=50,
                    dis_measure='aad',
                    random_state=self.seed
                    )

    def feature_bagging(self) -> FeatureBagging:
        """
        Config Feature Bagging

        Returns
        -------
        FeatureBagging:
        """
        return FeatureBagging(base_estimator=None,
                              n_estimators=10,
                              contamination=self.outlier_threshold,
                              max_features=1.0,
                              bootstrap_features=False,
                              check_detector=True,
                              check_estimator=False,
                              n_jobs=1,
                              random_state=self.seed,
                              combination='average',
                              verbose=0,
                              estimator_params=None
                              )

    def histogram_based_outlier_detection(self) -> HBOS:
        """
        Config Histogram-Based Outlier Detection

        Returns
        -------
        HBOS:
        """
        return HBOS(n_bins=10,
                    alpha=0.1,
                    tol=0.5,
                    contamination=self.outlier_threshold
                    )

    def isolation_forest(self) -> IForest:
        """
        Config Isolation Forest

        Returns
        -------
        IForest:
        """
        return IForest(n_estimators=100,
                       max_samples='auto',
                       contamination=self.outlier_threshold,
                       max_features=1.0,
                       bootstrap=False,
                       n_jobs=1,
                       behaviour='old',
                       random_state=self.seed,
                       verbose=0
                       )

    def k_nearest_neighbor(self) -> KNN:
        """
        Config K-Nearest-Neighbor

        Returns
        -------
        KNN:
        """
        return KNN(contamination=self.outlier_threshold,
                   n_neighbors=5,
                   method='largest',
                   radius=1.0,
                   algorithm='auto',
                   leaf_size=30,
                   metric='minkowski',
                   p=2,
                   metric_params=None
                   )

    def local_correlated_integral(self) -> LOCI:
        """
        Config Local Correlated Integral

        Returns
        -------
        LOCI:
        """
        return LOCI(contamination=self.outlier_threshold,
                    alpha=0.5,
                    k=3
                    )

    def local_outlier_fraction(self) -> LOF:
        """
        Config Local Outlier Fraction

        Returns
        -------
        LOF:
        """
        return LOF(n_neighbors=20,
                   algorithm='auto',
                   leaf_size=30,
                   metric='minkowski',
                   p=2,
                   metric_params=None,
                   contamination=self.outlier_threshold,
                   n_jobs=1
                   )

    def locally_selective_combination_of_parallel_outlier_ensambles(self) -> LSCP:
        """
        Config Locally Selective Combination of Parallel Outlier Ensambles

        Returns
        -------
        LSCP:
        """
        return LSCP(detector_list=None,
                    local_region_size=30,
                    local_max_features=1.0,
                    n_bins=10,
                    random_state=self.seed,
                    contamination=self.outlier_threshold
                    )

    def minimum_covariance_determinant(self) -> MCD:
        """
        Config Minimum Covariance Determinant

        Returns
        -------
        MCD:
        """
        return MCD(contamination=self.outlier_threshold,
                   store_precision=True,
                   assume_centered=False,
                   support_fraction=None,
                   random_state=self.seed
                   )

    def one_class_support_vector_machine(self) -> OCSVM:
        """
        Config One-Class Support Vector Machine

        Returns
        -------
        OCSVM:
        """
        return OCSVM(kernel='rbf',
                     degree=3,
                     gamma='auto',
                     coef0=0.0,
                     tol=0.001,
                     nu=0.5,
                     shrinking=True,
                     cache_size=200,
                     verbose=False,
                     max_iter=-1,
                     contamination=self.outlier_threshold
                     )

    def principal_component_analysis(self) -> PCA:
        """
        Config Principal Component Analysis

        Returns
        -------
        PCA:
        """
        return PCA(n_components=None,
                   n_selected_components=None,
                   contamination=self.outlier_threshold,
                   copy=True,
                   whiten=False,
                   svd_solver='auto',
                   tol=0.0,
                   iterated_power='auto',
                   random_state=self.seed,
                   weighted=True,
                   standardization=True
                   )

    def subspace_outlier_detection(self) -> SOD:
        """
        Config Subspace Outlier Detection

        Returns
        -------
        SOD:
        """
        return SOD(contamination=self.outlier_threshold,
                   n_neighbors=20,
                   ref_set=10,
                   alpha=0.8
                   )

    def stochastic_outlier_selection(self) -> SOS:
        """
        Config Stochastic Outlier Selection

        Returns
        -------
        SOS:
        """
        return SOS(contamination=self.outlier_threshold,
                   perplexity=4.5,
                   metric='euclidean',
                   eps=1e-5
                   )

    def univariate(self) -> dict:
        """
        Detect univariate outliers based on the Inter-Quantile-Range (IQR) algorithm

        Returns
        -------
        list: Index number of cases marked as outlier
        """
        _univariate: dict = {}
        for feature in self.features:
            self.cases = []
            _univariate.update({feature: dict(pred=[], cases=[])})
            _iqr: np.array = np.quantile(a=self.df[feature].values, q=0.75) - np.quantile(a=self.df[feature].values, q=0.25)
            _lower: bool = self.df[feature].values < (np.quantile(a=self.df[feature].values, q=0.25) - (1.5 * _iqr))
            _lower_cases: list = np.where(_lower)[0].tolist()
            _upper: bool = self.df[feature].values > (np.quantile(a=self.df[feature].values, q=0.75) + (1.5 * _iqr))
            _upper_cases: list = np.where(_upper)[0].tolist()
            self.cases.extend(_upper_cases)
            self.cases.extend(_lower_cases)
            _univariate[feature]['cases'] = self.cases
            _univariate[feature]['pred'] = [1 if case in self.cases else 0 for case in range(0, self.df.shape[0], 1)]
        return _univariate

    def multivariate(self) -> dict:
        """
        Detect multivariate outliers by using supervised machine learning algorithms

        Returns
        -------
        dict:
        """
        _anomaly_detection: dict = {}
        for ml in self.meth:
            _anomaly_detection.update({ml: {}})
            _model = getattr(self, ANOMALY_METH.get(ml))()
            _model.fit(self.df[self.features].values)
            _anomaly_detection[ml].update({'pred': _model.predict(self.df[self.features].values)})
        return _anomaly_detection
