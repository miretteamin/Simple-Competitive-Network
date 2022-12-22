import numpy as np


class PCA:
    def __init__(self, n):
        self.num_comopnents = n
        self.__covMat = 0
        self.__eigenVec = 0
        self.__eigenVals = 0
        self.__meanedData = 0

    def fit(self, data):        
        # rowvar makes meanedData transposed
        self.__meanedData = data - np.mean(data , axis = 0)        
        self.__covMat = np.cov(self.__meanedData , rowvar = False)
        self.__eigenVals , self.__eigenVec = np.linalg.eigh(self.__covMat)        
        
        # Sort Eigen Vectors and Eigen Values
        idx_sorted = np.argsort(self.__eigenVals)[::-1]
        self.__eigenVals = self.__eigenVals[idx_sorted]
        self.__eigenVec = self.__eigenVec[:,idx_sorted]

    def getCovarianceMat(self):
        return self.__covMat

    def getEigenVectors(self):
        return self.getEigenVectors

    def getEigenVals(self):
        return self.getEigenVals

    def transform(self):
        subset = self.__eigenVec[:,0:self.num_comopnents]
        reduced = np.dot(subset.transpose(), self.__meanedData.transpose()).transpose()
        return reduced

    def fit_transform(self, data):
        self.fit(data)
        res = self.transform()
        return res