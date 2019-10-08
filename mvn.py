import numpy as np
import scipy.stats as stats

from typing import Tuple


# * * * STATISTICAL TESTS * * *

def mardia(data: np.ndarray, cov: bool = True) -> Tuple[float, float, float, float]:
    """
    https://rdrr.io/cran/MVN/src/R/mvn.R
    https://stats.stackexchange.com/questions/317147/how-to-get-a-single-p-value-from-the-two-p-values-of-a-mardias-multinormality-t
    Mardia's multivariate skewness and kurtosis.
    Calculates the Mardia's multivariate skewness and kurtosis coefficients
    as well as their corresponding statistical test. For large sample size
    the multivariate skewness is asymptotically distributed as a Chi-square
    random variable; here it is corrected for small sample size. However,
    both uncorrected and corrected skewness statistic are presented. Likewise,
    the multivariate kurtosis it is distributed as a unit-normal.

     Syntax: function [Mskekur] = Mskekur(X,c,alpha)

     Inputs:
          X - multivariate data matrix [Size of matrix must be n(data)-by-p(variables)].
          cov - boolean to whether to normalize the covariance matrix by n (c=1[default]) or by n-1 (c~=1)

     Outputs:
          - skewness test statistic
          - kurtosis test statistic
          - significance value for skewness
          - significance value for kurtosis
    """
    n, p = data.shape

    # correct for small sample size
    small: bool = True if n < 20 else False

    if cov:
        S = ((n - 1)/n) * np.cov(data.T)
    else:
        S = np.cov(data.T)

    # calculate mean
    data_mean = data.mean(axis=0)
    # inverse - check if singular matrix
    try:
        iS = np.linalg.inv(S)
    except Exception as e:
        # print for now
        print(e)
        return 0.0, 0.0, 0.0, 0.0
    # squared-Mahalanobis' distances matrix
    D: np.ndarray = (data - data_mean) @ iS @ (data - data_mean).T
    # multivariate skewness coefficient
    g1p: float = np.sum(D**3)/n**2
    # multivariate kurtosis coefficient
    g2p: float = np.trace(D**2)/n
    # small sample correction
    k: float = ((p + 1)*(n + 1)*(n + 3))/(n*(((n + 1)*(p + 1)) - 6))
    # degrees of freedom
    df: float = (p * (p + 1) * (p + 2))/6

    if small:
        # skewness test statistic corrected for small sample: it approximates to a chi-square distribution
        g_skew = (n * g1p * k)/6
    else:
        # skewness test statistic:it approximates to a chi-square distribution
        g_skew = (n * g1p)/6

    # significance value associated to the skewness corrected for small sample
    p_skew: float = 1.0 - stats.chi2.cdf(g_skew, df)

    # kurtosis test statistic: it approximates to a unit-normal distribution
    g_kurt = (g2p - (p*(p + 2)))/(np.sqrt((8 * p * (p + 2))/n))
    # significance value associated to the kurtosis
    p_kurt: float = 2 * (1.0 - stats.norm.cdf(np.abs(g_kurt)))

    return g_skew, g_kurt, p_skew, p_kurt


def hztest(data: np.ndarray, cov: bool = True) -> Tuple[float, float]:
    """
    Henze-Zirkler method for goodness of fit of data to a multivariate normal distribution.
    Researchers tend to use this MVN test for larger samples (N > 100) which is our use case.
    https://www.tandfonline.com/doi/abs/10.1080/03610929008830400

    :param data: multivariate data matrix [Size of matrix must be n(data)-by-p(variables)].
    :param cov: boolean to whether to normalize the covariance matrix by n (c=1[default]) or by n-1 (c~=1)
    :return:
        hz - Henze-Zirkler test statistic
        p_value - significance value
    """
    n, p = data.shape

    if cov:
        S = ((n - 1)/n) * np.cov(data.T)
    else:
        S = np.cov(data.T)

    # calculate mean
    data_mean = data.mean(axis=0)

    try:
        iS = np.linalg.inv(S)
    except Exception as e:
        print(e)
        return 0.0, 0.0

    Y = data @ iS @ data.T
    Dj = np.diag((data - data_mean) @ iS @ (data - data_mean).T)

    Djk = - 2 * Y.T + np.tensordot(np.diag(Y.T), np.ones(n), axes=0) + np.tensordot(np.ones(n), np.diag(Y.T), axes=0)
    b: float = 1 / (np.sqrt(2)) * ((2 * p + 1) / 4) ** (1 / (p + 4)) * (n ** (1 / (p + 4)))

    # calculate rank of matrix
    S_rank = np.linalg.matrix_rank(S)

    if S_rank == p:
        HZ = n * (1 / (n ** 2) * np.sum(np.sum(np.exp(- (b ** 2) / 2 * Djk))) - 2 * ((1 + (b ** 2)) ** (- p / 2)) * (1 / n) * (np.sum(np.exp(- ((b ** 2) / (2 * (1 + (b ** 2)))) * Dj))) + ((1 + (2 * (b ** 2))) ** (- p / 2)))
    else:
        HZ = n * 4

    wb = (1 + b ** 2) * (1 + 3 * b ** 2)
    a = 1 + 2 * b ** 2

    # HZ mean
    mu = 1 - a ** (- p / 2) * (1 + p * b ** 2 / a + (p * (p + 2) * (b ** 4)) / (2 * a ** 2))  # HZ mean

    # HZ variance
    si2 = 2 * (1 + 4 * b ** 2) ** (- p / 2) + 2 * a ** (- p) * (1 + (2 * p * b ** 4) / a ** 2 + (3 * p * (p + 2) * b ** 8) / (4 * a ** 4)) - 4 * wb ** (- p / 2) * (1 + (3 * p * b ** 4) / (2 * wb) + (p * (p + 2) * b ** 8) / (2 * wb ** 2))

    pmu = np.log(np.sqrt(mu ** 4 / (si2 + mu ** 2)))  # lognormal HZ mean
    psi = np.sqrt(np.log((si2 + mu ** 2) / mu ** 2))  # lognormal HZ standard deviation

    # calculate p-value
    p_value = 1.0 - stats.lognorm.cdf(HZ, psi, scale=np.exp(pmu))

    return HZ, p_value
