# -*- coding: UTF-8 -*-
"""
@Project ：pythonProject1 
@File    ：demo.py
@IDE     ：PyCharm 
@Author  ：sdz0114
@Date    ：2024/12/5 20:50 
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from rich.progress import Progress
# 设定随机种子
np.random.seed(42)


class SigmaGenerator:
    """
    生成 Model 1 的协方差矩阵
    """

    def __init__(self, p):
        self.p = int(p)

    def A_1_sigma(self):
        """
        生成样本协方差和真实协方差
        """
        matrix = np.zeros((self.p, self.p))

        for i in range(self.p):
            for j in range(self.p):
                matrix[i][j] = max(0, 1 - abs(i - j) / 10)
        sample = np.random.multivariate_normal(np.zeros(self.p), matrix, 50)

        return 1 / 50 * np.dot(sample.T, sample), matrix, sample

    """
    生成 Model 2 的协方差矩阵
    """

    def A_2_sigma(self):
        """
        生成 A2
        """
        # 初始化 A2，范对角矩阵
        matrix = np.zeros((self.p, self.p))
        for i in range(self.p):
            for j in range(self.p):
                if i == j:
                    matrix[i][j] = 1
                elif j // 20 == i // 20:
                    matrix[i][j] = 0.4
                elif abs(j // 20 - i // 20) == 1 & (i % 19 == 0) & i != 0:
                    matrix[i][j] = 0.4
                elif abs(j // 20 - i // 20) == 1 & (j % 19 == 0) & j != 0:
                    matrix[i][j] = 0.4
                # elif (j//20+1)==i//20 & j%20==0:
                #     matrix[i][j]=0.4
                # else:
                #     matrix[i][j]=0
        sample = np.random.multivariate_normal(np.zeros(100), matrix, 50)
        return 1 / 50 * np.dot(sample.T, sample), matrix

class COV_ADMM:
    """
    利用ADMM算法对协方差矩阵进行估计
    """
    def __init__(self, sample):
        self.sample = sample
        self.matrix = 1/self.sample.shape[0]*np.dot(sample.T,sample)

    def cov(self,sample):
        return 1/sample.shape[0]*np.dot(sample.T,sample)

    def definite(self, X, lambd=10e-5):
        """
        正定性保证
        """
        y = np.zeros((X.shape[0], X.shape[1]))
        e_vals, e_vecs = np.linalg.eig(X)
        e_vals = [max(i, lambd) for i in e_vals]
        smat = np.zeros((X.shape[0], X.shape[1]))
        smat = np.diag(e_vals)

        return np.dot(e_vecs, np.dot(smat, np.linalg.inv(e_vecs)))

    def soft_threshold(self, X, lambd):
        """
        软阈值算子
        """
        y = np.zeros((X.shape[0], X.shape[1]))
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if i != j:
                    y[i][j] = np.sign(X[i][j]) * max(abs(X[i][j]) - lambd, 0)
                else:
                    y[i][j] = X[i][j]

        return y

    def cal_risk(
            self, cov_1: np.ndarray, cov_2: np.ndarray, norm_type: str
    ):
        """
        计算两个协方差矩阵之差的 norm

        Args:
            cov_1: 协方差矩阵 1
            cov_2: 协方差矩阵 2
            norm_type: norm 的类型，可选'Spectral norm' or 'Frobenius norm'

        Returns:
            risk: 两个协方差矩阵之差的 norm
        """
        # 断言 cov_1 和 cov_2 的形状相同
        # assert cov_1.shape == cov_2.shape, "cov_1 and cov_2 must have the same shape"
        diff = cov_1 - cov_2
        if norm_type == "Operator norm":
            risk = np.linalg.norm(diff, ord=2)
        elif norm_type == "Spectral norm":
            singular_values = np.linalg.svd(diff, full_matrices=False)[1]
            risk = np.max(singular_values)
            # risk = np.linalg.norm(diff, ord=1)
        elif norm_type == "Frobenius norm":
            risk = np.linalg.norm(diff, ord="fro")
        else:
            raise ValueError(
                "norm_type must be 'Operator norm', 'Spectral norm' or 'Frobenius norm'"
            )
        return risk

    def ADMM(self, mu, lambd,sample,max_iter=30):
        """
        Args:
            mu:
            lambd: lagrange multiplier
        Returns:
            sigma_esm_h:
            Lambd_h:
            theta_h:
        """
        matrix = self.cov(sample)
        n = matrix.shape[1]
        sigma_esm = np.zeros((n, n))
        Lambd = np.zeros((n, n))

        sigma_esm_h = [sigma_esm]
        Lambd_h = [Lambd]
        theta_h = []
        for i in range(max_iter):
            theta_h.append(self.definite(sigma_esm + mu * Lambd))
            sigma_esm = 1 / (1 + mu) * self.soft_threshold(mu * (matrix - Lambd[-1]) + theta_h[-1], lambd * mu)
            sigma_esm_h.append(sigma_esm)
            Lambd = Lambd_h[-1] - 1 / mu * (theta_h[-1] - sigma_esm_h[-1])
            e_vals, e_vecs = np.linalg.eig(sigma_esm)
            if sum([1 for i in e_vals if i > 0]) == len(e_vals):
                return sigma_esm_h, Lambd_h, theta_h
        return sigma_esm_h, Lambd_h, theta_h

    def get_best_esm_using_cv(self,N,norm_type):
        risks = []
        spliter = KFold(n_splits=N, shuffle=True, random_state=42)
        split_index = spliter.split(self.sample)
        for para in np.arange(0.01,1,0.01):
            risk = 0
            for train_sample, test_sample in split_index:
                train_cov = self.cov(self.sample[train_sample])
                test_cov = self.cov(self.sample[test_sample])
                train_cov_esm, Lambd_h, theta_h = self.ADMM(2,para,train_cov)
                risk += self.cal_risk(train_cov_esm[-1], test_cov, norm_type)
                # 将 risk 的均值添加到 risks 中
            risks.append(risk / N)
            para_best = 0.1*np.argmin(risks)
        return para_best


# 构造双重索引
idx = pd.MultiIndex.from_product(
    [
        ["Spectral norm", "Frobenius norm"],
        [100, 200, 500],
    ],
    names=["norm_type", "p"],
)
# 定义列名
col_names = ["1"]
# 创建空的 DataFrame
table_1 = pd.DataFrame(columns=col_names, index=idx)


# 重复 100 次，计算 risk
def replication(progress, task, p, norm_type):
    # 定义 risks 列表，用于存储 risk
    risk_list = []
    # 重复 100 次，计算 risk 的均值和标准差
    for _ in range(10):
        # 生成 100 个 p 元正态分布的随机向量
        sigma, matrix, sample = SigmaGenerator(p).A_1_sigma()

        ans=COV_ADMM(sample)
        para = ans.get_best_esm_using_cv(5, 'Spectral norm')
        train_cov_esm, Lambd_h, theta_h = ans.ADMM(2,para,sample)
        # 计算 risk

        risk = ans.cal_risk(cov_1=train_cov_esm[-1], cov_2=matrix, norm_type=norm_type)
        # 将 risk 加入 risk_list
        risk_list.append(risk)
        # 更新进度条
        progress.update(task, advance=1)
    return risk_list


with Progress() as progress:
    task = progress.add_task("[green] 正在计算。..", total=3 * 100)
    for p in [100, 200 , 500]:
        # 生成真实的 p 元正态分布的均值向量和协方差矩阵
        # sigma,matrix,sample = SigmaGenerator(p).A_1_sigma()
        for norm_type in ["Spectral norm", "Frobenius norm"]:
            # 重复 100 次，计算 risk
            risk_list = replication(progress, task, p, norm_type)
            # 计算 risk 的均值和标准差
            risk_mean = np.mean(risk_list)
            risk_std_error = np.std(risk_list) / np.sqrt(100)
            table_1.loc[(norm_type, p), 1] = f"{risk_mean:.2f} ({risk_std_error:.2f})"

table_1.to_csv('table_1.csv')