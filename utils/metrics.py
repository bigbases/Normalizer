import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


class StreamingMetrics:
    """Batch-wise MSE/MAE/RSE accumulated in float64 (no full-array copies)."""

    def __init__(self):
        self.n = 0
        self.sq = 0.0
        self.abs = 0.0
        self.true_sum = 0.0
        self.true_sq = 0.0

    def update(self, pred, true):
        pred = np.asarray(pred, dtype=np.float64)
        true = np.asarray(true, dtype=np.float64)
        diff = pred - true
        self.n += diff.size
        self.sq += float(np.sum(diff * diff))
        self.abs += float(np.sum(np.abs(diff)))
        self.true_sum += float(np.sum(true))
        self.true_sq += float(np.sum(true * true))

    def compute(self):
        if self.n == 0:
            raise ValueError('no test samples were evaluated')
        mse = self.sq / self.n
        mae = self.abs / self.n
        centered = self.true_sq - self.true_sum ** 2 / self.n
        rse = float(np.sqrt(self.sq) / np.sqrt(centered)) if centered > 0 else float('nan')
        return mse, mae, rse


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)

    return mae, mse, rmse, mape, mspe, rse, corr
