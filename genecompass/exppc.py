import numpy as np
from scipy import stats


def vectorized_correlation(target_vector, matrix):
    """
    向量化计算相关系数和p值（更高效）
    """
    # 确保输入是numpy数组
    target = np.asarray(target_vector)
    matrix = np.asarray(matrix)
    
    # 计算相关系数
    n = len(target)
    target_mean = target.mean()
    matrix_mean = matrix.mean(axis=1, keepdims=True)
    
    # 计算协方差和标准差
    cov = np.dot(matrix - matrix_mean, target - target_mean) / (n - 1)
    std_target = np.std(target, ddof=1)
    std_matrix = np.std(matrix, axis=1, ddof=1)
    
    # 计算相关系数
    correlations = cov / (std_matrix * std_target)
    
    # 计算t统计量和p值
    t_values = correlations * np.sqrt((n - 2) / (1 - correlations**2))
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_values), n - 2))
    
    return correlations, p_values



def percentile_analysis(data, alpha=0.05):
    """
    高级百分位分析，包含统计信息和可视化
    """
    data = np.asarray(data)
    
    # 基本统计信息
    stats_info = {
        'mean': np.mean(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'median': np.median(data)
    }
    
    # 计算百分位数
    percentiles = np.percentile(data, [alpha*100, (1-alpha)*100, 25, 50, 75])
    
    # 获取索引
    lower_idx = np.where(data <= percentiles[0])[0]
    upper_idx = np.where(data >= percentiles[1])[0]
    
    # 概率密度估计
    kde = stats.gaussian_kde(data)
    x = np.linspace(data.min(), data.max(), 1000)
    density = kde(x)
    
    return {
        'data': data,
        'stats': stats_info,
        'percentiles': {
            'lower': percentiles[0],
            'upper': percentiles[1],
            'q1': percentiles[2],
            'median': percentiles[3],
            'q3': percentiles[4]
        },
        'indices': {
            'lower': lower_idx,
            'upper': upper_idx
        },
        'density': {
            'x': x,
            'y': density
        }
    }