import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
import pandas as pd

def calculate_ef_for_prescreened_with_rank(predictions_df, total_molecules, total_hits, percentages=[1, 5, 10]):
    """
    计算预筛选数据的富集因子(Enrichment Factor, EF)
    使用排名信息而不是预测分数
    
    参数:
    predictions_df: DataFrame, 包含预测结果，需有'rank'和'is_active'列
    total_molecules: int, 整个数据集中的分子总数(固定值，如32000)
    total_hits: int, 整个数据集中真实结合分子的总数
    percentages: list, 要计算的百分比点列表
    
    返回:
    results: dict, 包含每个百分比点的EF值和其他统计信息
    """
    # 确保数据框包含必要的列
    required_columns = ['rank', 'label']
    if not all(col in predictions_df.columns for col in required_columns):
        raise ValueError(f"DataFrame必须包含以下列: {required_columns}")
    
    # 按排名排序（从低到高，排名1是最好的）
    sorted_df = predictions_df.sort_values('rank')
    
    results = {}
    
    for p in percentages:
        # 计算前x%的分子数量
        n_x_percent = int(np.ceil(total_molecules * p / 100))
        
        # 检查我们是否有足够的数据
        max_rank_in_data = sorted_df['rank'].max()
        if n_x_percent > max_rank_in_data:
            print(f"警告: 无法计算EF@{p}%，需要前{n_x_percent}个分子，但数据中最大排名为{max_rank_in_data}")
            results[p] = {
                'EF': np.nan,
                'Hits_in_top_percent': np.nan,
                'Molecules_in_top_percent': n_x_percent,
                'Total_hits': total_hits,
                'Total_molecules': total_molecules,
                'Expected_hits': (total_hits / total_molecules) * n_x_percent
            }
            continue
        
        # 获取前x%的分子（排名 ≤ n_x_percent）
        top_x_percent = sorted_df[sorted_df['rank'] <= n_x_percent]
        
        # 计算前x%中的真正结合分子数
        hits_x_percent = top_x_percent['label'].sum()
        
        # 计算富集因子
        expected_hits = (total_hits / total_molecules) * n_x_percent
        if expected_hits == 0:
            ef = float('inf')  # 避免除以零
        else:
            ef = hits_x_percent / expected_hits
        
        results[p] = {
            'EF': ef,
            'Hits_in_top_percent': hits_x_percent,
            'Molecules_in_top_percent': n_x_percent,
            'Total_hits': total_hits,
            'Total_molecules': total_molecules,
            'Expected_hits': expected_hits
        }
    
    return results

def plot_ef_curve(ef_results, save_path=None):
    """
    绘制富集因子曲线
    
    参数:
    ef_results: dict, calculate_ef函数的返回结果
    save_path: str, 可选，保存图像的路径
    """
    import matplotlib.pyplot as plt
    
    percentages = []
    ef_values = []
    
    for p, res in ef_results.items():
        if not np.isnan(res['EF']):
            percentages.append(p)
            ef_values.append(res['EF'])
    
    if not percentages:
        print("没有可用的EF数据用于绘图")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(percentages, ef_values, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Top Percentage (%)', fontsize=12)
    plt.ylabel('Enrichment Factor', fontsize=12)
    plt.title('Enrichment Factor Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 添加数据标签
    for i, (p, ef) in enumerate(zip(percentages, ef_values)):
        plt.annotate(f'EF={ef:.2f}', (p, ef), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def calculate_ef_curve(predictions_df, total_molecules, total_hits, max_percentage=10, num_points=100):
    """
    计算完整的EF曲线
    
    参数:
    predictions_df: DataFrame, 包含预测结果，需有'rank'和'is_active'列
    total_molecules: int, 整个数据集中的分子总数
    total_hits: int, 整个数据集中真实结合分子的总数
    max_percentage: float, 计算的最大百分比
    num_points: int, 计算点的数量
    
    返回:
    percentages: array, 百分比点
    ef_values: array, 对应的EF值
    """
    # 确保数据框包含必要的列
    if not all(col in predictions_df.columns for col in ['rank', 'label']):
        raise ValueError("DataFrame必须包含'rank'和'is_active'列")
    
    # 按排名排序（从低到高，排名1是最好的）
    sorted_df = predictions_df.sort_values('rank')
    
    # 生成百分比点
    percentages = np.linspace(0, max_percentage, num_points)
    ef_values = []
    
    for p in percentages:
        if p == 0:
            ef_values.append(0)  # 0%时的EF为0
            continue
            
        # 计算前p%的分子数量
        n_p_percent = int(np.ceil(total_molecules * p / 100))
        
        # 获取前p%的分子（排名 ≤ n_p_percent）
        top_p_percent = sorted_df[sorted_df['rank'] <= n_p_percent]
        
        # 计算前p%中的真正结合分子数
        hits_p_percent = top_p_percent['label'].sum()
        
        # 计算富集因子
        expected_hits = (total_hits / total_molecules) * n_p_percent
        if expected_hits == 0:
            ef = 0  # 避免除以零
        else:
            ef = hits_p_percent / expected_hits
        
        ef_values.append(ef)
    
    return np.array(percentages), np.array(ef_values)

def calculate_auc_ef(percentages, ef_values, normalization=False):
    """
    计算富集曲线下面积(AUC-EF)

    参数:
    percentages: list or np.array, 筛选百分比 (x轴)
    ef_values: list or np.array, 对应的富集因子 (y轴)
    normalization: bool, 是否进行标准化

    返回:
    auc_ef: float, 富集曲线下面积
    """
    # 计算实际EF曲线的面积
    # 使用关键字参数 y 和 x 明确指定数据对应的轴
    auc_ef = simpson(y=ef_values, x=percentages)

    if normalization:
        max_ef = np.max(ef_values) * 2
        ideal_auc = simpson(y=[0, max_ef, max_ef], x=[0, percentages[-1]/2, percentages[-1]])
        if ideal_auc > 0:
            return auc_ef / ideal_auc
        else:
            return auc_ef

    return auc_ef

def plot_ef_curve_with_auc(percentages, ef_values, auc_value, save_path=None):
    """
    绘制富集曲线并显示AUC值
    
    参数:
    percentages: array, 百分比点
    ef_values: array, 对应的EF值
    auc_value: float, AUC-EF值
    save_path: str, 可选，保存图像的路径
    """
    plt.figure(figsize=(5, 5))
    plt.plot(percentages, ef_values, 'b-', linewidth=2, label=f'EF Curve (AUC = {auc_value:.3f})')
    plt.axhline(y=1, color='r', linestyle='--', linewidth=1, label='Random (EF = 1)')
    
    plt.xlabel('Top Percentage (%)', fontsize=12)
    plt.ylabel('Enrichment Factor', fontsize=12)
    plt.title('Enrichment Factor Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.fill_between(percentages, 0, ef_values, alpha=0.2, color='blue')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()