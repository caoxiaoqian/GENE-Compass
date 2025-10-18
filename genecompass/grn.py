from typing import Optional, List, Dict
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed
from .portia import portia as pt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def protia1(expression_data: pd.DataFrame,
            gene_names: Optional[list[str]] = None,
            tf_names: Optional[list[str]] = None,
            ):
    """Run Portia GRN inference.
    
    Args:
        expression_data: Gene expression data (samples x genes)
        gene_names: List of gene names
        tf_names: List of transcription factor names
        method: Inference method to use
    
    Returns:
        M_bar: Inferred regulatory network
    """
    dataset = pt.GeneExpressionDataset()

    # Add each experiment to dataset
    for exp_id, data in expression_data.iterrows():
        dataset.add(pt.Experiment(exp_id, data))
    
    # Use column names if gene_names not provided
    if gene_names is None:
        gene_names = expression_data.columns.tolist()
    
    # Get indices of TFs
    tf_idx = list(np.where(np.isin(gene_names, tf_names))[0])
    
    # Run Portia inference
    M_bar, S_bar = pt.run(dataset, tf_idx=tf_idx, method='fast', return_sign=True)
    pd_M_bar = pd.DataFrame(M_bar, index=expression_data.columns.tolist(), columns=expression_data.columns.tolist())
    pd_S_bar = pd.DataFrame(S_bar, index=expression_data.columns.tolist(), columns=expression_data.columns.tolist())
    adj_matrix = pd_M_bar * pd_S_bar
    # g = nx.from_pandas_adjacency(adj_matrix)
    return adj_matrix

def random_forest_regressor(expression_data: pd.DataFrame,
                             gene_names: Optional[List[str]] = None,
                             tf_names: Optional[List[str]] = None,
                             target_gene_name: Optional[str] = None,
                             **kwargs): # 接受额外的 kwargs
    """使用随机森林回归模型预测功能基因表达并获取TF重要性."""
    if gene_names is None:
        gene_names = expression_data.columns.tolist()
    if target_gene_name not in gene_names:
        raise ValueError(f"Target gene '{target_gene_name}' not found in gene names.")

    tf_expression = expression_data[tf_names]
    target_gene_expression = expression_data[target_gene_name]

    model = RandomForestRegressor(random_state=42, **kwargs) # 传递 kwargs
    model.fit(tf_expression, target_gene_expression)

    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'TF': tf_names,
        'importance': feature_importances
    })
    return importance_df

def xgboost_regressor(expression_data: pd.DataFrame,
                        gene_names: Optional[List[str]] = None,
                        tf_names: Optional[List[str]] = None,
                        target_gene_name: Optional[str] = None,
                        **kwargs): # 接受额外的 kwargs
    """使用 XGBoost 回归模型预测功能基因表达并获取TF重要性."""
    if gene_names is None:
        gene_names = expression_data.columns.tolist()
    if target_gene_name not in gene_names:
        raise ValueError(f"Target gene '{target_gene_name}' not found in gene names.")

    tf_expression = expression_data[tf_names]
    target_gene_expression = expression_data[target_gene_name]

    model = XGBRegressor(random_state=42, **kwargs) # 传递 kwargs
    model.fit(tf_expression, target_gene_expression)

    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'TF': tf_names,
        'importance': feature_importances
    })
    return importance_df

# def lightgbm_regressor(expression_data: pd.DataFrame,
#                          gene_names: Optional[List[str]] = None,
#                          tf_names: Optional[List[str]] = None,
#                          target_gene_name: Optional[str] = None,
#                          **kwargs): # 接受额外的 kwargs
#     """使用 LightGBM 回归模型预测功能基因表达并获取TF重要性."""
#     if gene_names is None:
#         gene_names = expression_data.columns.tolist()
#     if target_gene_name not in gene_names:
#         raise ValueError(f"Target gene '{target_gene_name}' not found in gene names.")

#     tf_expression = expression_data[tf_names]
#     target_gene_expression = expression_data[target_gene_name]

#     model = LGBMRegressor(random_state=42, **kwargs) # 传递 kwargs
#     model.fit(tf_expression, target_gene_expression)

#     feature_importances = model.feature_importances_
#     importance_df = pd.DataFrame({
#         'TF': tf_names,
#         'importance': feature_importances
#     })
#     return importance_df

def svm_regressor(expression_data: pd.DataFrame,
                  gene_names: Optional[List[str]] = None,
                  tf_names: Optional[List[str]] = None,
                  target_gene_name: Optional[str] = None,
                  **kwargs):
    if gene_names is None:
        gene_names = expression_data.columns.tolist()
    if target_gene_name not in gene_names:
        raise ValueError(f"Target gene '{target_gene_name}' not found in gene names.")

    tf_expression = expression_data[tf_names]
    target_gene_expression = expression_data[target_gene_name]

    model = LinearSVR(**kwargs)
    model.fit(tf_expression, target_gene_expression)

    coefficients = model.coef_
    importance_df = pd.DataFrame({
        'TF': tf_names,
        'importance': np.abs(coefficients[0])
    })
    return importance_df



def bagging_ridge_posterior_p_values_ttest_cv(expression_data: pd.DataFrame,
                                           gene_names: Optional[List[str]] = None,
                                           tf_names: Optional[List[str]] = None,
                                           target_gene_name: Optional[str] = None,
                                           n_estimators=10,
                                           p_value=.01,
                                           ridge_alphas: Optional[List[float]] = None,
                                           cv: Optional[int] = 3):
    """
    Estimates posterior p-values for Bagging Ridge coefficients using one-sample t-test with RidgeCV.
    (Following CellOracle paper description).
    """
    if gene_names is None:
        gene_names = expression_data.columns.tolist()
    if target_gene_name not in gene_names:
        raise ValueError(f"Target gene '{target_gene_name}' not found in gene names.")

    if tf_names is None:
        raise ValueError("tf_names must be provided for Ridge regression.")

    tf_expression = expression_data[tf_names]
    target_gene_expression = expression_data[target_gene_name]

    if ridge_alphas is None:
        ridge_alphas = np.logspace(-3, 3, 10) # 提供一个默认的 alpha 值范围

    ridge_cv = RidgeCV(
        alphas=ridge_alphas,
        cv=cv, # 使用用户提供的 cv 值，如果没有则使用 RidgeCV 的默认值
    )

    bagging_ridge = BaggingRegressor(
        estimator=ridge_cv,
        n_estimators=n_estimators,
        random_state=42
    )
    bagging_ridge.fit(tf_expression, target_gene_expression)

    # 修复：直接访问基础估计器的 coef_ 属性，而不是通过 estimator_ 属性
    coefs = np.array([est.coef_ for est in bagging_ridge.estimators_])
    p_values_ttest = []
    mean_coefs = np.mean(coefs, axis=0)

    for j in range(coefs.shape[1]): # For each TF
        tf_coef_distribution = coefs[:, j]
        _, p_value = ttest_1samp(tf_coef_distribution, 0) # One-sample t-test against mu=0
        p_values_ttest.append(p_value)

    results_df = pd.DataFrame({
        'TF': tf_names,
        'importance': mean_coefs,
        'ttest_p_value': p_values_ttest
    })
    # 将p值大于p_value的importance置为0
    # results_df.loc[results_df['ttest_p_value'] > p_value, 'importance'] = 0
    return results_df

def bagging_lasso_posterior_p_values_ttest_cv(expression_data: pd.DataFrame,
                                          gene_names: Optional[List[str]] = None,
                                          tf_names: Optional[List[str]] = None,
                                          target_gene_name: Optional[str] = None,
                                          n_estimators=10,
                                          lasso_alphas: Optional[List[float]] = None,
                                          cv: Optional[int] = 3):
    """
    Estimates posterior p-values for Bagging Lasso coefficients using one-sample t-test with LassoCV.
    (Adapted from Bagging Ridge function, using Lasso regression with cross-validation).
    """
    if gene_names is None:
        gene_names = expression_data.columns.tolist()
    if target_gene_name not in gene_names:
        raise ValueError(f"Target gene '{target_gene_name}' not found in gene names.")

    if tf_names is None:
        raise ValueError("tf_names must be provided for Lasso regression.")

    tf_expression = expression_data[tf_names]
    target_gene_expression = expression_data[target_gene_name]

    if lasso_alphas is None:
        lasso_alphas = np.logspace(-3, 3, 10) # 提供一个默认的 alpha 值范围

    lasso_cv = LassoCV(
        alphas=lasso_alphas,
        cv=cv, # 使用用户提供的 cv 值，如果没有则使用 LassoCV 的默认值
        random_state=42, # LassoCV 也支持 random_state
        n_jobs=None, # 可以设置并行运行的作业数
        eps=1e-3, # Length of the path. As a fraction of the maximum alpha.
        max_iter=100 # Maximum number of iterations to run coordinate descent
    )

    bagging_lasso = BaggingRegressor(
        estimator=lasso_cv,
        n_estimators=n_estimators,
        random_state=42
    )
    bagging_lasso.fit(tf_expression, target_gene_expression)

    coefs = np.array([est.coef_ for est in bagging_lasso.estimators_]) # 注意这里需要访问 estimator_ 的 coef_
    p_values_ttest = []
    mean_coefs = np.mean(coefs, axis=0)

    for j in range(coefs.shape[1]): # For each TF
        tf_coef_distribution = coefs[:, j]
        _, p_value_val = ttest_1samp(tf_coef_distribution, 0) # One-sample t-test against mu=0
        p_values_ttest.append(p_value_val)

    results_df = pd.DataFrame({
        'TF': tf_names,
        'importance': mean_coefs,
        'ttest_p_value': p_values_ttest
    })
    # 将p值大于p_value的importance置为0
    return results_df

grns_optimized_v3 = {
    'random_forest': random_forest_regressor,
    'xgboost': xgboost_regressor,
    'svm': svm_regressor,
    'ridge': bagging_ridge_posterior_p_values_ttest_cv,
    'lasso': bagging_lasso_posterior_p_values_ttest_cv
}

class Meta_Grn_Optimized_v3:
    def __init__(self,
                 expression_data: pd.DataFrame,
                 gene_names: Optional[List[str]] = None,
                 tf_names: Optional[List[str]] = None,
                 target_gene_name: Optional[str] = None,
                 grn_methods: Optional[List[str]] = None,
                 method_params: Optional[dict] = None) -> None:
        self.expression_data = expression_data
        # print(f'The rna exp_data shape is {expression_data.shape}')
        self.gene_names = gene_names
        self.tf_names = tf_names
        self.target_gene_name = target_gene_name
        self.grn_methods = grn_methods
        self.method_params = method_params if method_params is not None else {}
        self.grn_results = {}
        
        # 运行每个回归模型
        for method in grn_methods:
            # print(f'Now, starting for {method} method!!!')
            if method in grns_optimized_v3:
                params = self.method_params.get(method, {}) # 获取当前方法的参数，如果没有则为空字典
                self.grn_results[method] = grns_optimized_v3[method](
                    self.expression_data,
                    gene_names=self.gene_names,
                    tf_names=self.tf_names,
                    target_gene_name=self.target_gene_name,
                    **params #  传递参数
                )
            else:
                raise ValueError(f"Unknown GRN method: {method}")


def process_single_tg(tg2tfs: Dict[str, List[str]],
                      data: pd.DataFrame,
                      grn_methods: List[str],
                      normalization_method: Optional[str] = None) -> Optional[pd.DataFrame]:
    if not grn_methods:
        raise ValueError("grn_methods cannot be empty")

    try:
        tg = list(tg2tfs.keys())[0]
        tfs = tg2tfs[tg]
        conexp_genes = [tg] + tfs if tg not in tfs else tfs  # Include target gene and TFs in expression_data
        expression_data_unnormalized = data.loc[:, conexp_genes].copy() # Keep original for normalization

        if not isinstance(expression_data_unnormalized, pd.DataFrame):
            raise ValueError("expression_data must be a pandas DataFrame")
        
        if grn_methods[0] == 'protial':
            sign_adj = protia1(expression_data_unnormalized, 
                        gene_names=conexp_genes, 
                        tf_names=tfs)
            return {tg: sign_adj}
        
        else:
            expression_data = expression_data_unnormalized.copy() # Work on a copy to avoid modifying original

            # Apply normalization if specified
            if normalization_method == 'zscore':
                scaler = StandardScaler()
                expression_data = pd.DataFrame(
                    scaler.fit_transform(expression_data),
                    columns=expression_data.columns,
                    index=expression_data.index
                )
                # print(f"Applied z-score normalization to target gene: {tg}")
            elif normalization_method == 'minmax':
                scaler = MinMaxScaler()
                expression_data = pd.DataFrame(
                    scaler.fit_transform(expression_data),
                    columns=expression_data.columns,
                    index=expression_data.index
                )
                # print(f"Applied min-max normalization to target gene: {tg}")
            elif normalization_method is not None:
                raise ValueError(f"Unknown normalization method: {normalization_method}. \
                                 Choose from 'zscore', 'minmax', or None.")

            grn = Meta_Grn_Optimized_v3(
                expression_data=expression_data, # Use normalized expression data
                gene_names=expression_data.columns.tolist(), # Ensure gene_names match columns after normalization
                tf_names=tfs,  # Use only TFs as features, excluding target gene
                target_gene_name=tg,
                grn_methods=grn_methods
            )

            if len(grn_methods) == 1:
                return {tg: grn.grn_results[grn_methods[0]]}
            else:
                return Meta_Grn_Optimized_v3.rank_grn(grn.grn_results, grn_methods)

    except Exception as e:
        print(f"Error processing target gene {tg}: {str(e)}")
        return None

def parallel_grn(tg2tfs_dct, 
                 data, 
                 grn_methods, 
                 normalization_method, 
                 n_jobs=-1):
    """Run GRN analysis in parallel with proper error handling.
    
    Args:
        sig_dfs: Dictionary of significant TFs for each target gene
        data: Expression data (pandas DataFrame, samples x genes)
        grn_methods: List of GRN methods to use
        n_jobs: Number of parallel jobs
        
    Returns:
        List of GRN results for each target gene
    """
    # Validate inputs
    if data.empty:
        raise ValueError("data cannot be empty")
    if grn_methods == 'protial':
        raise ValueError('parallel_grn function not support the protial, as it can completely \
                         construct the grn, please use the process_single_tg or protia1 function.')
        
    tgs = list(tg2tfs_dct.keys())
    
    try:
        from tqdm import tqdm
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_single_tg)({tg: tg2tfs_dct[tg]}, data, grn_methods, normalization_method)
            for tg in tqdm(tgs, desc="Processing target genes", total=len(tgs))
        )
        # Filter out None results from failed jobs
        return [res for res in results if res is not None]
    except Exception as e:
        print(f"Error in parallel execution: {str(e)}")
        return [] 