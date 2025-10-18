


def fimo_filter(tf, motif_infos, pcc_genes, motif_scan_res, fimo_score=12, tf_f=None):
    """
    根据FIMO motif扫描结果过滤潜在的目标基因。
    注意: 此函数中的 'MYB_related' 是硬编码的，如果需要分析其他TF家族，需要修改。
    """
    print("--- Running FIMO Filter ---")
    
    # 建议: 可以将 'MYB_related' 替换为变量 tf_family 来增加通用性
    target_motif_ids = motif_infos[motif_infos['Family'] == tf_f]['Matrix_id'].tolist()
    
    motif_scan_res.index = [str(i).split(' ')[0] for i in motif_scan_res['seq_name']]
    
    down_gene_scan_res = motif_scan_res[motif_scan_res.index.isin(pcc_genes)]
    
    filtered_down_genes_df = down_gene_scan_res[
        (down_gene_scan_res['motif'].str.strip().isin(target_motif_ids)) & 
        (down_gene_scan_res['score'] >= fimo_score)
    ]
    
    scanned_genes = list(set(filtered_down_genes_df.index.tolist()))
    
    # 确保原始的TF总是在基因列表中
    if tf not in scanned_genes:
        scanned_genes.append(tf)
        
    print(f'After FIMO filtering, {len(scanned_genes)} genes remain.')
    return scanned_genes