# -*- coding: utf-8 -*-
import os
from pathlib import Path
import sys
import argparse
import pandas as pd
from Bio.Seq import Seq
from Bio import SeqIO
from pymemesuite.common import MotifFile, Sequence
from pymemesuite.fimo import FIMO
from tqdm import tqdm

# -----------------------------
# 现有辅助函数
# -----------------------------
def _read_first_motif_with_bg(motif_file_path):
    """
    读取 MEME 文件中的第一个 motif 以及背景频率。
    返回 (motif, background)
    """
    with MotifFile(motif_file_path) as mf:
        motif = mf.read()  # 读取第一个 motif
        background = mf.background
    return motif, background

def read_meme_paths(motif_dir):
    """
    返回 motif_dir 下所有 .meme 文件路径（递归）
    """
    motif_dir = Path(motif_dir)
    return [str(p) for p in motif_dir.rglob("*.meme")]

# -----------------------------
# 单核实现（保留原函数）
# -----------------------------
def scan_motifs(
    motif_dir,
    fasta_file=None,
    score_threshold=10,
    crop_range=5,
    scan_rc=False
):
    """
    使用 FIMO 扫描序列，返回 DataFrame。
    保持原有数据结构，并新增 p_value 与 q_value 两列。
    """
    # 读取 motifs（及其背景）
    meme_paths = read_meme_paths(motif_dir)
    motifs_with_bg = []
    for mp in meme_paths:
        try:
            motif, bg = _read_first_motif_with_bg(mp)
            motifs_with_bg.append((mp, motif, bg))
        except Exception:
            # 跳过无法读取的文件
            continue
    if not motifs_with_bg:
        return pd.DataFrame(columns=[
            'seq_name', 'motif', 'score', 'position', 'strand', 'crop_seq', 'p_value', 'q_value'
        ])

    # 读取序列并构造 id -> Seq 映射，以及 FIMO Sequence 列表
    fasta_seq_dct = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))
    seq_strings = [str(fasta_seq_dct[idx].seq) for idx in fasta_seq_dct.keys()]
    ids = list(fasta_seq_dct.keys())

    id2seq = {idx: Seq(seq) for idx, seq in zip(ids, seq_strings)}
    # FIMO 需要 Sequence(name=bytes, sequence=str)
    fimo_sequences = [Sequence(str(id2seq[idx]), name=idx.encode()) for idx in ids]
    # 构造快速长度索引
    id2len = {idx: len(id2seq[idx]) for idx in ids}

    # 初始化 FIMO
    fimo = FIMO(both_strands=scan_rc)

    results = []
    for mp, motif, bg in tqdm(motifs_with_bg, desc='Scanning motifs'):
        # motif 名称
        motif_name = str(mp.stem)
        # 扫描
        pattern = fimo.score_motif(motif, fimo_sequences, bg)

        for m in pattern.matched_elements:
            # 取基础字段
            seq_name = m.source.accession.decode() if hasattr(m.source.accession, "decode") else str(m.source.accession)
            start1 = int(m.start)   # FIMO 通常为 1-based
            stop1  = int(m.stop)
            score  = float(m.score)
            pval   = float(m.pvalue) if m.pvalue is not None else None
            qval   = float(m.qvalue) if m.qvalue is not None else None

            # 过滤分数阈值
            if score <= score_threshold:
                continue

            # 兼容 strand 表达
            raw_strand = m.strand
            if raw_strand in ['+', '-']:
                strand = raw_strand
            elif raw_strand in [1, 'P', 'p']:
                strand = '+'
            elif raw_strand in [-1, 'N', 'n']:
                strand = '-'
            else:
                strand = '+'

            # 计算 motif_len 与 0-based 起始位置 pos0
            motif_len = max(1, stop1 - start1 + 1)
            pos0 = start1 - 1  # 将 FIMO 的 1-based 转为 Python 0-based

            # 保持原数据结构的 position：使用 pos0 作为“pos”
            # 三元组：(pos, pos - motif_len - crop_range, pos + motif_len + crop_range)
            left = max(0, pos0 - motif_len - crop_range)
            right = min(id2len[seq_name], pos0 + motif_len + crop_range)

            # 裁剪序列
            subseq = id2seq[seq_name][left:right]
            if strand == '-':
                subseq = subseq.reverse_complement()

            results.append({
                'seq_name': seq_name,
                'motif': motif_name,
                'score': score,
                'position': (pos0, pos0 - motif_len - crop_range, pos0 + motif_len + crop_range),
                'strand': strand,
                'crop_seq': str(subseq),
                'p_value': pval,
                'q_value': qval
            })

    return pd.DataFrame(results)

# -----------------------------
# 多核实现（按 .meme 文件并行）
# -----------------------------
# 全局缓存（在每个 worker 进程中初始化），只存放可序列化的基本类型
_G_IDS = None
_G_SEQ_STRINGS = None
_G_ID2LEN = None

def _init_worker(ids, seq_strings):
    """在每个进程内初始化共享的序列数据（纯字符串与 ID）"""
    global _G_IDS, _G_SEQ_STRINGS, _G_ID2LEN
    _G_IDS = list(ids)
    _G_SEQ_STRINGS = list(seq_strings)
    _G_ID2LEN = {idx: len(seq) for idx, seq in zip(_G_IDS, _G_SEQ_STRINGS)}

def _scan_one_meme_path(mp_path, score_threshold, crop_range, scan_rc):
    """子进程执行：扫描单个 motif 文件"""
    from Bio.Seq import Seq
    from pymemesuite.common import Sequence
    from pymemesuite.fimo import FIMO

    # 读取 motif 与背景
    motif, bg = _read_first_motif_with_bg(mp_path)

    # 重建 FIMO 所需的对象
    id2seq = {idx: Seq(seq) for idx, seq in zip(_G_IDS, _G_SEQ_STRINGS)}
    fimo_sequences = [Sequence(_G_SEQ_STRINGS[i], name=_G_IDS[i].encode()) for i in range(len(_G_IDS))]
    id2len = _G_ID2LEN

    # 扫描
    fimo = FIMO(both_strands=scan_rc)
    pattern = fimo.score_motif(motif, fimo_sequences, bg)

    motif_name = Path(mp_path).stem
    
    out_rows = []
    for m in pattern.matched_elements:
        seq_name = m.source.accession.decode() if hasattr(m.source.accession, "decode") else str(m.source.accession)
        start1 = int(m.start)
        stop1  = int(m.stop)
        score  = float(m.score)
        pval   = float(m.pvalue) if m.pvalue is not None else None
        qval   = float(m.qvalue) if m.qvalue is not None else None

        if score <= score_threshold:
            continue

        raw_strand = m.strand
        if raw_strand in ['+', '-']:
            strand = raw_strand
        elif raw_strand in [1, 'P', 'p']:
            strand = '+'
        elif raw_strand in [-1, 'N', 'n']:
            strand = '-'
        else:
            strand = '+'

        motif_len = max(1, stop1 - start1 + 1)
        pos0 = start1 - 1

        left = max(0, pos0 - motif_len - crop_range)
        right = min(id2len[seq_name], pos0 + motif_len + crop_range)

        subseq = id2seq[seq_name][left:right]
        if strand == '-':
            subseq = subseq.reverse_complement()

        out_rows.append({
            'seq_name': seq_name,
            'motif': motif_name,
            'score': score,
            'position': (pos0, pos0 - motif_len - crop_range, pos0 + motif_len + crop_range),
            'strand': strand,
            'crop_seq': str(subseq),
            'p_value': pval,
            'q_value': qval
        })
    return out_rows

def scan_motifs_parallel(
    motif_dir,
    fasta_file,
    score_threshold=10,
    crop_range=5,
    scan_rc=False,
    max_workers=None,
    chunk_size=1
):
    """
    并行扫描 motif：
    - 任务单元：单个 .meme 文件
    - 并行度：默认为 os.cpu_count()
    - 返回：与 scan_motifs 相同结构的 DataFrame
    """
    from multiprocessing import get_context
    meme_paths = read_meme_paths(motif_dir)
    if not meme_paths:
        return pd.DataFrame(columns=[
            'seq_name', 'motif', 'score', 'position', 'strand', 'crop_seq', 'p_value', 'q_value'
        ])

    # 读取 fasta 一次，广播到 worker
    fasta_seq_dct = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))
    ids = list(fasta_seq_dct.keys())
    seq_strings = [str(fasta_seq_dct[idx].seq) for idx in ids]

    if max_workers is None or max_workers <= 0:
        max_workers = os.cpu_count() or 1

    ctx = get_context("fork")  # Linux/macOS 下使用 fork；Windows 会自动用 spawn
    all_rows = []
    with ctx.Pool(processes=max_workers, initializer=_init_worker, initargs=(ids, seq_strings)) as pool:
        # 通过 starmap 或 imap_unordered 提交任务
        from functools import partial
        worker = partial(_scan_one_meme_path,
                         score_threshold=score_threshold,
                         crop_range=crop_range,
                         scan_rc=scan_rc)
        for rows in pool.imap_unordered(worker, meme_paths, chunksize=chunk_size):
            if rows:
                all_rows.extend(rows)

    return pd.DataFrame(all_rows)

# -----------------------------
# 命令行入口
# -----------------------------
def _write_output(df: pd.DataFrame, out: str, fmt: str):
    fmt = (fmt or "csv").lower()
    if fmt == "csv":
        df.to_csv(out, index=False)
    elif fmt == "tsv":
        df.to_csv(out, sep="\t", index=False)
    elif fmt == "parquet":
        df.to_parquet(out, index=False)
    else:
        raise ValueError(f"Unsupported out-format: {fmt}")

def main(argv=None):
    parser = argparse.ArgumentParser(description="FIMO motif scanner (single or multi-core CLI)")
    parser.add_argument('-md', "--motif-dir", required=True, help="目录，递归查找 *.meme")
    parser.add_argument('-f', "--fasta", required=True, help="FASTA 序列文件")
    parser.add_argument('-t',"--threshold", type=float, default=10.0, help="分数阈值（过滤 <= 阈值）")
    parser.add_argument('-crange',"--crop-range", type=int, default=5, help="裁剪片段范围（左右各加）")
    parser.add_argument('-rc', "--scan-rc", action="store_true", help="是否同时扫描反向互补链")
    parser.add_argument('-j', "--jobs", type=int, default=0, help="并行进程数（0 表示自动使用 CPU 核心数；1 表示单核）")
    parser.add_argument('-cs', "--chunk-size", type=int, default=1, help="每个 worker 的任务批大小（对大量 .meme 时可调优）")
    parser.add_argument('-o',"--out", required=True, help="输出文件路径")
    parser.add_argument('-of',"--out-format", default="csv", choices=["csv", "tsv", "parquet"], help="输出格式")
    parser.add_argument('-bar',"--progress", action="store_true", help="单核时显示进度条")
    args = parser.parse_args(argv)

    motif_dir = Path(args.motif_dir)
    fasta_file = args.fasta
    score_threshold = args.threshold
    crop_range = args.crop_range
    scan_rc = args.scan_rc
    jobs = args.jobs
    chunk_size = args.chunk_size
    out = args.out
    out_fmt = args.out_format

    # 参数校验
    if not Path(motif_dir).exists():
        print(f"[Error] motif-dir 不存在: {motif_dir}", file=sys.stderr)
        sys.exit(2)
    if not Path(fasta_file).exists():
        print(f"[Error] fasta 不存在: {fasta_file}", file=sys.stderr)
        sys.exit(2)

    # 选择单核或多核
    if jobs == 1:
        # 单核；可选显示进度条（已在 scan_motifs 内加入 tqdm）
        df = scan_motifs(
            motif_dir=motif_dir,
            fasta_file=fasta_file,
            score_threshold=score_threshold,
            crop_range=crop_range,
            scan_rc=scan_rc
        )
    else:
        # 多核（jobs=0 使用 os.cpu_count()）
        max_workers = jobs if jobs and jobs > 0 else None
        df = scan_motifs_parallel(
            motif_dir=motif_dir,
            fasta_file=fasta_file,
            score_threshold=score_threshold,
            crop_range=crop_range,
            scan_rc=scan_rc,
            max_workers=max_workers,
            chunk_size=chunk_size
        )

    # 写出结果
    _write_output(df, out, out_fmt)
    print(f"完成：{len(df)} 条匹配写入 -> {out} ({out_fmt})")

if __name__ == "__main__":
    main()