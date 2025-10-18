# -*- coding: utf-8 -*-
from gimmemotifs.scanner import Scanner
from gimmemotifs.fasta import Fasta
from Bio.Seq import Seq
from gimmemotifs.motif import read
import os
import pandas as pd

def scan_motifs(motif_dir, 
                genome_file,
                fasta_file = None, 
                fasta_seq_dct = None, 
                fpr=0.01, 
                score_threshold=10, 
                crop_range=5, 
                ncpus=20, 
                scan_rc=False):
    """Scan sequences for motifs and return results as DataFrame"""
    # Read motifs
    motif_paths = [os.path.join(motif_dir, i) for i in os.listdir(motif_dir)]
    motifs = []
    for path in motif_paths:
        ms = read.read_motifs(path, as_dict=False, fmt='pwm')
        if len(ms) > 0:
            motifs.append(ms[0])
    
    # Initialize scanner
    s = Scanner(ncpus=ncpus, progress=True)
    s.set_motifs(motifs)
    s.set_genome(genome_file)
    s.set_threshold(fpr=fpr)
    
    # Read sequences
    if fasta_seq_dct is None:
        seqs = Fasta(fasta_file)
        id2seq = {idx: Seq(seq) for idx, seq in zip(seqs.ids, seqs.seqs)}
    else:
        seqs = Fasta(fdict = fasta_seq_dct)
        id2seq = {idx: Seq(seq) for idx, seq in zip(seqs.ids, seqs.seqs)}
    
    # Scan sequences
    results = []
    # Precompute motif lengths to avoid repeated calculations
    motif_lengths = [m.pwm.shape[0] for m in motifs]
    
    # Use list comprehension for faster processing
    results = [
        {
            'seq_name': seqs.ids[i],
            'motif': motifs[m].id.strip(),
            'score': score,
            'position': (pos, pos - motif_lengths[m] - crop_range, 
                        pos + motif_lengths[m] + crop_range),
            'strand': strand,
            'crop_seq': str(
            id2seq[seqs.ids[i]][pos - motif_lengths[m] - crop_range : pos + motif_lengths[m] + crop_range].reverse_complement()
            if strand == '-' else # neg_strand 在这里的标记为-1.
            id2seq[seqs.ids[i]][pos - motif_lengths[m] - crop_range : pos + motif_lengths[m] + crop_range]
            )
        }
        for i, result in enumerate(s.scan(seqs, scan_rc=scan_rc))
        for m, matches in enumerate(result)
        for score, pos, strand in matches
        if score > score_threshold
    ]
    
    return pd.DataFrame(results)

# -----------------------------
# CLI support
# -----------------------------
def _build_arg_parser():
    import argparse
    parser = argparse.ArgumentParser(
        description="Scan sequences for motifs using GimmeMotifs and output results as CSV."
    )
    parser.add_argument("-m", "--motif-dir", required=True, help="Directory containing motif PWM files.")
    parser.add_argument("-g", "--genome-file", required=True, help="Reference genome FASTA file.")
    parser.add_argument("-f", "--fasta-file", required=False, help="FASTA file of sequences to scan.")
    parser.add_argument("--fpr", type=float, default=0.01, help="False positive rate used to set threshold. Default: 0.01")
    parser.add_argument("--score-threshold", type=float, default=10, help="Score threshold to filter matches. Default: 10")
    parser.add_argument("--crop-range", type=int, default=5, help="Crop range around motif. Default: 5")
    parser.add_argument("--ncpus", type=int, default=20, help="Number of CPUs to use. Default: 20")
    parser.add_argument("--scan-rc", action="store_true", help="Scan reverse-complement strand as well.")
    parser.add_argument("-o", "--output", required=False, help="Output CSV file path. If not provided, prints to stdout.")
    return parser

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.fasta_file is None:
        raise SystemExit("Error: --fasta-file is required for CLI usage.")

    df = scan_motifs(
        motif_dir=args.motif_dir,
        genome_file=args.genome_file,
        fasta_file=args.fasta_file,
        fpr=args.fpr,
        score_threshold=args.score_threshold,
        crop_range=args.crop_range,
        ncpus=args.ncpus,
        scan_rc=args.scan_rc
    )

    # Flatten position tuple columns into start, left, right for easier CSV usage
    if not df.empty and "position" in df.columns:
        pos_df = df["position"].apply(pd.Series)
        pos_df.columns = ["pos", "left", "right"]
        df = pd.concat([df.drop(columns=["position"]), pos_df], axis=1)

    if args.output:
        out_dir = os.path.dirname(os.path.abspath(args.output))
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"Saved {len(df)} rows to {args.output}")
    else:
        # Print to stdout
        print(df.to_csv(index=False))

if __name__ == "__main__":
    main()
