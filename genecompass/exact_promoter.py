from Bio import SeqIO
from Bio.Seq import Seq
from typing import Dict, List, Generator, Any 
import argparse
import sys
import re
import gzip 


def parse_gff3(file_path: str) -> Generator[List[Dict[str, Any]], None, None]:
    """
    解析GFF3文件，返回包含基因组特征的生成器
    每个特征以字典形式返回，包含所有标准字段和解析后的属性字段
    按基因分组处理，返回每个基因的所有特征 (assuming groups are separated by ### or implicitly by file end)
    """
    current_gene = []
    open_func = gzip.open if file_path.endswith('.gz') else open
    read_mode = 'rt' if file_path.endswith('.gz') else 'r'
    print(f"Parsing GFF using generator: {file_path}")

    try:
        with open_func(file_path, read_mode, encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue

                # Check for standard GFF separator, handle end of file later
                if line == '###':
                    if current_gene:
                        yield current_gene
                        current_gene = []
                    continue

                if line.startswith('#'):
                    continue

                fields = line.split('\t')
                if len(fields) != 9:
                    # print(f"Skipping malformed line: {line}", file=sys.stderr)
                    continue

                seqid, source, type_, start_str, end_str, score_str, strand, phase_str, attributes_str = fields

                try:
                    start = int(start_str)
                    end = int(end_str)
                except ValueError:
                    # print(f"Skipping line with invalid coordinates: {line}", file=sys.stderr)
                    continue

                feature = {
                    'seqid': seqid,
                    'source': source,
                    'type': type_,
                    'start': start,
                    'end': end,
                    'score': None if score_str == '.' else float(score_str),
                    'strand': strand,
                    'phase': None if phase_str == '.' else int(phase_str),
                    'attributes': {}
                }

                for pair in attributes_str.split(';'):
                    if '=' in pair:
                        try:
                            key, value = pair.strip().split('=', 1)
                            feature['attributes'][key] = value
                        except ValueError:
                            # Handle cases like flag attributes without '='
                            # print(f"Skipping attribute without '=': {pair} in line: {line}", file=sys.stderr)
                            pass


                # --- Grouping Logic ---
                # Decide how to group. Simplest: group by features sharing the same *first* field (seqid)?
                # Or rely *only* on '###' separator? Let's rely on '###' for now.
                current_gene.append(feature)

            # Yield the last gene group if file doesn't end with ###
            if current_gene:
                yield current_gene
        print("Finished parsing GFF.")
    except FileNotFoundError:
        print(f"Error: GFF file not found: {file_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading GFF file {file_path}: {type(e).__name__} - {e}", file=sys.stderr)
        sys.exit(1)


def find_tss(gene_lst):
    # Extract gene basic information
    gene_features = [feature for feature in gene_lst if feature['type'] == 'gene']
    if not gene_features:
        return None
        
    first_gene = gene_features[0]
    chr_num = first_gene['seqid']
    strand = first_gene['strand']
    gene_name = first_gene['attributes'].get('ID', None)
    
    cds_starts = []
    cds_ends = []
    
    # Collect all CDS start and end positions
    for feature in gene_lst:
        if feature['type'] == 'CDS':
            cds_starts.append(feature['start'])
            cds_ends.append(feature['end'])
    
    # Determine TSS based on strand direction
    if strand == '+':
        tss = min(cds_starts) if cds_starts else None
    elif strand == '-':
        tss = max(cds_ends) if cds_ends else None
    else:
        tss = None  # Handle non-standard strand values

    if tss is not None and gene_name is not None:
        return [gene_name.split(':')[-1], chr_num, tss, strand]
    return None



# ===========================================
# Main Extraction and Writing Logic (Now in Main or Helper)
# ===========================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract promoter regions relative to the first CDS start (+ strand) or last CDS end (- strand), format case, and write to a single FASTA file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('--gff', required=False, 
                        default='./odata/Rosa_chinensis/Rosa_chinensis.RchiOBHm-V2.60.gff3',
                        help='Input GFF3 annotation file (can be .gz). Assumes ### gene separator or groups implicitly.')
    parser.add_argument('-g', '--genome', required=False, 
                        default='./odata/Rosa_chinensis/Rosa_chinensis.RchiOBHm-V2.dna_sm.toplevel.fa', 
                        help='Input Genome FASTA file (can be .gz).')
    parser.add_argument('-o', '--output', required=False, 
                        default = './Rosa_chinensis_promoter.fasta',
                        help='Output FASTA file name (all promoters written here).')
    parser.add_argument('-u', '--upstream', type=int, default=2000, help='Bases UPSTREAM of CDS anchor (will be uppercase).')
    parser.add_argument('-d', '--downstream', type=int, default=200, help='Bases DOWNSTREAM of CDS anchor, including anchor position (will be lowercase).')

    args = parser.parse_args()

    # --- Load Genome into Memory ---
    print(f"\nLoading genome file '{args.genome}' into memory...")
    try:
        genome_dict = SeqIO.to_dict(SeqIO.parse(args.genome, "fasta"))
        print(f"Loaded {len(genome_dict)} sequences from genome file.")
        if not genome_dict:
             print("Error: No sequences loaded from the genome file.", file=sys.stderr)
             sys.exit(1)
    except FileNotFoundError:
        print(f"Error: Genome file not found at '{args.genome}'", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading genome file '{args.genome}': {type(e).__name__} - {e}", file=sys.stderr)
        sys.exit(1)

    # --- Process GFF and Extract Promoters Gene by Gene ---
    print(f"\nProcessing GFF and extracting promoters to: {args.output}...")
    success_count = 0
    error_count = 0
    gff_gene_groups_processed = 0

    try:
        with open(args.output, 'w') as outfile:
            # Iterate through gene feature lists yielded by the parser
            for gene_feature_list in parse_gff3(args.gff):
                gff_gene_groups_processed += 1
                if not gene_feature_list: continue # Skip empty groups

                # Find the anchor point (TSS proxy) using your function
                tss_info = find_tss(gene_feature_list)

                if tss_info is None:
                    error_count += 1
                    continue # Skip if TSS anchor couldn't be determined

                gene_id, chrom, tss_anchor, strand = tss_info

                # --- Core Extraction Logic (adapted from previous function) ---
                if chrom not in genome_dict:
                    print(f"Warning: Chromosome '{chrom}' for gene '{gene_id}' not found in genome. Skipping.", file=sys.stderr)
                    error_count += 1
                    continue
                if tss_anchor is None:
                    print(f"Warning: gene '{gene_id}' not found tts. Skipping.", file=sys.stderr)
                    error_count += 1
                    continue

                genome_seq_record = genome_dict[chrom]
                chrom_len = len(genome_seq_record.seq)
                promoter_seq_final_case = None
                promoter_start, promoter_end = 0, 0
                split_index_in_extracted = -1

                # --- Coordinate Calculation (1-based) using tss_anchor ---
                if strand == '+':
                    # tss_anchor is min(cds_starts)
                    promoter_start = max(1, tss_anchor - args.upstream)
                    promoter_end = min(chrom_len, tss_anchor + args.downstream - 1)
                    if promoter_start > promoter_end: promoter_end = promoter_start
                    split_index_in_extracted = max(0, tss_anchor - promoter_start)

                elif strand == '-':
                    # tss_anchor is max(cds_ends)
                    promoter_start = max(1, tss_anchor - args.downstream + 1) # Downstream biological region has lower coordinates
                    promoter_end = min(chrom_len, tss_anchor + args.upstream)    # Upstream biological region has higher coordinates
                    if promoter_start > promoter_end: promoter_start = promoter_end
                    # Split index calculation remains the same conceptually after reverse complement
                    split_index_in_extracted = max(0, promoter_end - tss_anchor)
                else:
                     # Should have been caught by find_tss, but double check
                    error_count += 1
                    print('stand errotr')
                    continue

                # --- Sequence Extraction (0-based slicing) ---
                try:
                    seq_0_start = promoter_start - 1
                    seq_0_end = promoter_end

                    if seq_0_start < 0 or seq_0_end > chrom_len or seq_0_start >= seq_0_end:
                        raise IndexError(f"Invalid slice coords [{seq_0_start}:{seq_0_end}] for chrom len {chrom_len}")

                    promoter_seq_slice = genome_seq_record.seq[seq_0_start:seq_0_end]
                    extracted_seq = str(promoter_seq_slice)

                    if not extracted_seq:
                        error_count += 1
                        continue

                    # --- Handle Strand and Case ---
                    if strand == '-':
                        extracted_seq = str(Seq(extracted_seq).reverse_complement())

                    len_extracted = len(extracted_seq)
                    if split_index_in_extracted >= len_extracted:
                         promoter_seq_final_case = extracted_seq.lower()
                    elif split_index_in_extracted <= 0:
                         promoter_seq_final_case = extracted_seq.upper()
                    else:
                         upstream_part = extracted_seq[:split_index_in_extracted].upper()
                         downstream_part = extracted_seq[split_index_in_extracted:].lower()
                         promoter_seq_final_case = upstream_part + downstream_part

                    # --- Write to single file ---
                    # Clean up gene_id for FASTA header (replace spaces, etc.) - Optional
                    safe_gene_id = re.sub(r'\s+', '_', str(gene_id)) # Basic cleaning
                    outfile.write(f">{safe_gene_id} promoter_cds_anchor {chrom}:{promoter_start}-{promoter_end}({strand}) anchor:{tss_anchor}\n")
                    for i in range(0, len(promoter_seq_final_case), 60):
                        outfile.write(promoter_seq_final_case[i:i+60] + "\n")
                    success_count += 1

                except IndexError as e:
                    print(f"Error slicing sequence for gene '{gene_id}' ({chrom}:{promoter_start}-{promoter_end}): {e}. Skipping.", file=sys.stderr)
                    error_count += 1
                except Exception as e:
                    print(f"Unexpected error processing promoter for '{gene_id}': {type(e).__name__} - {e}", file=sys.stderr)
                    error_count += 1
                # --- End Core Extraction Logic ---

    except IOError as e:
        print(f"Error: Could not write to output file '{args.output}': {e}", file=sys.stderr)
        # No good way to report counts accurately if file writing fails mid-way

    print(f"\nProcessed {gff_gene_groups_processed} gene groups from GFF.")
    print(f"Successfully wrote {success_count} promoter sequences to: {args.output}")
    if error_count > 0:
         print(f"Encountered {error_count} errors/warnings (genes skipped).")

if __name__ == '__main__':
    main()