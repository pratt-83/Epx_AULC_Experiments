#!/usr/bin/env python3

# About
# Author: Nicholas Pratt
# This script samples fixed-size genomic windows from bacterial FASTA files,
# uses matching GFF annotations to assign each window to a biological bucket,
# and writes a metadata table for later dataset construction.

# Imports
from pathlib import Path

import random

import csv

# Settings
RAW_DIR = Path("data/raw")

OUT_DIR = Path("data/processed/microbial_annotation_512bp_v1")

WINDOW_BP = 512

N_CANDIDATE_WINDOWS_PER_GENOME = 20000

MAX_N_FRAC = 0.05

SEED = 1

OUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)

# Annotation types to use from the GFF files
RNA_TYPES = {

	"tRNA",
	"rRNA",
	"ncRNA",
	"scRNA",
	"tmRNA",
	"RNase_P_RNA",
	"antisense_RNA",
	"pseudogenic_tRNA",

}

FEATURE_TYPES_TO_TRACK = {"CDS", "pseudogene"} | RNA_TYPES


def read_fasta(fna_path):

	# Read a FASTA file into a dictionary: contig_name -> sequence.
	seqs = {}
	name = None
	parts = []
	with open(fna_path) as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			if line.startswith(">"):
				if name is not None:
					seqs[name] = "".join(parts).upper()
				name = line[1:].split()[0]
				parts = []
			else:
				parts.append(line)
	if name is not None:
		seqs[name] = "".join(parts).upper()
	return seqs


def read_gff_features(gff_path):

	# Read relevant GFF features.
	# GFF coordinates are 1-based closed, so convert them to 0-based half-open.
	features = {}
	with open(gff_path) as f:
		for line in f:
			if line.startswith("#"):
				continue
			fields = line.rstrip("\n").split("\t")
			if len(fields) < 9:
				continue
			contig = fields[0]
			feature_type = fields[2]
			start = int(fields[3]) - 1
			end = int(fields[4])
			if feature_type not in FEATURE_TYPES_TO_TRACK:
				continue
			if contig not in features:
				features[contig] = []
			features[contig].append((start, end, feature_type))
	for contig in features:
		features[contig].sort()
	return features


def get_overlap(start1, end1, start2, end2):

	# Return the number of overlapping bases between two intervals.
	return max(0, min(end1, end2) - max(start1, start2))


def merge_intervals(intervals):

	# Merge overlapping intervals so annotated bases are not double-counted.
	if not intervals:
		return []
	intervals = sorted(intervals)
	merged = [intervals[0]]
	for start, end in intervals[1:]:
		last_start, last_end = merged[-1]
		if start <= last_end:
			merged[-1] = (last_start, max(last_end, end))
		else:
			merged.append((start, end))
	return merged


def get_gc_frac(seq):

	# Return GC fraction for a DNA sequence.
	return (seq.count("G") + seq.count("C")) / len(seq)


def get_window_fractions(contig_features, window_start, window_end):

	# Calculate how much of the window overlaps each annotation type.
	bp_CDS = 0
	bp_tRNA = 0
	bp_rRNA = 0
	bp_all_RNA = 0
	bp_pseudogene = 0
	annotated_intervals = []
	feature_edges_inside = 0
	for feature_start, feature_end, feature_type in contig_features:
		if feature_end <= window_start:
			continue
		if feature_start >= window_end:
			break
		overlap_bp = get_overlap(window_start, window_end, feature_start, feature_end)
		if overlap_bp == 0:
			continue
		annotated_intervals.append((
			max(window_start, feature_start),
			min(window_end, feature_end)
		))
		if feature_type == "CDS":
			bp_CDS += overlap_bp
		elif feature_type == "tRNA":
			bp_tRNA += overlap_bp
			bp_all_RNA += overlap_bp
		elif feature_type == "rRNA":
			bp_rRNA += overlap_bp
			bp_all_RNA += overlap_bp
		elif feature_type in RNA_TYPES:
			bp_all_RNA += overlap_bp
		elif feature_type == "pseudogene":
			bp_pseudogene += overlap_bp
		if window_start < feature_start < window_end:
			feature_edges_inside += 1
		if window_start < feature_end < window_end:
			feature_edges_inside += 1
	merged_intervals = merge_intervals(annotated_intervals)
	annotated_bp = sum(end - start for start, end in merged_intervals)
	intergenic_bp = WINDOW_BP - annotated_bp
	return {
		"frac_CDS": bp_CDS / WINDOW_BP,
		"frac_intergenic": intergenic_bp / WINDOW_BP,
		"frac_tRNA": bp_tRNA / WINDOW_BP,
		"frac_rRNA": bp_rRNA / WINDOW_BP,
		"frac_all_RNA": bp_all_RNA / WINDOW_BP,
		"frac_pseudogene": bp_pseudogene / WINDOW_BP,
		"is_boundary": feature_edges_inside > 0,
	}


def assign_bucket(fracs):

	# Assign a biological bucket using simple threshold rules.
	if fracs["frac_CDS"] >= 0.80:
		return "CDS_high"
	if fracs["frac_intergenic"] >= 0.70:
		return "intergenic_high"
	if fracs["frac_all_RNA"] >= 0.10:
		return "RNA_enriched"
	if fracs["frac_pseudogene"] >= 0.30:
		return "pseudogene_enriched"
	max_frac = max(
		fracs["frac_CDS"],
		fracs["frac_intergenic"],
		fracs["frac_all_RNA"],
		fracs["frac_pseudogene"],
	)
	if fracs["is_boundary"] and max_frac < 0.70:
		return "boundary_mixed"
	return "mixed_other"


def main():

	rows = []
	for genome_dir in sorted(RAW_DIR.iterdir()):
		if not genome_dir.is_dir():
			continue
		fna_files = list(genome_dir.glob("*.fna"))
		gff_path = genome_dir / "genomic.gff"
		if len(fna_files) != 1 or not gff_path.exists():
			print(f"Skipping {genome_dir}")
			continue
		species = genome_dir.name
		print(f"Processing {species}")
		seqs = read_fasta(fna_files[0])
		features = read_gff_features(gff_path)
		contigs = [contig for contig in seqs if len(seqs[contig]) >= WINDOW_BP]
		for i in range(N_CANDIDATE_WINDOWS_PER_GENOME):
			contig = random.choice(contigs)
			seq = seqs[contig]
			start = random.randint(0, len(seq) - WINDOW_BP)
			end = start + WINDOW_BP
			window_seq = seq[start:end]
			n_frac = window_seq.count("N") / WINDOW_BP
			if n_frac > MAX_N_FRAC:
				continue
			fracs = get_window_fractions(features.get(contig, []), start, end)
			bucket = assign_bucket(fracs)
			rows.append({
				"window_id": f"{species}__{contig}__{start}__{end}",
				"species": species,
				"contig": contig,
				"start": start,
				"end": end,
				"sequence": window_seq,
				"gc_frac": get_gc_frac(window_seq),
				"n_frac": n_frac,
				**fracs,
				"bucket": bucket,
			})
	out_path = OUT_DIR / "windows_metadata.tsv"
	with open(out_path, "w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
		writer.writeheader()
		writer.writerows(rows)
	print(f"\nWrote: {out_path}")
	print("\nBucket counts:")
	counts = {}
	for row in rows:
		bucket = row["bucket"]
		counts[bucket] = counts.get(bucket, 0) + 1
	for bucket, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
		print(f"{bucket}\t{count}")



if __name__ == "__main__":

	main()
