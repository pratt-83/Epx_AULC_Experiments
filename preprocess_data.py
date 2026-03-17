#!/usr/bin/env python3
# preprocess_data.py
#
# PURPOSE
#	Build a minimal-but-defensible dataset for Q1 (controls) in an AULC epiplexity pilot.
#	Input: multiple microbial genomes as .fna FASTA files under data/raw/**.fna
#	Output: 6 text files (real/shuf/rand × train/val), one window per line, plus meta.tsv
#
# DESIGN CHOICES
#	- We sample fixed-length nucleotide windows (e.g., 2048 bp) from real genomes.
#	- Controls are generated PER-WINDOW:
#		(1) dinucleotide-preserving shuffle (composition preserved up to dinucleotide counts)
#		(2) GC-matched i.i.d. random (matches GC%, destroys higher-order structure)
#	- We split the sampled windows into train vs validation so the loss curve is measured on held-out data.
#
# NOTES
#	- It doesn't matter if a genome has 1 contig or multiple contigs; we treat each contig independently.
#	- We never sample across contig boundaries.
#	- We reject windows with too many Ns to avoid artifacts, and replace any remaining Ns before shuffling/randomizing.

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import ushuffle

# Allowed characters in raw FASTA after normalization.
VALID = set("ACGTN")
BASES = "ACGT"


def read_fasta_records(path: Path) -> List[Tuple[str, str]]:
	"""
	Read a multi-FASTA file and return a list of (contig_id, sequence) tuples.

	- contig_id is taken as the first token after '>' (up to the first whitespace).
	- sequence is concatenated across lines, uppercased, and any non-ACGTN characters are replaced with 'N'.

	This function does NOT do any window sampling; it only parses the FASTA content into memory.
	"""
	records: List[Tuple[str, str]] = []

	contig_id = None
	seq_parts: List[str] = []

	def flush_current_record():
		"""
		Helper closure:
		When we hit a new '>' header or reach EOF, finalize the previous contig by joining its sequence.
		"""
		nonlocal contig_id, seq_parts
		if contig_id is None:
			return

		seq = "".join(seq_parts).upper()
		# Normalize any non-ACGTN to 'N' so downstream logic doesn't crash on ambiguous bases.
		seq = "".join([c if c in VALID else "N" for c in seq])

		records.append((contig_id, seq))
		contig_id = None
		seq_parts = []

	with path.open() as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			if line.startswith(">"):
				# Starting a new contig: flush previous one first.
				flush_current_record()
				contig_id = line[1:].split()[0]
				seq_parts = []
			else:
				seq_parts.append(line)

	# End of file: flush the final contig.
	flush_current_record()

	return records


def gc_fraction(seq: str) -> float:
	"""
	Compute GC fraction ignoring 'N' bases.

	GC = (count(G)+count(C)) / (count(A)+count(C)+count(G)+count(T))

	If the denominator is 0 (e.g., sequence is all N), return 0.0.
	"""
	g = seq.count("G") + seq.count("C")
	a = seq.count("A") + seq.count("T")
	den = g + a
	return (g / den) if den > 0 else 0.0


def n_fraction(seq: str) -> float:
	"""
	Fraction of Ns in a window.
	Used to reject low-quality windows (many assemblies use N to mark unknown bases).
	"""
	return seq.count("N") / len(seq) if seq else 1.0


def weighted_choice_index(weights: List[int], rng: random.Random) -> int:
	"""
	Sample an index i with probability proportional to weights[i].

	We use this to select contigs proportional to how many windows they can contribute:
	available = len(contig) - window_bp

	This avoids over-sampling small contigs.
	"""
	total = sum(weights)
	r = rng.randrange(total)
	acc = 0
	for i, w in enumerate(weights):
		acc += w
		if r < acc:
			return i
	return len(weights) - 1


def replace_Ns(seq: str, rng: random.Random) -> str:
	"""
	Replace any 'N' with a random A/C/G/T.

	Why:
	- ushuffle treats characters as symbols; if N is present, it will preserve N as a symbol too.
	- For clean composition controls, we want only A/C/G/T in shuffled and random controls.

	We still KEEP Ns in the 'real' windows (they reflect assembly uncertainty), but we reject windows with too many Ns.
	For the control versions, we remove leftover Ns to avoid artifacts.
	"""
	if "N" not in seq:
		return seq
	return "".join([c if c != "N" else rng.choice(BASES) for c in seq])


def dinuc_shuffle(seq: str, rng: random.Random) -> str:
	"""
	Dinucleotide-preserving shuffle using the ushuffle extension.

	Important detail:
	- This build of ushuffle expects BYTES, not str.
	- let_size=2 => preserve dinucleotide counts.

	We also ensure there are no Ns before shuffling.
	"""
	seq = replace_Ns(seq, rng)
	shuffled_bytes = ushuffle.shuffle(seq.encode("ascii"), 2)  # returns bytes
	return shuffled_bytes.decode("ascii")


def gc_matched_random(seq: str, rng: random.Random) -> str:
	"""
	Generate an i.i.d. random DNA sequence of the same length as seq with matched GC%.

	We compute GC% from the (N-replaced) window, then sample each base independently:
	P(G)=P(C)=GC/2 ; P(A)=P(T)=(1-GC)/2

	This control destroys higher-order structure while matching base composition.
	"""
	seq = replace_Ns(seq, rng)
	gc = gc_fraction(seq)

	p_g = gc / 2.0
	p_c = gc / 2.0
	p_a = (1.0 - gc) / 2.0
	p_t = (1.0 - gc) / 2.0

	choices = ["A", "C", "G", "T"]
	weights = [p_a, p_c, p_g, p_t]

	return "".join(rng.choices(choices, weights=weights, k=len(seq)))


def write_windows(path: Path, windows: List[str]):
	"""
	Write a list of windows to a text file, one sequence per line.
	"""
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text("\n".join(windows) + "\n")


def main():
	# ----------------------------
	# Parse arguments
	# ----------------------------
	ap = argparse.ArgumentParser(description="Build real/shuf/rand window datasets for AULC experiments.")
	ap.add_argument("--raw_dir", default="data/raw", help="Directory containing subfolders with .fna files")
	ap.add_argument("--out_dir", default="data/processed", help="Output directory for processed datasets")
	ap.add_argument("--window_bp", type=int, default=2048, help="Window length in base pairs (bp)")
	ap.add_argument("--n_windows", type=int, default=15000, help="Number of REAL windows to sample")
	ap.add_argument("--val_frac", type=float, default=0.1, help="Fraction of windows reserved for validation")
	ap.add_argument("--max_n_frac", type=float, default=0.05, help="Reject windows with N_fraction > this")
	ap.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")
	args = ap.parse_args()

	rng = random.Random(args.seed)

	raw_dir = Path(args.raw_dir)
	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	# ----------------------------
	# 1) Discover genome FASTA files
	# ----------------------------
	fna_files = sorted(list(raw_dir.rglob("*.fna")))
	if not fna_files:
		raise SystemExit(f"No .fna files found under {raw_dir}")

	# ----------------------------
	# 2) Parse FASTAs into contig records
	#	Each contig is treated independently.
	# ----------------------------
	# We'll store contigs as simple dicts for clarity.
	contigs = []  # each item: {"genome_id": str, "contig_id": str, "seq": str}
	for fp in fna_files:
		# Use folder name as a stable genome identifier (matches your current layout).
		genome_id = fp.parent.name

		records = read_fasta_records(fp)
		for contig_id, seq in records:
			# Only keep contigs that can yield at least one window.
			if len(seq) >= args.window_bp + 1:
				contigs.append({"genome_id": genome_id, "contig_id": contig_id, "seq": seq})

	if not contigs:
		raise SystemExit("No contigs long enough for the selected window size.")

	# ----------------------------
	# 3) Build contig sampling weights
	#	Weight ∝ number of possible start positions.
	# ----------------------------
	weights = [len(c["seq"]) - args.window_bp for c in contigs]

	# ----------------------------
	# 4) Sample REAL windows
	#	We sample within-contig only; no boundary crossing.
	#	We reject windows with too many Ns.
	# ----------------------------
	real_windows: List[str] = []
	meta_rows: List[Tuple] = []

	reject_due_to_N = 0
	total_attempts = 0

	while len(real_windows) < args.n_windows:
		total_attempts += 1

		# Choose a contig proportional to its "available" length.
		ci = weighted_choice_index(weights, rng)
		c = contigs[ci]
		seq = c["seq"]

		available = len(seq) - args.window_bp
		start = rng.randrange(available)
		end = start + args.window_bp

		w = seq[start:end]  # raw window (may contain Ns)

		nf = n_fraction(w)
		if nf > args.max_n_frac:
			reject_due_to_N += 1
			continue

		gc = gc_fraction(w)
		# If the window is all Ns, gc_fraction returns 0.0 and denominator is 0; reject it.
		if (w.count("A") + w.count("C") + w.count("G") + w.count("T")) == 0:
			reject_due_to_N += 1
			continue

		real_windows.append(w)
		meta_rows.append((len(real_windows) - 1, c["genome_id"], c["contig_id"], start, end, f"{gc:.4f}", f"{nf:.4f}"))

	# ----------------------------
	# 5) Generate controls PER WINDOW
	#	shuf: dinucleotide-preserving shuffle
	#	rand: GC-matched i.i.d. random
	# ----------------------------
	shuf_windows: List[str] = []
	rand_windows: List[str] = []

	for w in real_windows:
		# Controls use a version with Ns replaced so we don't preserve N as a symbol.
		w_clean = replace_Ns(w, rng)
		shuf_windows.append(dinuc_shuffle(w_clean, rng))
		rand_windows.append(gc_matched_random(w_clean, rng))

	# ----------------------------
	# 6) Train/validation split
	#	Why: the AULC curve should be measured on held-out data (validation)
	#	to avoid confusing memorization with genuine learnability.
	# ----------------------------
	N = len(real_windows)
	indices = list(range(N))
	rng.shuffle(indices)

	n_val = int(N * args.val_frac)
	val_idx = set(indices[:n_val])
	train_idx = set(indices[n_val:])

	real_train = [real_windows[i] for i in range(N) if i in train_idx]
	real_val = [real_windows[i] for i in range(N) if i in val_idx]

	shuf_train = [shuf_windows[i] for i in range(N) if i in train_idx]
	shuf_val = [shuf_windows[i] for i in range(N) if i in val_idx]

	rand_train = [rand_windows[i] for i in range(N) if i in train_idx]
	rand_val = [rand_windows[i] for i in range(N) if i in val_idx]

	# ----------------------------
	# 7) Write outputs
	# ----------------------------
	write_windows(out_dir / "real_train.txt", real_train)
	write_windows(out_dir / "real_val.txt", real_val)

	write_windows(out_dir / "shuf_train.txt", shuf_train)
	write_windows(out_dir / "shuf_val.txt", shuf_val)

	write_windows(out_dir / "rand_train.txt", rand_train)
	write_windows(out_dir / "rand_val.txt", rand_val)

	# Metadata for later debugging/analysis (optional but helpful).
	meta_path = out_dir / "meta.tsv"
	with meta_path.open("w") as f:
		f.write("idx\tgenome_id\tcontig_id\tstart\tend\tgc\tn_frac\n")
		for row in meta_rows:
			f.write("\t".join(map(str, row)) + "\n")

	# ----------------------------
	# Print summary (sanity info)
	# ----------------------------
	print(f"Found {len(fna_files)} FASTA (.fna) files under {raw_dir}")
	print(f"Parsed {len(contigs)} contigs eligible for window_bp={args.window_bp}")
	print(f"Sampled {len(real_windows)} real windows; rejected {reject_due_to_N} for N_frac>{args.max_n_frac} (attempts={total_attempts})")
	print(f"Wrote processed datasets to: {out_dir}")
	print(f"Wrote metadata to: {meta_path}")


if __name__ == "__main__":
	main()

