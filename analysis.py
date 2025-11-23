# ======================================================
# 🔬 E. coli Genome Structure Analyzer (GBFF version)
#     (Matches Project Requirements Exactly)
# ======================================================

from Bio import SeqIO
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# =========================================
# 1) โหลดไฟล์ GBFF
# =========================================
gbff_file = "genomic.gbff"  # ใส่ไฟล์ .gbff ของคุณ
record = SeqIO.read(gbff_file, "genbank")

genome = str(record.seq).upper()
genome_len = len(genome)

print("Genome ID:", record.id)
print("Genome length:", genome_len, "bp\n")


# =========================================
# 2) ดึง Coding Regions (จาก Feature CDS)
# =========================================
cds_regions = []
for feature in record.features:
    if feature.type == "CDS":
        start = int(feature.location.start)
        end   = int(feature.location.end)
        cds_regions.append((start, end))

cds_regions.sort()

coding_len = sum(end - start for start, end in cds_regions)
coding_pct = coding_len / genome_len * 100
nc_pct = 100 - coding_pct

print(f"Coding DNA: {coding_pct:.2f}%")
print(f"Non-coding DNA: {nc_pct:.2f}%\n")


# =========================================
# 3) หา Intergenic Regions (ncDNA)
# =========================================
intergenic = []
prev_end = 0

for start, end in cds_regions:
    if start > prev_end:
        intergenic.append(genome[prev_end:start])
    prev_end = end

if prev_end < genome_len:
    intergenic.append(genome[prev_end:genome_len])

intergenic_lengths = [len(x) for x in intergenic]

print("Intergenic regions:", len(intergenic))
print("Average intergenic length:", np.mean(intergenic_lengths), "bp\n")

# Plot intergenic distribution
plt.hist(intergenic_lengths, bins=40, color="#7ec0ff", edgecolor="black")
plt.title("Distribution of Intergenic Region Lengths")
plt.xlabel("Length (bp)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


# =========================================
# 4) GC-content : whole, coding, ncDNA
# =========================================
def GC(seq):
    return (seq.count("G") + seq.count("C")) / len(seq) * 100 if len(seq)>0 else 0

gc_whole  = GC(genome)
gc_coding = np.mean([GC(genome[s:e]) for s, e in cds_regions])
gc_nc     = np.mean([GC(i) for i in intergenic if len(i) > 0])

print(f"GC whole genome : {gc_whole:.2f}%")
print(f"GC coding DNA   : {gc_coding:.2f}%")
print(f"GC ncDNA        : {gc_nc:.2f}%\n")

# Plot GC boxplot
plt.boxplot([ [GC(genome[s:e]) for s,e in cds_regions],
              [GC(i) for i in intergenic if len(i)>0] ],
            tick_labels=["Coding", "Non-coding"])
plt.ylabel("GC%")
plt.title("GC-content Comparison")
plt.tight_layout()
plt.show()


# =========================================
# 5) Sliding-window GC-content
# =========================================
window = 1000
positions = []
gc_values = []

for i in range(0, genome_len - window, window):
    win = genome[i:i+window]
    gc_values.append(GC(win))
    positions.append(i)

plt.plot(positions, gc_values, color="green")
plt.title("Sliding-window GC% (1000 bp)")
plt.xlabel("Genome Position (bp)")
plt.ylabel("GC%")
plt.tight_layout()
plt.show()


# =========================================
# 6) Dinucleotide Frequency
# =========================================
dinucs = [a+b for a in "ATCG" for b in "ATCG"]
counts = {d: genome.count(d) for d in dinucs}

print("Top dinucleotides:")
print(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5], "\n")

# Heatmap matrix
matrix = np.array([[counts[a+b] for b in "ATCG"] for a in "ATCG"])

plt.imshow(matrix, cmap="viridis")
plt.colorbar()
plt.xticks(range(4), list("ATCG"))
plt.yticks(range(4), list("ATCG"))
plt.title("Dinucleotide Frequency Heatmap")
plt.tight_layout()
plt.show()
