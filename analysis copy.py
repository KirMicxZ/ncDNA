import streamlit as st
from Bio import SeqIO
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import io

# ============================================
# Genome Analyzer UI (Streamlit)
# ============================================

st.title("🔬 E. coli Genome Analyzer (GBFF)")
st.write("วิเคราะห์โครงสร้าง DNA / ncDNA / Intergenic / GC-content")

# File Uploader
uploaded = st.file_uploader("อัปโหลดไฟล์ .gbff", type=["gbff"])

# --- START Analysis Block: This block only runs when a file is uploaded ---
if uploaded:
    # 1. Read binary content from Streamlit uploader
    # 2. Decode the content to a standard Python string (text mode)
    gbff_text = uploaded.getvalue().decode("utf-8")

    # 3. Wrap the string in io.StringIO to create a file-like object (text stream)
    # This resolves the Bio.StreamModeError
    try:
        record = SeqIO.read(io.StringIO(gbff_text), "genbank")
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์: {e}")
        st.write("โปรดตรวจสอบว่าไฟล์ .gbff ที่อัปโหลดเป็นไฟล์ GenBank ที่สมบูรณ์และถูกต้อง")
        st.stop()
    
    # Extract Sequence and Genome Length
    seq = str(record.seq).upper()
    genome_len = len(seq)

    st.subheader("📌 ข้อมูลจีโนม")
    st.write(f"**ID:** {record.id}")
    st.write(f"**Length:** {genome_len:,} bp")

    # Extract CDS
    cds = []
    for f in record.features:
        # We only care about CDS features
        if f.type == "CDS":
            # Extract start and end positions (Biopython uses 0-based indexing)
            # .start and .end return objects, cast them to int
            cds.append((int(f.location.start), int(f.location.end)))

    # Sort CDS by position
    cds.sort()

    # Calculate Coding Length and Percentage
    coding_len = sum(e - s for s, e in cds)
    coding_pct = coding_len / genome_len * 100
    nc_pct = 100 - coding_pct

    st.metric("Coding DNA (%)", f"{coding_pct:.2f}%")
    st.metric("Non-coding DNA (%)", f"{nc_pct:.2f}%")

    # Intergenic (Non-coding regions between genes)
    intergenic = []
    prev = 0
    for s, e in cds:
        if s > prev:
            # The sequence from the end of the previous feature to the start of the current feature
            intergenic.append(seq[prev:s])
        prev = e
        
    # Handle the gap between the last feature and the start of the circular genome (if applicable)
    if prev < genome_len:
         intergenic.append(seq[prev:genome_len])

    # Intergenic histogram
    st.subheader("📏 การกระจายความยาวของ Intergenic Region")
    lengths = [len(i) for i in intergenic]
    lengths = [l for l in lengths if l > 0] # Filter out 0 length segments
    
    if lengths:
        fig, ax = plt.subplots()
        ax.hist(lengths, bins=min(50, len(lengths)//10), color="#7ec0ff", edgecolor='black')
        ax.set_title("Intergenic Length Distribution")
        ax.set_xlabel("Length (bp)")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
    else:
        st.info("ไม่พบ Intergenic Region ที่มีขนาด > 0 bp")

    # GC-content Calculation Function
    @st.cache_data
    def GC(x):
        """Calculates GC percentage of a sequence x."""
        return (x.count("G") + x.count("C")) / len(x) * 100 if len(x) > 0 else 0

    st.subheader("🧬 การเปรียบเทียบ GC-content")
    
    # GC-content for Coding and Non-coding regions
    gc_coding = [GC(seq[s:e]) for s, e in cds if (e - s) > 0]
    gc_nc = [GC(i) for i in intergenic if len(i) > 0]

    if gc_coding or gc_nc:
        fig2, ax2 = plt.subplots()
        # Ensure only non-empty data sets are plotted
        plot_data = []
        labels = []
        if gc_coding:
            plot_data.append(gc_coding)
            labels.append("Coding")
        if gc_nc:
            plot_data.append(gc_nc)
            labels.append("Non-coding")
            
        ax2.boxplot(plot_data, labels=labels, patch_artist=True, 
                    boxprops=dict(facecolor='#a3e635', color='#4d7c0f'),
                    medianprops=dict(color='black'))
        ax2.set_title("GC-content Comparison (Boxplot)")
        ax2.set_ylabel("GC %")
        st.pyplot(fig2)
    else:
        st.info("ไม่มีข้อมูล Coding หรือ Non-coding เพียงพอสำหรับสร้าง Boxplot GC-content")


    # Sliding window GC%
    st.subheader("📈 GC% ตลอดความยาวจีโนม (Sliding Window)")
    window = 1000
    pos = []
    vals = []
    
    # Calculate GC% in 1000 bp windows
    for i in range(0, genome_len, window):
        end = min(i + window, genome_len)
        window_seq = seq[i:end]
        if len(window_seq) == window: # Only plot full windows for smooth results
            pos.append(i + window/2) # Center of the window
            vals.append(GC(window_seq))

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(pos, vals, color="#065f46", linewidth=1.5)
    ax3.set_title(f"Sliding-window GC% ({window:,} bp)")
    ax3.set_xlabel("Position on Genome (bp)")
    ax3.set_ylabel("GC %")
    ax3.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig3)

    # Dinucleotide Heatmap
    st.subheader("🔠 Dinucleotide Frequency Heatmap")
    dinucs = [a+b for a in "ATCG" for b in "ATCG"]
    counts = {d: seq.count(d) for d in dinucs}
    # Create 4x4 matrix from counts
    mat = np.array([[counts[a+b] for b in "ATCG"] for a in "ATCG"])

    fig4, ax4 = plt.subplots()
    im = ax4.imshow(mat, cmap="coolwarm") # Use 'coolwarm' for better contrast
    
    # Add labels
    bases = list("ATCG")
    ax4.set_xticks(range(4))
    ax4.set_xticklabels(bases)
    ax4.set_yticks(range(4))
    ax4.set_yticklabels(bases)
    ax4.set_title("Dinucleotide Heatmap (Observed Count)")
    
    # Add color bar
    cbar = fig4.colorbar(im)
    cbar.set_label('Frequency Count')
    
    st.pyplot(fig4)

    st.success("✅ การวิเคราะห์จีโนมเสร็จสมบูรณ์!")

# --- END Analysis Block ---