import streamlit as st
from Bio import SeqIO, Entrez
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import plotly.express as px
import re
from sklearn.cluster import KMeans
from itertools import product

# ============================================
# 1. UI Setup
# ============================================
st.set_page_config(page_title="GBFF Genome Analyzer", layout="wide", page_icon="🧬")
plt.style.use('dark_background') # ธีมกราฟสีมืด

# CSS
st.markdown("""
<style>
    .stApp { background-color: #262730; color: #FFFFFF; }
    [data-testid="stSidebar"] { background-color: #1E1E1E; }
    h1, h2, h3 { color: #FFFFFF !important; font-family: 'Helvetica', sans-serif; }
    [data-testid="stMetricValue"] { color: #4ADE80 !important; }
    .stButton button { width: 100%; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# Session State 
if 'fetched_data' not in st.session_state:
    st.session_state['fetched_data'] = []

# ============================================
# 2. Main Func
# ============================================
@st.cache_data
def calculate_gc(sequence):
    """คำนวณเปอร์เซ็นต์ GC Content"""
    if not sequence: return 0
    return (sequence.count("G") + sequence.count("C")) / len(sequence) * 100

def get_gene_architecture(strand1, strand2):
    """ระบุรูปแบบการเรียงตัวของยีน (Divergent/Convergent/Unidirectional)"""
    if strand1 == -1 and strand2 == 1: return "Divergent (<- ->)"
    elif strand1 == 1 and strand2 == -1: return "Convergent (-> <-)"
    else: return "Unidirectional (-> ->)"

def get_kmer_features(seq, k=2):
    """แปลงลำดับเบสเป็นตัวเลข (Vector) สำหรับ AI (นับความถี่ Dinucleotide)"""
    if not seq: return [0]*16
    kmers = [''.join(p) for p in product('ATCG', repeat=k)]
    counts = {kmer: 0 for kmer in kmers}
    
    seq = seq.upper()
    total = len(seq) - k + 1
    if total <= 0: return [0]*16
    
    for i in range(total):
        kmer = seq[i:i+k]
        if kmer in counts: counts[kmer] += 1
            
    return [counts[k]/total for k in kmers] 

def process_genbank(file_content, filename):
    """อ่านไฟล์ GenBank และถอดรหัสข้อมูลทั้งหมด (รองรับหลายโครโมโซม)"""
    try:
        records = list(SeqIO.parse(io.StringIO(file_content), "genbank"))
        if not records: return None, "Error: ไม่พบข้อมูล หรือไฟล์ผิดรูปแบบ"

        # ตัวแปรสำหรับเก็บผลรวม
        total_len = 0
        total_coding = 0
        total_gc_bases = 0
        
        all_intergenic = []
        all_cds_seqs = []
        chrom_stats = []
        
        arch_counts = {"Divergent (<- ->)": 0, "Convergent (-> <-)": 0, "Unidirectional (-> ->)": 0}
        combined_seq = "" 

        for record in records:
            seq = str(record.seq).upper()
            if not seq or set(seq) == {'?'}: continue 
            slen = len(seq)
            total_len += slen
            combined_seq += seq
            total_gc_bases += (seq.count("G") + seq.count("C"))

            # ดึงข้อมูลยีน 
            cds_feats = sorted(
                [{'s': int(f.location.start), 'e': int(f.location.end), 'str': f.location.strand} 
                 for f in record.features if f.type == "CDS"], 
                key=lambda x: x['s']
            )

            # คำนวณสถิติรายโครโมโซม
            c_coding_len = sum(f['e'] - f['s'] for f in cds_feats)
            chrom_stats.append({
                "ID": record.id,
                "Length": slen,
                "Genes": len(cds_feats),
                "GC%": calculate_gc(seq),
                "Coding%": (c_coding_len/slen*100) if slen else 0,
                "Non-coding%": 100 - ((c_coding_len/slen*100) if slen else 0)
            })
            total_coding += c_coding_len

            # เก็บ Sequence ของ Coding
            for f in cds_feats: all_cds_seqs.append(seq[f['s']:f['e']])

            # หา Intergenic Regions (ช่องว่างระหว่างยีน) และวิเคราะห์ Architecture
            prev_e = 0
            prev_str = None
            for f in cds_feats:
                if f['s'] > prev_e:
                    gap = seq[prev_e:f['s']]
                    if len(gap) >= 3: all_intergenic.append(gap)
                    
                    if prev_str is not None:
                        arch = get_gene_architecture(prev_str, f['str'])
                        arch_counts[arch] += 1
                prev_e = f['e']
                prev_str = f['str']
            
            # เก็บส่วนท้ายโครโมโซม
            if prev_e < slen: 
                gap = seq[prev_e:slen]
                if len(gap) >= 3: all_intergenic.append(gap)

        if total_len == 0: return None, "Error: ไม่พบข้อมูลลำดับเบส"

        # ปรับชื่อให้แสดงผลถูกต้องกรณีมีหลายโครโมโซม
        display_name = records[0].description.split(',')[0]
        if len(records) > 1:
            display_name = re.sub(r'\s*chromosome\s+[IVX0-9]+', '', display_name, flags=re.IGNORECASE)
            display_name = f"{display_name} (Total {len(records)} Chromosomes)"

        # สรุปผลรวม
        return {
            "id": records[0].id,
            "name": display_name,
            "filename": filename,
            "len": total_len,
            "seq": combined_seq,
            "cds_seqs": all_cds_seqs,
            "intergenic_seqs": all_intergenic,
            "coding_pct": (total_coding / total_len * 100),
            "nc_pct": 100 - (total_coding / total_len * 100),
            "gc_total": (total_gc_bases / total_len * 100),
            "avg_intergenic": np.mean([len(s) for s in all_intergenic]) if all_intergenic else 0,
            "architecture": arch_counts,
            "chromosomes": chrom_stats
        }, None

    except Exception as e: return None, f"Error: {e}"

def fetch_ncbi(acc_id, email):
    """ดึงข้อมูลจาก NCBI"""
    Entrez.email = email
    with Entrez.efetch(db="nucleotide", id=acc_id, rettype="gbwithparts", retmode="text") as handle:
        return handle.read()

# ============================================
# 3. Frontend
# ============================================
with st.sidebar:
    st.title("GBFF Analyzer")
    st.markdown("---")
    st.subheader("🌐 ดึงข้อมูล NCBI")
    
    email = st.text_input("Email (Required)", placeholder="email@example.com")
    acc_id = st.text_input("Accession ID", placeholder="e.g. NC_000913")
    
    if st.button("📥 ดึงข้อมูล"):
        if not email or not acc_id:
            st.error("กรุณากรอกข้อมูลให้ครบ")
        elif any(d['id'] == acc_id for d in st.session_state['fetched_data']):
            st.warning("มีข้อมูลนี้อยู่แล้ว")
        else:
            with st.spinner(f"Downloading {acc_id}..."):
                try:
                    raw = fetch_ncbi(acc_id.strip(), email.strip())
                    data, err = process_genbank(raw, f"NCBI_{acc_id}.gb")
                    if data: 
                        st.session_state['fetched_data'].append(data)
                        st.success("สำเร็จ!")
                    else: st.error(err)
                except Exception as e: st.error(f"Failed: {e}")

    if st.session_state['fetched_data']:
        st.markdown(f"**รายการ ({len(st.session_state['fetched_data'])}):**")
        for i, d in enumerate(st.session_state['fetched_data']): st.text(f"{i+1}. {d['id']}")
        if st.button("🗑️ ล้างรายการ"): 
            st.session_state['fetched_data'] = []
            st.rerun()
            
    st.markdown("---")
    files = st.file_uploader("หรือ อัปโหลดไฟล์ .gbff", type=["gbff"], accept_multiple_files=True)

# Main Logic
st.markdown('<h1 class="main-header">โปรแกรมวิเคราะห์ไฟล์จีโนม (Genome Analyzer)</h1>', unsafe_allow_html=True)

results = st.session_state['fetched_data'].copy()
if files:
    for f in files:
        d, e = process_genbank(f.getvalue().decode("utf-8"), f.name)
        if d: results.append(d)
        else: st.error(e)

if not results:
    st.info("กรุณาเพิ่มข้อมูลจากแถบเมนูด้านซ้าย")
else:
    # --- MODE A: Single File ---
    if len(results) == 1:
        d = results[0]
        st.subheader(f"🦠 {d['name']} (ID: {d['id']})")
        
        # เพิ่มแท็บ Lite ML ในโหมดไฟล์เดียวด้วย
        tab1, tab2, tab3 = st.tabs(["ภาพรวม (Overview)", "รายละเอียด (Breakdown)", "Lite ML Analysis"])
        
        with tab1:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Genome Size", f"{d['len']:,} bp")
            c2.metric("GC Content", f"{d['gc_total']:.2f}%")
            c3.metric("Coding DNA", f"{d['coding_pct']:.2f}%")
            c4.metric("Non-coding DNA", f"{d['nc_pct']:.2f}%")
            st.markdown("---")
            
            # Graphs
            g1, g2 = st.columns(2)
            with g1:
                st.markdown("**การกระจายตัวความยาว ncDNA**")
                lens = [len(s) for s in d['intergenic_seqs']]
                if lens:
                    fig, ax = plt.subplots(figsize=(6,3))
                    ax.hist(lens, bins=50, color='#818cf8', edgecolor='#1f2937')
                    st.pyplot(fig)
            with g2:
                st.markdown("**เปรียบเทียบ GC Content**")
                # Sample 1000 sequences for boxplot speed optimization
                gc_c = [calculate_gc(s) for s in d['cds_seqs'][:1000]] 
                gc_n = [calculate_gc(s) for s in d['intergenic_seqs'][:1000]]
                if gc_c and gc_n:
                    fig, ax = plt.subplots(figsize=(6,3))
                    ax.boxplot([gc_c, gc_n], labels=['Coding', 'Non-coding'], patch_artist=True)
                    st.pyplot(fig)
            
            st.markdown("---")
            st.markdown("**รูปแบบสถาปัตยกรรมยีน (Gene Architecture)**")
            st.bar_chart(d['architecture'])
            
        with tab2:
            if d['chromosomes']:
                df_chrom = pd.DataFrame(d['chromosomes'])
                st.dataframe(df_chrom, use_container_width=True)
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**ขนาดโครโมโซม**")
                    st.plotly_chart(px.bar(df_chrom, x="ID", y="Length", color="GC%"), use_container_width=True)
                with c2:
                    st.markdown("**สัดส่วน ncDNA**")
                    st.plotly_chart(px.scatter(df_chrom, x="Length", y="Non-coding%", size="Genes", color="GC%"), use_container_width=True)
        
        with tab3:
            st.markdown("### Lite ML: K-Means Clustering (Internal Analysis)")
            st.write("อัลกอริทึมจะจัดกลุ่มลำดับเบส ncDNA ภายในจีโนมนี้ออกเป็นกลุ่มย่อยตามคุณสมบัติทางเคมี (Dinucleotide Pattern)")
            
            if st.button("Run Internal Clustering"):
                with st.spinner("Training..."):
                    vecs = []
                    # สุ่ม 2000 sequence เพื่อความเร็ว
                    seqs = d['intergenic_seqs'][:2000]
                    valid_seqs = [s for s in seqs if len(s) >= 10]
                    
                    if len(valid_seqs) > 10:
                        for s in valid_seqs:
                            vecs.append(get_kmer_features(s))
                        
                        # Clustering
                        n_clusters = 3
                        km = KMeans(n_clusters=n_clusters, random_state=42).fit(vecs)
                        
                        # Show Cluster Properties (e.g., Average GC)
                        cluster_gcs = {i: [] for i in range(n_clusters)}
                        for i, label in enumerate(km.labels_):
                            cluster_gcs[label].append(calculate_gc(valid_seqs[i]))
                        
                        avg_gcs = [np.mean(cluster_gcs[i]) for i in range(n_clusters)]
                        counts = [len(cluster_gcs[i]) for i in range(n_clusters)]
                        
                        res_df = pd.DataFrame({
                            "Cluster": [f"Cluster {i+1}" for i in range(n_clusters)],
                            "Count": counts,
                            "Avg GC%": avg_gcs
                        })
                        
                        c1, c2 = st.columns(2)
                        with c1:
                            st.caption("จำนวนสมาชิกในแต่ละกลุ่ม")
                            st.plotly_chart(px.bar(res_df, x="Cluster", y="Count", color="Avg GC%"), use_container_width=True)
                        with c2:
                            st.caption("ค่าเฉลี่ย GC ของแต่ละกลุ่ม")
                            st.plotly_chart(px.scatter(res_df, x="Cluster", y="Avg GC%", size="Count", color="Avg GC%"), use_container_width=True)
                            
                        st.info("💡 การจัดกลุ่มนี้ช่วยให้เห็นว่า ncDNA ในสิ่งมีชีวิตนี้มีความหลากหลายเพียงใด (เช่น มีกลุ่มที่ GC สูงและต่ำแยกกันชัดเจนหรือไม่)")
                    else:
                        st.warning("ข้อมูล ncDNA น้อยเกินไปสำหรับการวิเคราะห์")

    # --- MODE B: Comparison ---
    else:
        st.subheader(f" เปรียบเทียบ {len(results)} ตัวอย่าง")
        
        df = pd.DataFrame([{
            "Name": r['name'], 
            "Length": r['len'], 
            "Coding%": r['coding_pct'], 
            "Non-coding%": r['nc_pct'],
            "GC%": r['gc_total'],
            "Avg ncDNA Len": r['avg_intergenic'],
            "Divergent": r['architecture']['Divergent (<- ->)'],
            "Unidirectional": r['architecture']['Unidirectional (-> ->)'],
            "Convergent": r['architecture']['Convergent (-> <-)']
        } for r in results])
        
        st.dataframe(df.style.highlight_max(axis=0, color='#1e40af'), use_container_width=True)
        
        # แท็บที่ 4: Lite ML จะแสดงที่นี่เมื่อมีหลายไฟล์
        t1, t2, t3, t4 = st.tabs(["Overview", "Architecture", "GC Trends", "🤖 Lite ML"])
        
        with t1:
            st.plotly_chart(px.scatter(df, x="Length", y="Non-coding%", color="GC%", size="Length", hover_name="Name"), use_container_width=True)
        
        with t2:
            df_melt = df.melt(id_vars=["Name"], value_vars=["Divergent", "Unidirectional", "Convergent"])
            st.plotly_chart(px.bar(df_melt, x="Name", y="value", color="variable", barmode="group"), use_container_width=True)
            
        with t3:
            st.plotly_chart(px.bar(df, x="Name", y="GC%", color="GC%"), use_container_width=True)
            
        with t4:
            st.markdown("### AI Clustering (Cross-species Comparison)")
            st.write("เปรียบเทียบว่า ncDNA ของสิ่งมีชีวิตแต่ละชนิด มีลักษณะทางเคมีคล้ายคลึงกันในกลุ่มใดบ้าง")
            
            if st.button("Run Clustering"):
                with st.spinner("Training..."):
                    vecs, labels = [], []
                    for r in results:
                        # สุ่มตัวอย่างเพื่อความเร็ว (Max 1000 seqs per organism)
                        seqs = r['intergenic_seqs'][:1000]
                        for s in seqs:
                            if len(s) >= 10:
                                vecs.append(get_kmer_features(s))
                                labels.append(r['name'])
                    
                    if len(vecs) > 10:
                        km = KMeans(n_clusters=3).fit(vecs)
                        res_df = pd.DataFrame({"Organism": labels, "Cluster": km.labels_})
                        counts = res_df.groupby(["Organism", "Cluster"]).size().reset_index(name="Count")
                        st.plotly_chart(px.bar(counts, x="Organism", y="Count", color="Cluster", barmode="stack"), use_container_width=True)
                    else:
                        st.warning("ข้อมูลไม่เพียงพอ")
