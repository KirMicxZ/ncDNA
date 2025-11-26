import streamlit as st
from Bio import SeqIO
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import plotly.express as px
import re
import google.generativeai as genai # เพิ่ม Library สำหรับ AI

# ============================================
# 1. Page Configuration & UI Setup
# ============================================
st.set_page_config(page_title="Genome Analyzer", layout="wide", page_icon="🧬")

# Set Matplotlib to Dark Background style globally
plt.style.use('dark_background')

# Custom CSS for Dark Theme & Professional UI
st.markdown("""
<style>
    /* Main Background */
    .stApp { background-color: #262730; color: #FFFFFF; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #1E1E1E; }
    
    /* Headers */
    h1, h2, h3, .main-header { color: #FFFFFF !important; font-family: 'Helvetica', sans-serif; }
    .sub-header { color: #A3A3A3 !important; font-size: 1.1rem; }
    
    /* Metrics */
    [data-testid="stMetricValue"] { color: #4ADE80 !important; } /* Neon Green */
    [data-testid="stMetricLabel"] { color: #D1D5DB !important; }
    
    /* Table */
    [data-testid="stDataFrame"] { background-color: #262730; }
    
    /* Buttons */
    .stButton button { width: 100%; }
</style>
""", unsafe_allow_html=True)

# ============================================
# 2. Helper Functions (Logic)
# ============================================
@st.cache_data
def calculate_gc(sequence):
    """Calculates GC percentage."""
    if not sequence: return 0
    return (sequence.count("G") + sequence.count("C")) / len(sequence) * 100

def find_simple_repeats(seq, motif="AT", threshold=5):
    """Counts occurrences of a motif repeating > threshold times."""
    if not seq: return 0
    pattern = f"({motif}){{{threshold},}}"
    matches = [m.group(0) for m in re.finditer(pattern, seq)]
    return len(matches)

def process_genbank(file_content, filename):
    """Reads a GenBank file and extracts key metrics."""
    try:
        record = next(SeqIO.parse(io.StringIO(file_content), "genbank"))
    except Exception as e:
        return None, f"Error reading {filename}: {e}"

    seq = str(record.seq).upper()
    genome_len = len(seq)
    
    # Extract CDS
    cds_regions = []
    for f in record.features:
        if f.type == "CDS":
            cds_regions.append((int(f.location.start), int(f.location.end)))
    cds_regions.sort()

    # Calculate metrics
    coding_len = sum(e - s for s, e in cds_regions)
    coding_pct = (coding_len / genome_len) * 100
    nc_pct = 100 - coding_pct
    
    # Extract Intergenic Regions
    intergenic_seqs = []
    prev = 0
    for s, e in cds_regions:
        if s > prev:
            intergenic_seqs.append(seq[prev:s])
        prev = e
    if prev < genome_len:
        intergenic_seqs.append(seq[prev:genome_len])

    return {
        "id": record.id,
        "name": record.description,
        "filename": filename,
        "len": genome_len,
        "seq": seq,
        "cds_regions": cds_regions,
        "coding_pct": coding_pct,
        "nc_pct": nc_pct,
        "intergenic_seqs": intergenic_seqs,
        "gc_total": calculate_gc(seq)
    }, None

# --- AI Helper Function ---
def get_ai_response(api_key, prompt):
    """Connects to Gemini API to get analysis."""
    if not api_key:
        return "กรุณาระบุ Google API Key ในแถบเมนูด้านซ้ายเพื่อใช้งาน AI"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        with st.spinner('AI กำลังวิเคราะห์ข้อมูล...'):
            response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"เกิดข้อผิดพลาดในการเชื่อมต่อ AI: {str(e)}"

# ============================================
# 3. Sidebar: Inputs & Instructions
# ============================================
with st.sidebar:
    st.title("Genome Analyzer")
    st.markdown("เครื่องมือวิเคราะห์โครงสร้างจีโนม")
    
    # AI Settings Section
    st.markdown("---")
    st.subheader("AI Configuration")
    api_key = st.text_input("Google API Key", type="password", help="ใส่ API Key จาก Google AI Studio เพื่อเปิดใช้งานฟีเจอร์วิเคราะห์อัตโนมัติ")
    
    st.markdown("---")
    st.subheader("📂 อัปโหลดไฟล์")
    uploaded_files = st.file_uploader(
        "รองรับไฟล์ .gbff", 
        type=["gbff"], 
        accept_multiple_files=True
    )

    st.markdown("---")
    with st.expander("วิธีการใช้งาน"):
        st.markdown("""
        1. **เตรียมไฟล์:** ไฟล์จีโนมสกุล `.gbff`
        2. **API Key:** ใส่ Google API Key หากต้องการคำแนะนำจาก AI
        3. **โหมดไฟล์เดียว:** แสดงกราฟเชิงลึก, Heatmap และ AI Analysis
        4. **โหมดเปรียบเทียบ:** ดูตาราง Ranking และให้ AI สรุปแนวโน้ม
        """)
    
    st.caption("Developed for High School Science Project")

# ============================================
# 4. Main Analysis Area
# ============================================

st.markdown('<h1 class="main-header">Genome Analysis Dashboard</h1>', unsafe_allow_html=True)

if not uploaded_files:
    st.info("⬅️ กรุณาอัปโหลดไฟล์ .gbff ที่แถบเมนูด้านซ้ายเพื่อเริ่มต้น")
    
    cols = st.columns(3)
    with cols[0]:
        st.markdown("### Deep Analysis")
        st.write("วิเคราะห์โครงสร้างยีนและ Junk DNA เชิงลึก")
    with cols[1]:
        st.markdown("### Comparison")
        st.write("เปรียบเทียบสิ่งมีชีวิตหลายสายพันธุ์ (Interactive)")
    with cols[2]:
        st.markdown("### Data Export")
        st.write("ดาวน์โหลดลำดับเบส Junk DNA ไปศึกษาต่อ")

else:
    # --- PROCESSING FILES ---
    results = []
    errors = []
    
    with st.spinner('กำลังประมวลผล DNA...'):
        for uploaded_file in uploaded_files:
            content = uploaded_file.getvalue().decode("utf-8")
            data, err = process_genbank(content, uploaded_file.name)
            if data:
                results.append(data)
            else:
                errors.append(err)

    if errors:
        for e in errors:
            st.error(e)

    # ============================================
    # MODE A: Single File (Deep Dive)
    # ============================================
    if len(results) == 1:
        data = results[0]
        st.markdown(f"### ผลการวิเคราะห์: {data['name']}")
        st.caption(f"File: {data['filename']} | ID: {data['id']}")
        
        # 1. Key Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Genome Length", f"{data['len']:,} bp")
        m2.metric("GC Content", f"{data['gc_total']:.2f}%")
        m3.metric("Coding DNA (CDS)", f"{data['coding_pct']:.2f}%")
        m4.metric("Junk/Non-coding", f"{data['nc_pct']:.2f}%")
        
        st.divider()

        # --- AI Analysis Section (New) ---
        st.subheader("AI Biological Analysis")
        ai_col1, ai_col2 = st.columns([1, 2])
        
        with ai_col1:
            st.markdown("สอบถาม AI เกี่ยวกับจีโนมนี้")
            user_question = st.text_input("พิมพ์คำถาม (หรือเว้นว่างเพื่อขอสรุปผล)", placeholder="เช่น สิ่งมีชีวิตนี้ทนความร้อนได้หรือไม่?")
            analyze_btn = st.button("Start Analysis")
        
        with ai_col2:
            if analyze_btn:
                # Prepare context
                context = f"""
                You are a Bioinformatics expert. Analyze this organism:
                Name: {data['name']}
                Genome Length: {data['len']} bp
                GC Content: {data['gc_total']:.2f}%
                Non-coding DNA: {data['nc_pct']:.2f}%
                """
                if user_question:
                    prompt = f"{context}\n\nUser Question: {user_question}\nAnswer in Thai language clearly."
                else:
                    prompt = f"{context}\n\nPlease provide a biological summary and analysis based on the GC content and Junk DNA percentage. What does this imply about the organism's complexity or environment? Answer in Thai language."
                
                response_text = get_ai_response(api_key, prompt)
                st.markdown(response_text)
            else:
                st.info("กดปุ่ม Start Analysis เพื่อเริ่มการวิเคราะห์ด้วย AI")
        
        st.divider()

        # 2. Charts Row 1
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("**1. การกระจายตัวของ Intergenic Length**")
            lengths = [len(i) for i in data['intergenic_seqs'] if len(i) > 0]
            if lengths:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(lengths, bins=50, color="#818cf8", edgecolor='#1f2937', alpha=0.9)
                ax.set_xlabel("Length (bp)")
                ax.set_ylabel("Frequency")
                ax.grid(axis='y', alpha=0.2, linestyle='--')
                st.pyplot(fig)
            else:
                st.warning("ไม่พบ Intergenic Regions")

        with c2:
            st.markdown("**2. GC-Content Comparison**")
            gc_coding = [calculate_gc(data['seq'][s:e]) for s, e in data['cds_regions']]
            gc_nc = [calculate_gc(s) for s in data['intergenic_seqs'] if len(s) > 0]
            
            if gc_coding and gc_nc:
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                bp = ax2.boxplot([gc_coding, gc_nc], labels=["Coding", "Non-coding"], patch_artist=True)
                for box in bp['boxes']:
                    box.set(color='#34d399', linewidth=2)
                    box.set(facecolor='#065f46')
                for median in bp['medians']:
                    median.set(color='white', linewidth=2)
                ax2.set_ylabel("GC %")
                ax2.grid(axis='y', alpha=0.2, linestyle='--')
                st.pyplot(fig2)

        st.divider()
        
        # 3. Sliding Window
        st.markdown("**3. GC% Variation (Sliding Window)**")
        window = 1000
        seq = data['seq']
        pos = []
        vals = []
        for i in range(0, len(seq), window):
            sub = seq[i:i+window]
            if len(sub) == window:
                pos.append(i)
                vals.append(calculate_gc(sub))
        
        if vals:
            st.area_chart(pd.DataFrame({'GC%': vals}, index=pos), color="#6366f1")

        # 4. Advanced Analysis Section
        st.markdown("---")
        st.subheader("Advanced Analysis: Repeats & Data")
        
        ac1, ac2 = st.columns(2)
        
        with ac1:
            st.markdown("#### Motif Search in Junk DNA")
            st.caption("ค้นหาจำนวนจุดที่พบรหัสพันธุกรรมซ้ำๆ ในส่วน Non-coding")
            
            sc1, sc2 = st.columns(2)
            with sc1:
                motif_input = st.text_input("Pattern (e.g. AT, G)", value="AT")
            with sc2:
                threshold_input = st.number_input("Min Repeats", min_value=3, value=5)
            
            # Search
            total_repeats = 0
            for s in data['intergenic_seqs']:
                total_repeats += find_simple_repeats(s, motif_input, threshold_input)
            
            st.metric(f"Found '{motif_input}' repeated >{threshold_input} times", f"{total_repeats:,} spots")

        with ac2:
            st.markdown("#### Export Data")
            st.caption("ดาวน์โหลดลำดับเบสส่วน Junk DNA (FASTA) เพื่อนำไปวิเคราะห์ต่อ")
            
            fasta_str = ""
            for i, seq_segment in enumerate(data['intergenic_seqs']):
                if len(seq_segment) > 0:
                    fasta_str += f">Intergenic_{i+1}\n{seq_segment}\n"
            
            st.download_button(
                label="Download Non-coding Sequences (.fasta)",
                data=fasta_str,
                file_name=f"{data['id']}_junk_dna.fasta",
                mime="text/plain"
            )

    # ============================================
    # MODE B: Multi-File (Comparison)
    # ============================================
    elif len(results) > 1:
        st.markdown(f"### เปรียบเทียบสิ่งมีชีวิต ({len(results)} ตัวอย่าง)")
        
        df = pd.DataFrame([
            {
                "Organism": r['name'].split(',')[0],
                "Length (bp)": r['len'],
                "Coding %": r['coding_pct'],
                "Non-coding %": r['nc_pct'],
                "GC %": r['gc_total']
            } for r in results
        ])

        # 1. Summary Table
        st.markdown("#### ตารางสรุปข้อมูล (Summary Table)")
        st.dataframe(df.style.highlight_max(axis=0, color='#1e40af'), use_container_width=True)

        # --- AI Comparative Report (New) ---
        st.markdown("---")
        st.subheader("AI Comparative Report")
        if st.button("Generate Comparative Analysis"):
            # Prepare data string for AI
            data_str = df.to_string()
            prompt = f"""
            You are a Bioinformatics expert. Analyze this comparative data table of multiple organisms:
            {data_str}
            
            Please provide a comparative analysis.
            1. Which organism has the highest complexity based on coding DNA?
            2. Identify any correlation between Genome Size and Non-coding DNA (Junk DNA).
            3. Comment on the GC content variations.
            Answer in Thai language.
            """
            response_text = get_ai_response(api_key, prompt)
            st.markdown(response_text)

        # 2. Interactive Charts
        st.markdown("---")
        st.markdown("#### ความสัมพันธ์: ขนาดจีโนม vs Junk DNA (Interactive)")
        st.caption("ℹ️ เอาเมาส์ชี้ที่จุดเพื่อดูชื่อสิ่งมีชีวิต / หมุนลูกกลิ้งเพื่อซูมเข้า-ออก")
        
        # Interactive Scatter Plot using Plotly
        fig = px.scatter(
            df, 
            x="Length (bp)", 
            y="Non-coding %",
            color="GC %",
            size="Length (bp)",
            hover_name="Organism",
            color_continuous_scale="Viridis",
            template="plotly_dark",
            title="Genome Size vs. Non-coding DNA %"
        )
        st.plotly_chart(fig, use_container_width=True)

        # 3. Static Comparison
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**เปรียบเทียบ Junk DNA %**")
            fig_bar, ax_bar = plt.subplots()
            df_sorted = df.sort_values("Non-coding %", ascending=True)
            ax_bar.barh(df_sorted["Organism"], df_sorted["Non-coding %"], color="#ac3632")
            ax_bar.set_xlabel("% Non-coding DNA")
            ax_bar.grid(axis='x', linestyle='--', alpha=0.3)
            st.pyplot(fig_bar)

        with c2:
            st.markdown("**เปรียบเทียบ Genome Size**")
            fig_bar2, ax_bar2 = plt.subplots()
            df_sorted_len = df.sort_values("Length (bp)", ascending=True)
            ax_bar2.barh(df_sorted_len["Organism"], df_sorted_len["Length (bp)"], color="#60a5fa") 
            ax_bar2.set_xlabel("Base pairs (bp)")
            ax_bar2.grid(axis='x', linestyle='--', alpha=0.3)
            st.pyplot(fig_bar2)
