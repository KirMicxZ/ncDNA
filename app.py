import streamlit as st
from Bio import SeqIO, Entrez  
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import plotly.express as px
import plotly.graph_objects as go  
import re

# ============================================
# 1. Page Configuration & UI Setup
# ============================================
st.set_page_config(page_title="Genome Analyzer", layout="wide", page_icon="🧬")

plt.style.use('dark_background')

st.markdown("""
<style>
    .stApp { background-color: #262730; color: #FFFFFF; }
    [data-testid="stSidebar"] { background-color: #1E1E1E; }
    h1, h2, h3, .main-header { color: #FFFFFF !important; font-family: 'Helvetica', sans-serif; }
    .sub-header { color: #A3A3A3 !important; font-size: 1.1rem; }
    [data-testid="stMetricValue"] { color: #4ADE80 !important; } 
    [data-testid="stMetricLabel"] { color: #D1D5DB !important; }
    [data-testid="stDataFrame"] { background-color: #262730; }
    .stButton button { width: 100%; }
</style>
""", unsafe_allow_html=True)

# สร้าง Session State สำหรับเก็บข้อมูลจาก NCBI
if 'ncbi_cache' not in st.session_state:
    st.session_state['ncbi_cache'] = []

# ============================================
# 2. Helper Functions (Logic)
# ============================================
def fetch_ncbi(acc_id, email):
    """ดึงข้อมูล GenBank จาก NCBI ผ่าน API"""
    Entrez.email = email
    with Entrez.efetch(db="nucleotide", id=acc_id, rettype="gbwithparts", retmode="text") as handle:
        return handle.read()

@st.cache_data
def calculate_gc(sequence):
    if not sequence: return 0
    return (sequence.count("G") + sequence.count("C")) / len(sequence) * 100

def find_simple_repeats(seq, motif="AT", threshold=5):
    if not seq: return 0
    pattern = f"({motif}){{{threshold},}}"
    matches = [m.group(0) for m in re.finditer(pattern, seq)]
    return len(matches)

def process_genbank(file_content, filename):
    """Reads a GenBank file and extracts key metrics for ALL chromosomes."""
    try:
        # ใช้ list() เพื่ออ่านทุกโครโมโซมในไฟล์
        records = list(SeqIO.parse(io.StringIO(file_content), "genbank"))
        if not records: return None, "ไม่พบข้อมูลในไฟล์"
    except Exception as e:
        return None, f"Error reading {filename}: {e}"

    chromosomes_data = {}
    total_len = 0
    total_coding_len = 0
    total_gc = 0
    
    # วนลูปประมวลผลทีละโครโมโซม
    for record in records:
        seq = str(record.seq).upper()
        slen = len(seq)
        total_len += slen
        total_gc += (seq.count("G") + seq.count("C"))
        
        # Extract CDS
        cds_regions = []
        for f in record.features:
            if f.type == "CDS":
                cds_regions.append((int(f.location.start), int(f.location.end)))
        cds_regions.sort()

        # Calculate metrics for this specific chromosome
        coding_len = sum(e - s for s, e in cds_regions)
        total_coding_len += coding_len
        
        coding_pct = (coding_len / slen) * 100 if slen > 0 else 0
        nc_pct = 100 - coding_pct
        
        # Extract Intergenic Regions
        intergenic_seqs = []
        prev = 0
        for s, e in cds_regions:
            if s > prev:
                intergenic_seqs.append(seq[prev:s])
            prev = e
        if prev < slen:
            intergenic_seqs.append(seq[prev:slen])

        # เก็บแยกตาม ID โครโมโซม
        chromosomes_data[record.id] = {
            "id": record.id,
            "desc": record.description,
            "len": slen,
            "seq": seq,
            "cds_regions": cds_regions,
            "coding_pct": coding_pct,
            "nc_pct": nc_pct,
            "intergenic_seqs": intergenic_seqs,
            "gc_total": calculate_gc(seq)
        }

    # สรุปภาพรวมสำหรับโหมดเปรียบเทียบหลายไฟล์
    overall_coding_pct = (total_coding_len / total_len) * 100 if total_len > 0 else 0
    
    return {
        "name": records[0].description.split(',')[0],
        "filename": filename,
        "total_chromosomes": len(records),
        "chromosomes": chromosomes_data,
        "len": total_len,
        "coding_pct": overall_coding_pct,
        "nc_pct": 100 - overall_coding_pct,
        "gc_total": (total_gc / total_len) * 100 if total_len > 0 else 0
    }, None

# ============================================
# 3. Sidebar: Inputs & Instructions
# ============================================
with st.sidebar:
    st.title("Genome Analyzer")
    st.markdown("เครื่องมือวิเคราะห์โครงสร้างจีโนม")
    
    # --- เพิ่มระบบดาวน์โหลดจาก NCBI ---
    st.markdown("---")
    st.subheader("🌐 ดึงข้อมูลจาก NCBI")
    ncbi_email = st.text_input("Email (จำเป็น)", placeholder="email@example.com")
    ncbi_id = st.text_input("RefSeq ID", placeholder="เช่น NC_000913")
    
    if st.button("📥 ดาวน์โหลดจาก NCBI"):
        if not ncbi_email or not ncbi_id:
            st.error("กรุณากรอก Email และ RefSeq ID ให้ครบถ้วน")
        else:
            with st.spinner(f"กำลังดาวน์โหลด {ncbi_id} (อาจใช้เวลาสักครู่)..."):
                try:
                    raw_data = fetch_ncbi(ncbi_id.strip(), ncbi_email.strip())
                    if not any(item['id'] == ncbi_id for item in st.session_state['ncbi_cache']):
                        st.session_state['ncbi_cache'].append({
                            "id": ncbi_id.strip(),
                            "filename": f"NCBI_{ncbi_id.strip()}.gbff",
                            "content": raw_data
                        })
                    st.success(f"ดาวน์โหลด {ncbi_id} สำเร็จ!")
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาดในการโหลด: {e}")
                    
    if st.session_state['ncbi_cache']:
        st.markdown(f"*(มีข้อมูลจาก NCBI จำนวน {len(st.session_state['ncbi_cache'])} รายการ)*")
        if st.button("🗑️ ล้างข้อมูล NCBI"):
            st.session_state['ncbi_cache'] = []
            st.rerun()
            
    # --- ระบบอัปโหลดไฟล์เดิม ---
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
        1. **เตรียมไฟล์:** ไฟล์จีโนมสกุล `.gbff` หรือใช้ RefSeq ID ดาวน์โหลดผ่านเน็ต
        2. **โหมดไฟล์เดียว:** แสดงกราฟเชิงลึก และสถิติเชิงตัวเลขของจีโนม
        3. **โหมดเปรียบเทียบ:** ดูตารางตัวเลขเทียบหลายสายพันธุ์ (Ranking) และกราฟ Interactive
        """)
    
    st.caption("Developed for High School Science Project")

# ============================================
# 4. Main Analysis Area
# ============================================

st.markdown('<h1 class="main-header">Genome Analysis Dashboard</h1>', unsafe_allow_html=True)

# เช็คว่ามีไฟล์ที่อัปโหลด หรือ มีไฟล์จาก NCBI อย่างใดอย่างหนึ่งหรือไม่
has_files = bool(uploaded_files)
has_ncbi = bool(st.session_state['ncbi_cache'])

if not has_files and not has_ncbi:
    st.info("⬅️ กรุณาอัปโหลดไฟล์ .gbff หรือดึงข้อมูลจาก NCBI ที่แถบเมนูด้านซ้ายเพื่อเริ่มต้น")
    
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
        # 1. ประมวลผลจากไฟล์ที่อัปโหลดเข้ามา
        if has_files:
            for uploaded_file in uploaded_files:
                content = uploaded_file.getvalue().decode("utf-8")
                data, err = process_genbank(content, uploaded_file.name)
                if data:
                    results.append(data)
                else:
                    errors.append(err)
        
        # 2. ประมวลผลจากไฟล์ NCBI 
        if has_ncbi:
            for item in st.session_state['ncbi_cache']:
                data, err = process_genbank(item['content'], item['filename'])
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
        st.caption(f"File: {data['filename']} | จำนวนโครโมโซมที่พบ: {data['total_chromosomes']} โครโมโซม")
        
        # --- ระบบสลับโครโมโซมด้วยการจิ้มแท่งกราฟโดยตรง ---
        chrom_ids = list(data['chromosomes'].keys())
        
        # ล็อกตัวแปรลงใน Session State เพื่อจำว่าตอนนี้เลือกแท่งไหนอยู่
        if 'selected_chrom_id' not in st.session_state or st.session_state.selected_chrom_id not in chrom_ids:
            st.session_state.selected_chrom_id = chrom_ids[0]
            
        if len(chrom_ids) > 1:
            st.markdown("### 🧬 แผนผังโครโมโซม (Chromosome Map)")
            st.write("👆 **คลิกจิ้มเลือกที่แท่งโครโมโซม** ในแผนผังด้านล่างนี้ เพื่อสลับดูสถิติและข้อมูลเชิงลึกได้ทันที")

            # 1. จัดเตรียมข้อมูลสำหรับวาดรูป
            c_names = []
            c_lengths = []
            for cid in chrom_ids:
                desc = data['chromosomes'][cid]['desc']
                match = re.search(r'chromosome\s+([A-Za-z0-9]+)', desc, re.IGNORECASE)
                short_name = match.group(1) if match else cid
                c_names.append(short_name)
                c_lengths.append(data['chromosomes'][cid]['len'])

            name_to_id = dict(zip(c_names, chrom_ids))

            # 2. สร้างระบบไฮไลท์สี: แท่งที่ถูกเลือกจะเป็นสีม่วงเด่น แท่งอื่นจะใสจางๆ แบบ NCBI
            colors = []
            line_colors = []
            for cid in chrom_ids:
                if cid == st.session_state.selected_chrom_id:
                    colors.append('rgba(99, 102, 241, 0.7)') # ไฮไลท์น้ำเงิน/ม่วงแบบทึบแสง
                    line_colors.append('#818cf8')
                else:
                    colors.append('rgba(255, 255, 255, 0.08)') # สีจางโปร่งใส
                    line_colors.append('#9CA3AF')

            fig = go.Figure(data=[
                go.Bar(
                    x=c_names,
                    y=c_lengths,
                    marker=dict(color=colors, line=dict(color=line_colors, width=2)),
                    width=0.4,
                    hoverinfo='x+y',
                    hovertemplate='Chromosome %{x}<br>Length: %{y:,} bp<extra></extra>'
                )
            ])

            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, zeroline=False, side='top', tickfont=dict(size=14, color="#E5E7EB"), fixedrange=True),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, autorange="reversed", fixedrange=True),
                height=220,
                margin=dict(l=0, r=0, t=40, b=0),
                dragmode=False
            )
            
            # เปิดโหมดดักฟังคำสั่งคลิกเลือกวัตถุบนตัวกราฟ
            chart_event = st.plotly_chart(
                fig, 
                use_container_width=True, 
                config={'displayModeBar': False},
                on_select="rerun",  # สั่งหน้าเว็กรันใหม่ทันทีเมื่อจิ้มกราฟ
                selection_mode="points"
            )
            
            # ตรวจสอบตัวแปรเหตุการณ์ว่ามีคนเอาเมาส์ไปจิ้มแท่งกราฟไหม
            if chart_event and "selection" in chart_event and "points" in chart_event["selection"]:
                points = chart_event["selection"]["points"]
                if len(points) > 0:
                    clicked_x = points[0]["x"]
                    clicked_id = name_to_id.get(clicked_x)
                    # ถ้าคลิกแท่งใหม่ที่ไม่ซ้ำกับของเดิม ให้บันทึกค่าแล้วเปลี่ยนโครโมโซมทันที
                    if clicked_id and clicked_id != st.session_state.selected_chrom_id:
                        st.session_state.selected_chrom_id = clicked_id
                        st.rerun()

            selected_chrom_id = st.session_state.selected_chrom_id
            st.markdown("<br>", unsafe_allow_html=True)
        else:
            selected_chrom_id = chrom_ids[0]
            
        # ดึงข้อมูลเฉพาะโครโมโซมที่ถูกเลือกมาใช้งาน
        c_data = data['chromosomes'][selected_chrom_id]
        
        # 1. Key Metrics 
        st.markdown(f"#### 📊 ข้อมูลเชิงลึกของ: {selected_chrom_id} ({c_data['desc']})")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Chromosome Length", f"{c_data['len']:,} bp")
        m2.metric("GC Content", f"{c_data['gc_total']:.2f}%")
        m3.metric("Coding DNA (CDS)", f"{c_data['coding_pct']:.2f}%")
        m4.metric("Junk/Non-coding", f"{c_data['nc_pct']:.2f}%")
        
        st.divider()

        # 2. Charts Row 1
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("**1. การกระจายตัวของ Intergenic Length**")
            lengths = [len(i) for i in c_data['intergenic_seqs'] if len(i) > 0]
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
            gc_coding = [calculate_gc(c_data['seq'][s:e]) for s, e in c_data['cds_regions']]
            gc_nc = [calculate_gc(s) for s in c_data['intergenic_seqs'] if len(s) > 0]
            
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
        seq = c_data['seq']
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
            for s in c_data['intergenic_seqs']:
                total_repeats += find_simple_repeats(s, motif_input, threshold_input)
            
            st.metric(f"Found '{motif_input}' repeated >{threshold_input} times", f"{total_repeats:,} spots")

        with ac2:
            st.markdown("#### Export Data")
            st.caption("ดาวน์โหลดลำดับเบสส่วน Junk DNA (FASTA) เพื่อนำไปวิเคราะห์ต่อ")
            
            fasta_str = ""
            for i, seq_segment in enumerate(c_data['intergenic_seqs']):
                if len(seq_segment) > 0:
                    fasta_str += f">Intergenic_{i+1}_{c_data['id']}\n{seq_segment}\n"
            
            st.download_button(
                label="Download Non-coding Sequences (.fasta)",
                data=fasta_str,
                file_name=f"{c_data['id']}_junk_dna.fasta",
                mime="text/plain"
            )
            
        # ============================================
        # เพิ่มเติม: ส่วนเปรียบเทียบโครโมโซมภายในสิ่งมีชีวิตเดียวกัน
        # ============================================
        if data['total_chromosomes'] > 1:
            st.markdown("---")
            st.markdown(f"### 📊 เปรียบเทียบโครโมโซมภายใน {data['name']} (Intra-organism)")
            st.write("ตารางและกราฟแสดงการเปรียบเทียบค่าสถิติระหว่างโครโมโซมต่างๆ ภายในสิ่งมีชีวิตนี้")

            # 1. เตรียมข้อมูลตาราง
            chrom_list = []
            for cid, cinfo in data['chromosomes'].items():
                chrom_list.append({
                    "Chromosome": cid,
                    "Length (bp)": cinfo['len'],
                    "GC %": cinfo['gc_total'],
                    "Coding %": cinfo['coding_pct'],
                    "Non-coding (Junk) %": cinfo['nc_pct']
                })
            df_chroms = pd.DataFrame(chrom_list)

            # แสดงตาราง
            st.dataframe(df_chroms.style.highlight_max(axis=0, color='#1e40af'), use_container_width=True)

            # 2. กราฟเปรียบเทียบโครโมโซม
            cc1, cc2 = st.columns(2)
            
            with cc1:
                st.markdown("**เปรียบเทียบขนาดโครโมโซม**")
                # ใช้ Plotly สร้างกราฟแท่งแบบ Interactive
                fig_c1 = px.bar(
                    df_chroms, 
                    x="Chromosome", 
                    y="Length (bp)", 
                    color="GC %", 
                    template="plotly_dark",
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig_c1, use_container_width=True)
                
            with cc2:
                st.markdown("**ความสัมพันธ์: ขนาดโครโมโซม vs Non-coding %**")
                # ใช้ Plotly สร้าง Scatter Plot
                fig_c2 = px.scatter(
                    df_chroms, 
                    x="Length (bp)", 
                    y="Non-coding (Junk) %", 
                    color="GC %", 
                    size="Length (bp)",
                    hover_name="Chromosome", 
                    template="plotly_dark",
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig_c2, use_container_width=True)

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

        # 2. Interactive Charts
        st.markdown("---")
        st.markdown("#### ความสัมพันธ์: ขนาดจีโนม vs Junk DNA (Interactive)")
        st.caption("ℹ️ เอาเมาส์ชี้ที่จุดเพื่อดูชื่อสิ่งมีชีวิต / หมุนลูกกลิ้งเพื่อซูม")
        
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
