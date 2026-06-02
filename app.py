import streamlit as st
from Bio import SeqIO, Entrez  
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import plotly.express as px
import plotly.graph_objects as go  
import re
import google.generativeai as genai  

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
    .stButton button { width: 100%; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# สร้าง Session State สำหรับเก็บข้อมูลจาก NCBI
if 'ncbi_cache' not in st.session_state:
    st.session_state['ncbi_cache'] = []

# ============================================
# 2. Helper Functions (Logic)
# ============================================
def fetch_ncbi(acc_id, email):
    """ดึงข้อมูล GenBank จาก NCBI พร้อมรองรับทั้งรหัส Nucleotide (NC_) และ Assembly (GCF_/GCA_)"""
    Entrez.email = email
    acc_id = acc_id.strip().upper()
    
    if acc_id.startswith("GCF_") or acc_id.startswith("GCA_"):
        # 1. ค้นหา Assembly ID
        with Entrez.esearch(db="assembly", term=acc_id) as search_handle:
            search_rec = Entrez.read(search_handle)
            if not search_rec["IdList"]:
                raise Exception(f"ไม่พบข้อมูลสำหรับ Assembly: {acc_id}")
            assembly_id = search_rec["IdList"][0]
        
        # 2. แกะกล่อง หาลิงก์โครโมโซมย่อยๆ ไปยังฐานข้อมูล Nucleotide
        with Entrez.elink(dbfrom="assembly", db="nucleotide", id=assembly_id) as link_handle:
            link_rec = Entrez.read(link_handle)
            if not link_rec[0].get("LinkSetDb"):
                raise Exception(f"ไม่พบข้อมูลลำดับเบสที่เชื่อมโยงกับ Assembly: {acc_id}")
            
            nucl_ids = [link["Id"] for link in link_rec[0]["LinkSetDb"][0]["Link"]]
            
            if len(nucl_ids) > 50:
                raise Exception(f"จีโนมนี้ประกอบด้วยชิ้นส่วนถึง {len(nucl_ids)} ชิ้น แนะนำให้ดาวน์โหลดไฟล์ .gbff จากเว็บ NCBI มาอัปโหลดเองครับ")
            
            # 3. ดาวน์โหลดโครโมโซมย่อยทั้งหมดรวดเดียว
            id_string = ",".join(nucl_ids)
            with Entrez.efetch(db="nucleotide", id=id_string, rettype="gbwithparts", retmode="text") as fetch_handle:
                return fetch_handle.read()
    else:
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
        records = list(SeqIO.parse(io.StringIO(file_content), "genbank"))
        if not records: return None, "ไม่พบข้อมูลในไฟล์"
    except Exception as e:
        return None, f"Error reading {filename}: {e}"

    # ระบบกรองโครโมโซมซ้ำ (Deduplication) เพื่อแก้ปัญหาแท่งกราฟซ้อนกัน
    seen_short_names = set()
    filtered_records = []
    for record in records:
        match = re.search(r'chromosome\s+([A-Za-z0-9]+)', record.description, re.IGNORECASE)
        short_name = match.group(1).upper() if match else record.id
        
        if short_name in seen_short_names:
            continue 
        seen_short_names.add(short_name)
        filtered_records.append(record)
    records = filtered_records

    chromosomes_data = {}
    total_len = 0
    total_coding_len = 0
    total_gc = 0
    
    for record in records:
        seq = str(record.seq).upper()
        slen = len(seq)
        total_len += slen
        total_gc += (seq.count("G") + seq.count("C"))
        
        # Extract CDS & Amino Acids
        cds_regions = []
        protein_seqs = []  
        for f in record.features:
            if f.type == "CDS":
                cds_regions.append((int(f.location.start), int(f.location.end)))
                if 'translation' in f.qualifiers:
                    protein_seqs.append(f.qualifiers['translation'][0].upper())
                    
        cds_regions.sort()

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

        # คำนวณความถี่ของกรดอะมิโน
        all_proteins_combined = "".join(protein_seqs)
        aa_list = list("ACDEFGHIKLMNPQRSTVWY")
        aa_dist = {aa: all_proteins_combined.count(aa) for aa in aa_list} if all_proteins_combined else {}

        chromosomes_data[record.id] = {
            "id": record.id,
            "desc": record.description,
            "len": slen,
            "seq": seq,
            "cds_regions": cds_regions,
            "coding_pct": coding_pct,
            "nc_pct": nc_pct,
            "intergenic_seqs": intergenic_seqs,
            "gc_total": calculate_gc(seq),
            "aa_dist": aa_dist, 
            "total_proteins": len(protein_seqs)
        }

    # --- 🛠️ ระบบจัดเรียงลำดับโครโมโซมตามลำดับธรรมชาติ (Natural Sorting) ---
    def roman_to_int(roman_str):
        roman_dict = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        res = 0
        for i in range(len(roman_str)):
            if i + 1 < len(roman_str) and roman_dict.get(roman_str[i], 0) < roman_dict.get(roman_str[i+1], 0):
                res -= roman_dict.get(roman_str[i], 0)
            else:
                res += roman_dict.get(roman_str[i], 0)
        return res

    def chrom_key(item):
        cinfo = item[1]
        match = re.search(r'chromosome\s+([A-Za-z0-9]+)', cinfo['desc'], re.IGNORECASE)
        if match:
            val = match.group(1).upper()
            if val.isdigit():
                return (0, int(val), val) 
            if re.match(r'^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$', val) and val != "":
                return (0, roman_to_int(val), val) 
            return (1, 0, val) 
        return (2, 0, item[0]) 

    chromosomes_data = dict(sorted(chromosomes_data.items(), key=chrom_key))
    # -----------------------------------------------------------------------

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

def get_ai_response(api_key, prompt):
    """ฟังก์ชันส่งข้อมูลติดต่อกับ Gemini API เพื่อขอคำวิเคราะห์"""
    if not api_key:
        return "⚠️ กรุณาระบุ Google API Key ในแถบเมนูด้านซ้ายเพื่อใช้งานฟีเจอร์ AI"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash') 
        with st.spinner('AI กำลังวิเคราะห์ข้อมูลชีวสารสนเทศ...'):
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
    
    # --- ส่วนตั้งค่า AI ---
    st.markdown("---")
    st.subheader("🤖 AI Configuration")
    api_key = st.text_input("Google API Key", type="password", help="ใส่ API Key จาก Google AI Studio เพื่อเปิดใช้งานฟีเจอร์วิเคราะห์ด้วย AI")
    
    # --- ระบบดาวน์โหลดจาก NCBI ---
    st.markdown("---")
    st.subheader("🌐 ดึงข้อมูลจาก NCBI")
    ncbi_email = st.text_input("Email (จำเป็น)", placeholder="email@example.com")
    ncbi_id = st.text_input("RefSeq ID", placeholder="เช่น NC_000913 หรือ GCF_000146045.2")
    
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
        2. **ป้อน API Key:** หากต้องการใช้ระบบผู้ช่วยวิเคราะห์ AI แนะนำให้ระบุคีย์ที่แถบด้านซ้าย
        3. **โหมดไฟล์เดียว:** คลิกเลือกแท่งโครโมโซมเพื่อดูรายละเอียดและสั่งงาน AI รายโครโมโซมได้
        4. **โหมดเปรียบเทียบ:** เปรียบเทียบสถิติข้ามสายพันธุ์และให้ AI ช่วยสรุปความสัมพันธ์
        """)
    
    st.caption("Developed for High School Science Project")

# ============================================
# 4. Main Analysis Area
# ============================================
st.markdown('<h1 class="main-header">Genome Analysis Dashboard</h1>', unsafe_allow_html=True)

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
        st.markdown("### Data Export & AI Assistant")
        st.write("ส่งออกข้อมูล Junk DNA และใช้ AI สรุปเชิงชีววิทยา")

else:
    results = []
    errors = []
    
    with st.spinner('กำลังประมวลผล DNA...'):
        if has_files:
            for uploaded_file in uploaded_files:
                content = uploaded_file.getvalue().decode("utf-8")
                data, err = process_genbank(content, uploaded_file.name)
                if data: results.append(data)
                else: errors.append(err)
        
        if has_ncbi:
            for item in st.session_state['ncbi_cache']:
                data, err = process_genbank(item['content'], item['filename'])
                if data: results.append(data)
                else: errors.append(err)

    if errors:
        for e in errors: st.error(e)

    # ============================================
    # MODE A: Single File (Deep Dive)
    # ============================================
    if len(results) == 1:
        data = results[0]
        st.markdown(f"### ผลการวิเคราะห์: {data['name']}")
        st.caption(f"File: {data['filename']} | จำนวนโครโมโซมที่พบ: {data['total_chromosomes']} โครโมโซม")
        
        chrom_ids = list(data['chromosomes'].keys())
        
        if 'selected_chrom_id' not in st.session_state or st.session_state.selected_chrom_id not in chrom_ids:
            st.session_state.selected_chrom_id = chrom_ids[0]
            
        if len(chrom_ids) > 1:
            st.markdown("### 🧬 แผนผังโครโมโซม (Chromosome Map)")
            st.write("👆 **คลิกจิ้มเลือกที่แท่งโครโมโซม** ในแผนผังด้านล่างนี้ เพื่อสลับดูสถิติและข้อมูลเชิงลึกได้ทันที")

            c_names = []
            c_lengths = []
            for cid in chrom_ids:
                desc = data['chromosomes'][cid]['desc']
                match = re.search(r'chromosome\s+([A-Za-z0-9]+)', desc, re.IGNORECASE)
                short_name = match.group(1).upper() if match else cid
                c_names.append(short_name)
                c_lengths.append(data['chromosomes'][cid]['len'])

            name_to_id = dict(zip(c_names, chrom_ids))

            colors = []
            line_colors = []
            for cid in chrom_ids:
                if cid == st.session_state.selected_chrom_id:
                    colors.append('rgba(99, 102, 241, 0.7)') 
                    line_colors.append('#818cf8')
                else:
                    colors.append('rgba(255, 255, 255, 0.08)') 
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
            
            chart_event = st.plotly_chart(
                fig, use_container_width=True, config={'displayModeBar': False},
                on_select="rerun", selection_mode="points"
            )
            
            if chart_event and "selection" in chart_event and "points" in chart_event["selection"]:
                points = chart_event["selection"]["points"]
                if len(points) > 0:
                    clicked_x = points[0]["x"]
                    clicked_id = name_to_id.get(clicked_x)
                    if clicked_id and clicked_id != st.session_state.selected_chrom_id:
                        st.session_state.selected_chrom_id = clicked_id
                        st.rerun()

            selected_chrom_id = st.session_state.selected_chrom_id
            st.markdown("<br>", unsafe_allow_html=True)
        else:
            selected_chrom_id = chrom_ids[0]
            
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
                fig3, ax = plt.subplots(figsize=(6, 4))
                ax.hist(lengths, bins=50, color="#818cf8", edgecolor='#1f2937', alpha=0.9)
                ax.set_xlabel("Length (bp)")
                ax.set_ylabel("Frequency")
                ax.grid(axis='y', alpha=0.2, linestyle='--')
                st.pyplot(fig3)
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

        # --- 🧬 ส่วนเสริม: สัดส่วนกรดอะมิโนบนโครโมโซม ---
        st.divider()
        st.markdown(f"**4. สัดส่วนการกระจายตัวของกรดอะมิโน (Amino Acid Composition) - พบโปรตีนทั้งหมด {c_data['total_proteins']:,} ชนิด**")
        if c_data.get('aa_dist'):
            df_aa = pd.DataFrame(list(c_data['aa_dist'].items()), columns=['Amino Acid', 'Count'])
            
            aa_full_names = {
                'A': 'Alanine (อะลานีน)', 'C': 'Cysteine (ซิสเตอีน)', 'D': 'Aspartic acid (กรดแอสพาร์ติก)',
                'E': 'Glutamic acid (กรดกลูตามิก)', 'F': 'Phenylalanine (ฟีนิลอะลานีน)', 'G': 'Glycine (ไกลซีน)',
                'H': 'Histidine (ฮิสทิดีน)', 'I': 'Isoleucine (ไอโซลิวซีน)', 'K': 'Lysine (ไลซีน)',
                'L': 'Leucine (ลิวซีน)', 'M': 'Methionine (เมไทโอนีน)', 'N': 'Asparagine (แอสพาราจีน)',
                'P': 'Proline (โพรลีน)', 'Q': 'Glutamine (กลูตามีน)', 'R': 'Arginine (อาร์จินีน)',
                'S': 'Serine (เซรีน)', 'T': 'Threonine (ทรีโอนีน)', 'V': 'Valine (วาลีน)',
                'W': 'Tryptophan (ทริปโตเฟน)', 'Y': 'Tyrosine (ไทโรซีน)'
            }
            df_aa['Amino Acid'] = df_aa['Amino Acid'].map(aa_full_names)
            df_aa = df_aa.sort_values(by="Count", ascending=False) 
            
            fig_aa = px.bar(
                df_aa, x='Amino Acid', y='Count', color='Count',
                template='plotly_dark', color_continuous_scale="Blugrn",
                labels={"Count": "จำนวนครั้งที่พบ"}
            )
            fig_aa.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=380)
            st.plotly_chart(fig_aa, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("⚠️ ไม่พบลำดับข้อมูลกรดอะมิโน (Translation Feature) ในไฟล์นี้")

        # --- 🤖 ส่วนเสริม: ระบบวิเคราะห์และถามตอบด้วย AI รายโครโมโซม (อัปเดต Focus Prompt) ---
        st.divider()
        st.subheader("🧬 AI Biological Assistant")
        ai_col1, ai_col2 = st.columns([1, 2])
        
        with ai_col1:
            st.markdown("ระบบจะส่งสถิติของทุกโครโมโซมให้ AI เห็นภาพรวม แต่สั่งให้ AI **โฟกัสคำตอบเฉพาะโครโมโซมที่คุณเลือก** หรือตอบคำถามเฉพาะเจาะจงด้านล่าง")
            user_question = st.text_input("💡 ถามคำถามชีววิทยาเกี่ยวกับโครโมโซมนี้", placeholder="เช่น ทำไมโครโมโซมนี้ถึงมีกรดอะมิโน Serine มากเป็นพิเศษเมื่อเทียบกับแท่งอื่น?")
            run_ai = st.button("✨ เริ่มการวิเคราะห์ด้วย AI")
            
        with ai_col2:
            if run_ai:
                if not api_key:
                    st.error("❌ กรุณากรอก Google API Key ที่แถบเมนูด้านซ้ายก่อนกดปุ่มวิเคราะห์ครับ")
                else:
                    # 1. สร้างตารางสรุปสถิติของ "ทุกโครโมโซม" เพื่อให้ AI ใช้เปรียบเทียบแนวโน้ม (Global Genome Context)
                    all_chroms_summary = ""
                    for cid, cinfo in data['chromosomes'].items():
                        all_chroms_summary += f"- โครโมโซม {cid}: ความยาว={cinfo['len']:,} bp, GC={cinfo['gc_total']:.2f}%, Non-coding={cinfo['nc_pct']:.2f}%, จำนวนโปรตีน={cinfo['total_proteins']:,} ชนิด\n"
                    
                    # 2. ออกแบบ Prompt สั่งควบคุมโฟกัสของ AI อย่างเข้มงวด
                    prompt = f"""
                    You are an expert Bioinformatics AI Assistant. Analyze the genomic data of this organism.
                    
                    [GLOBAL GENOME CONTEXT]
                    Organism Classification Name: {data['name']}
                    Total Chromosomes in this file: {data['total_chromosomes']}
                    Here is the statistical summary of ALL chromosomes for your baseline comparison:
                    {all_chroms_summary}
                    
                    [TARGET FOCUS - โครโมโซมที่ผู้ใช้เลือกตรวจสอบอยู่ตอนนี้]
                    Selected Chromosome ID: {selected_chrom_id}
                    Description: {c_data['desc']}
                    Length: {c_data['len']:,} bp
                    GC Content: {c_data['gc_total']:.2f}%
                    Coding Region (CDS) Ratio: {c_data['coding_pct']:.2f}%
                    Non-coding Region (Junk DNA) Ratio: {c_data['nc_pct']:.2f}%
                    Total Protein Products: {c_data['total_proteins']:,}
                    Amino Acid Distribution on this targeted chromosome: {c_data['aa_dist']}
                    
                    [USER QUESTION / INTENT]
                    Question: {user_question if user_question else "Please provide a comprehensive biological analysis and evolutionary summary of the selected chromosome."}
                    
                    CRITICAL INSTRUCTION:
                    1. Focus your answer primarily on the [TARGET FOCUS] chromosome and directly answer the user's question or analyze it deeply.
                    2. Use the [GLOBAL GENOME CONTEXT] data only to make meaningful biological comparisons (e.g., pointing out if the selected chromosome has distinct patterns like higher/lower GC or specific amino acid abundance compared to the rest of the genome).
                    3. Do not generalize the answer to the whole genome unless making a comparison. Keep the focus tight.
                    4. Answer in scientifically rigorous, clear, and beautiful Thai language.
                    """
                    
                    response_text = get_ai_response(api_key, prompt)
                    st.markdown("### 📝 สรุปและคำแนะนำจาก AI")
                    st.info(response_text)
            else:
                st.info("💡 ระบุ API Key ทางซ้ายมือ แล้วกดปุ่มวิเคราะห์เพื่อรับข้อมูลเชิงลึกจาก AI")

        # 5. Advanced Analysis Section
        st.markdown("---")
        st.subheader("Advanced Analysis: Repeats & Data")
        
        ac1, ac2 = st.columns(2)
        with ac1:
            st.markdown("#### Motif Search in Junk DNA")
            st.caption("ค้นหาจำนวนจุดที่พบรหัสพันธุกรรมซ้ำๆ ในส่วน Non-coding")
            
            sc1, sc2 = st.columns(2)
            with sc1: motif_input = st.text_input("Pattern (e.g. AT, G)", value="AT")
            with sc2: threshold_input = st.number_input("Min Repeats", min_value=3, value=5)
            
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
            
        if data['total_chromosomes'] > 1:
            st.markdown("---")
            st.markdown(f"### 📊 เปรียบเทียบโครโมโซมภายใน {data['name']} (Intra-organism)")
            st.write("ตารางและกราฟแสดงการเปรียบเทียบค่าสถิติระหว่างโครโมโซมต่างๆ ภายในสิ่งมีชีวิตนี้")

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
            st.dataframe(df_chroms.style.highlight_max(axis=0, color='#1e40af'), use_container_width=True)

            cc1, cc2 = st.columns(2)
            with cc1:
                st.markdown("**เปรียบเทียบขนาดโครโมโซม**")
                fig_c1 = px.bar(
                    df_chroms, x="Chromosome", y="Length (bp)", color="GC %", 
                    template="plotly_dark", color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig_c1, use_container_width=True)
                
            with cc2:
                st.markdown("**ความสัมพันธ์: ขนาดโครโมโซม vs Non-coding %**")
                fig_c2 = px.scatter(
                    df_chroms, x="Length (bp)", y="Non-coding (Junk) %", color="GC %", 
                    size="Length (bp)", hover_name="Chromosome", template="plotly_dark",
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

        # --- ส่วนเสริม: รายงานวิเคราะห์เปรียบเทียบข้ามสายพันธุ์ด้วย AI ---
        st.markdown("---")
        st.subheader("🤖 AI Comparative Insight")
        run_comp_ai = st.button("📊 สั่งให้ AI ทำรายงานวิเคราะห์เปรียบเทียบ")
        if run_comp_ai:
            if not api_key:
                st.error("❌ กรุณากรอก Google API Key ที่แถบเมนูด้านซ้ายก่อนกดปุ่มวิเคราะห์ครับ")
            else:
                data_str = df.to_string()
                prompt = f"""
                You are a Bioinformatics expert. Analyze this comparative data table of multiple organisms:
                {data_str}
                
                Please provide a rigorous comparative analysis addressing:
                1. Which organism demonstrates higher genetic complexity or evolutionary advancement based on genome size and coding vs non-coding ratios?
                2. Identify if there's any correlation between GC content variants and environmental adaptations or lifestyle among these species.
                3. Comment on the distribution patterns of Non-coding DNA (Junk DNA).
                Answer clearly in Thai language.
                """
                response_text = get_ai_response(api_key, prompt)
                st.markdown("### 📝 รายงานการวิเคราะห์ข้ามสิ่งมีชีวิตจาก AI")
                st.info(response_text)

        # 2. Interactive Charts
        st.markdown("---")
        st.markdown("#### ความสัมพันธ์: ขนาดจีโนม vs Junk DNA (Interactive)")
        st.caption("ℹ️ เอาเมาส์ชี้ที่จุดเพื่อดูชื่อสิ่งมีชีวิต / หมุนลูกกลิ้งเพื่อซูม")
        
        fig_scatter = px.scatter(
            df, x="Length (bp)", y="Non-coding %", color="GC %", size="Length (bp)",
            hover_name="Organism", color_continuous_scale="Viridis", template="plotly_dark",
            title="Genome Size vs. Non-coding DNA %"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

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
