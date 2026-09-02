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
import time  

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

# สร้าง Session State 
if 'ncbi_cache' not in st.session_state:
    st.session_state['ncbi_cache'] = []
if 'ncbi_search_results' not in st.session_state:
    st.session_state['ncbi_search_results'] = None

# ============================================
# 2. Helper Functions (Logic)
# ============================================

def safe_ncbi_call(func, max_retries=5, is_fetch=False, **kwargs):
    """🛡️ ฟังก์ชันเกราะป้องกัน: ครอบการทำงานของ NCBI ทุกจุดเพื่อสู้กับอาการเน็ตหลุด/เซิร์ฟเวอร์ล่ม"""
    for attempt in range(max_retries):
        try:
            with func(**kwargs) as handle:
                if is_fetch:
                    data = handle.read()
                    if "NCBI C++ Exception" in data or "Error: TXCLIENT" in data:
                        raise Exception("NCBI Internal Error")
                    return data
                else:
                    return Entrez.read(handle)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(3) 
                continue
            else:
                err_msg = str(e)
                if "IncompleteRead" in err_msg or "EOF" in err_msg:
                    raise Exception("เซิร์ฟเวอร์ NCBI ตัดการเชื่อมต่อ (IncompleteRead/EOF) โปรดลองใหม่อีกครั้ง")
                raise Exception(f"เกิดข้อผิดพลาดในการดึงข้อมูล: {err_msg}")

def search_ncbi_genomes(query, email):
    Entrez.email = email
    search_term = f"({query}[Organism] OR {query}[All Fields]) AND \"latest refseq\"[filter]"
    
    record = safe_ncbi_call(Entrez.esearch, db="assembly", term=search_term, retmax=5)
    id_list = record.get("IdList", [])
        
    if not id_list:
        return []
        
    results = []
    summaries = safe_ncbi_call(Entrez.esummary, db="assembly", id=",".join(id_list))
    doc_sums = summaries.get('DocumentSummarySet', {}).get('DocumentSummary', [])
    
    for summary in doc_sums:
        acc = summary.get('AssemblyAccession', '')
        org = summary.get('SpeciesName', summary.get('Organism', 'Unknown Organism'))
        name = summary.get('AssemblyName', '')
        if acc:
            results.append({"id": acc, "label": f"{org} ({acc}) - {name[:20]}..."})
    return results

def fetch_ncbi(acc_id, email):
    Entrez.email = email
    acc_id = acc_id.strip().upper()

    if acc_id.startswith("GCF_") or acc_id.startswith("GCA_"):
        search_rec = safe_ncbi_call(Entrez.esearch, db="assembly", term=acc_id)
        if not search_rec["IdList"]:
            raise Exception(f"ไม่พบข้อมูลสำหรับ Assembly: {acc_id}")
        assembly_id = search_rec["IdList"][0]
        
        link_rec = safe_ncbi_call(Entrez.elink, dbfrom="assembly", db="nucleotide", id=assembly_id)
        if not link_rec[0].get("LinkSetDb"):
            raise Exception(f"ไม่พบข้อมูลลำดับเบสที่เชื่อมโยงกับ Assembly: {acc_id}")
        
        nucl_ids = [link["Id"] for link in link_rec[0]["LinkSetDb"][0]["Link"]]
        
        if len(nucl_ids) > 300:
            raise Exception(f"ระบบตรวจพบชิ้นส่วนจีโนมจำนวน {len(nucl_ids)} ชิ้น ซึ่งเกินขีดจำกัดการเชื่อมต่อชั่วคราว โปรดดาวน์โหลดไฟล์ .gbff โดยตรงจากเว็บไซต์ NCBI เพื่อทำการวิเคราะห์")
        
        all_data = ""
        batch_size = 5 
        for i in range(0, len(nucl_ids), batch_size):
            batch_ids = nucl_ids[i:i+batch_size]
            id_string = ",".join(batch_ids)
            all_data += safe_ncbi_call(Entrez.efetch, is_fetch=True, db="nucleotide", id=id_string, rettype="gbwithparts", retmode="text")
            time.sleep(0.5)
            
        return all_data
    else:
        return safe_ncbi_call(Entrez.efetch, is_fetch=True, db="nucleotide", id=acc_id, rettype="gbwithparts", retmode="text")

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
    try:
        records = list(SeqIO.parse(io.StringIO(file_content), "genbank"))
        if not records: return None, "ไม่พบข้อมูลในไฟล์"
    except Exception as e:
        return None, f"Error reading {filename}: {e}"

    seen_short_names = set()
    filtered_records = []
    for record in records:
        match = re.search(r'chromosome\s+([A-Za-z0-9]+)', record.description, re.IGNORECASE)
        short_name = match.group(1).upper() if match else record.id
        
        if short_name in seen_short_names: continue 
        seen_short_names.add(short_name)
        filtered_records.append(record)
    records = filtered_records

    chromosomes_data = {}
    total_len = 0
    total_coding_len = 0
    total_gc = 0
    
    for record in records:
        from Bio.Seq import UndefinedSequenceError
    # ... ภายใน loop for record in records:
        try:
            seq = str(record.seq).upper()
        except UndefinedSequenceError:
    # ถ้าไม่มีลำดับเบส ให้สร้างเบส N ตามความยาวของ record แทน
            seq = "N" * len(record)
        slen = len(seq)
        total_len += slen
        total_gc += (seq.count("G") + seq.count("C"))
        
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
        
        intergenic_seqs = []
        prev = 0
        for s, e in cds_regions:
            if s > prev: intergenic_seqs.append(seq[prev:s])
            prev = e
        if prev < slen: intergenic_seqs.append(seq[prev:slen])

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
            if val.isdigit(): return (0, int(val), val) 
            if re.match(r'^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$', val) and val != "":
                return (0, roman_to_int(val), val) 
            return (1, 0, val) 
        return (2, 0, item[0]) 

    chromosomes_data = dict(sorted(chromosomes_data.items(), key=chrom_key))
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
    if not api_key: return "⚠️ โปรดระบุ Google API Key ในแถบเมนูด้านซ้ายเพื่อใช้งานระบบวิเคราะห์"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash') 
        with st.spinner('ระบบกำลังประมวลผลการวิเคราะห์ทางชีวสารสนเทศ...'):
            response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"เกิดข้อผิดพลาดในการเชื่อมต่อระบบ AI: {str(e)}"

# ============================================
# 3. Sidebar: Inputs & Instructions
# ============================================
with st.sidebar:
    st.title("Genome Analyzer")
    st.markdown("ระบบวิเคราะห์และประมวลผลข้อมูลจีโนม")
    
    st.markdown("---")
    st.subheader("⚙️ การตั้งค่าปัญญาประดิษฐ์ (AI)")
    api_key = st.text_input("Google API Key", type="password", help="โปรดระบุ API Key จาก Google AI Studio เพื่อเปิดใช้งานระบบวิเคราะห์เชิงลึก")
    
    st.markdown("---")
    st.subheader("🌐 ระบบสืบค้นฐานข้อมูล NCBI")
    ncbi_email = st.text_input("อีเมล (บังคับสำหรับการเข้าถึง NCBI)", placeholder="email@example.com")
    
    tab1, tab2 = st.tabs(["🔍 สืบค้นด้วยชื่อ", "📝 สืบค้นด้วยรหัสอ้างอิง"])
    
    with tab1:
        st.write("สืบค้นข้อมูลจีโนมจากฐานข้อมูล Assembly")
        search_query = st.text_input("ระบุชื่อวิทยาศาสตร์ (ตัวอย่าง: Yeast, E. coli)", key="search_q")
        
        if st.button("🔍 ดำเนินการสืบค้น"):
            if not ncbi_email:
                st.warning("⚠️ โปรดระบุอีเมลก่อนทำการสืบค้นข้อมูล")
            elif search_query:
                with st.spinner(f"กำลังสืบค้นข้อมูล '{search_query}'..."):
                    try:
                        results = search_ncbi_genomes(search_query, ncbi_email)
                        st.session_state['ncbi_search_results'] = results
                    except Exception as e:
                        st.error(f"เกิดข้อผิดพลาดในการสืบค้น: {e}")
        
        if st.session_state.get('ncbi_search_results') is not None:
            res_list = st.session_state['ncbi_search_results']
            if len(res_list) > 0:
                options = {r['id']: r['label'] for r in res_list}
                selected_acc = st.selectbox("ผลการค้นหา โปรดระบุจีโนมที่ต้องการ:", options=list(options.keys()), format_func=lambda x: options[x])
                
                if st.button("📥 นำเข้าข้อมูลจีโนม"):
                    with st.spinner(f"กำลังนำเข้าและประมวลผลข้อมูลรหัส {selected_acc} (อาจใช้เวลาสักครู่เนื่องจากมาตรการความปลอดภัยของเซิร์ฟเวอร์)..."):
                        try:
                            raw_data = fetch_ncbi(selected_acc, ncbi_email)
                            if not any(item['id'] == selected_acc for item in st.session_state['ncbi_cache']):
                                st.session_state['ncbi_cache'].append({
                                    "id": selected_acc,
                                    "filename": f"NCBI_{selected_acc}.gbff",
                                    "content": raw_data
                                })
                            st.success(f"นำเข้าข้อมูล {selected_acc} สำเร็จ!")
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))
            else:
                st.info("❌ ไม่พบข้อมูลจีโนมดังกล่าวในฐานข้อมูล โปรดตรวจสอบการสะกดชื่ออีกครั้ง")

    with tab2:
        st.write("สืบค้นข้อมูลผ่านรหัสอ้างอิง RefSeq หรือ GenBank")
        ncbi_id = st.text_input("รหัสอ้างอิง (Accession ID)", placeholder="ตัวอย่าง: NC_000913 หรือ GCF_000146045.2", key="manual_id")
        
        if st.button("📥 นำเข้าข้อมูลด้วยรหัสอ้างอิง"):
            if not ncbi_email or not ncbi_id:
                st.error("โปรดระบุอีเมลและรหัสอ้างอิงให้ครบถ้วน")
            else:
                with st.spinner(f"กำลังนำเข้าและประมวลผลข้อมูลรหัส {ncbi_id} (อาจใช้เวลาสักครู่)..."):
                    try:
                        raw_data = fetch_ncbi(ncbi_id.strip(), ncbi_email.strip())
                        if not any(item['id'] == ncbi_id for item in st.session_state['ncbi_cache']):
                            st.session_state['ncbi_cache'].append({
                                "id": ncbi_id.strip(),
                                "filename": f"NCBI_{ncbi_id.strip()}.gbff",
                                "content": raw_data
                            })
                        st.success("นำเข้าข้อมูลสำเร็จ!")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
                    
    if st.session_state['ncbi_cache']:
        st.markdown(f"*(ข้อมูลปัจจุบันจาก NCBI: จำนวน {len(st.session_state['ncbi_cache'])} รายการ)*")
        if st.button("🗑️ ล้างข้อมูลในระบบ"):
            st.session_state['ncbi_cache'] = []
            st.session_state['ncbi_search_results'] = None
            st.rerun()
            
    st.markdown("---")
    st.subheader("📂 นำเข้าไฟล์ข้อมูล (Upload)")
    uploaded_files = st.file_uploader(
        "รองรับเฉพาะไฟล์รูปแบบ .gbff", 
        type=["gbff"], 
        accept_multiple_files=True
    )

    st.markdown("---")
    with st.expander("📖 คู่มือการใช้งานระบบ"):
        st.markdown("""
        1. **การนำเข้าข้อมูล:** อัปโหลดไฟล์รูปแบบ `.gbff` หรือสืบค้นจากฐานข้อมูล NCBI ทางแถบเมนูด้านซ้าย
        2. **การตั้งค่าความสามารถขั้นสูง:** ระบุ API Key เพื่อเปิดใช้งานระบบผู้ช่วยวิเคราะห์ทางชีววิทยาด้วย AI
        3. **โหมดวิเคราะห์เดี่ยว:** เลือกโครโมโซมบนแผนภาพเพื่อประเมินค่าทางสถิติและวิเคราะห์ข้อมูลเชิงลึกเฉพาะส่วน
        4. **โหมดเปรียบเทียบ:** นำเข้าข้อมูลสิ่งมีชีวิตหลายชนิดเพื่อวิเคราะห์ความสัมพันธ์และสร้างรายงานเปรียบเทียบ
        """)

# ============================================
# 4. Main Analysis Area
# ============================================
st.markdown('<h1 class="main-header">Genome Analysis Dashboard</h1>', unsafe_allow_html=True)

has_files = bool(uploaded_files)
has_ncbi = bool(st.session_state['ncbi_cache'])

if not has_files and not has_ncbi:
    st.info("⬅️ โปรดนำเข้าไฟล์ `.gbff` หรือสืบค้นข้อมูลจากระบบ NCBI ทางเมนูด้านซ้ายเพื่อเริ่มต้นการวิเคราะห์")
    
    cols = st.columns(3)
    with cols[0]:
        st.markdown("### การวิเคราะห์เชิงลึก")
        st.write("วิเคราะห์โครงสร้างยีนและบริเวณ Non-coding DNA อย่างละเอียด")
    with cols[1]:
        st.markdown("### การเปรียบเทียบข้อมูล")
        st.write("ประเมินเปรียบเทียบข้อมูลทางสถิติระหว่างสายพันธุ์")
    with cols[2]:
        st.markdown("### ส่งออกข้อมูลและผู้ช่วย AI")
        st.write("ส่งออกข้อมูลลำดับเบสและสร้างรายงานวิเคราะห์ด้วยปัญญาประดิษฐ์")

else:
    results = []
    errors = []
    
    with st.spinner('ระบบกำลังประมวลผลข้อมูลจีโนม...'):
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
        st.markdown(f"### รายงานผลการวิเคราะห์จีโนม: {data['name']}")
        st.caption(f"แฟ้มข้อมูล: {data['filename']} | ข้อมูลโครโมโซมที่ตรวจพบ: {data['total_chromosomes']} หน่วย")
        
        st.markdown("#### 🌍 สรุปภาพรวมระดับจีโนม (Whole Genome Summary)")
        wg1, wg2, wg3, wg4 = st.columns(4)
        wg1.metric("ขนาดจีโนมรวม (Total Size)", f"{data['len']:,} bp")
        wg2.metric("ปริมาณ GC (Total GC)", f"{data['gc_total']:.2f}%")
        wg3.metric("รหัสสร้างโปรตีน (Overall CDS)", f"{data['coding_pct']:.2f}%")
        wg4.metric("Non-coding DNA", f"{data['nc_pct']:.2f}%")
        st.divider()
        
        chrom_ids = list(data['chromosomes'].keys())
        
        if 'selected_chrom_id' not in st.session_state or st.session_state.selected_chrom_id not in chrom_ids:
            st.session_state.selected_chrom_id = chrom_ids[0]
            
        if len(chrom_ids) > 1:
            st.markdown("### 🧬 แผนผังโครโมโซม (Chromosome Mapping)")
            st.write("👆 **โปรดคลิกเลือกแท่งโครโมโซมบนแผนผังด้านล่าง** เพื่อเรียกดูข้อมูลทางสถิติและผลการวิเคราะห์เชิงลึกเฉพาะส่วน")

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
        st.markdown(f"#### 📊 สรุปข้อมูลทางสถิติของส่วน: {selected_chrom_id} ({c_data['desc']})")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ความยาวลำดับเบส", f"{c_data['len']:,} bp")
        m2.metric("ปริมาณ GC (GC Content)", f"{c_data['gc_total']:.2f}%")
        m3.metric("รหัสสร้างโปรตีน (CDS)", f"{c_data['coding_pct']:.2f}%")
        m4.metric("ส่วนที่ไม่ได้สร้างโปรตีน", f"{c_data['nc_pct']:.2f}%")
        
        st.divider()

        # 2. Charts Row 1
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**1. การกระจายตัวของความยาวบริเวณ Intergenic (Length Distribution)**")
            lengths = [len(i) for i in c_data['intergenic_seqs'] if len(i) > 0]
            if lengths:
                fig3, ax = plt.subplots(figsize=(6, 4))
                ax.hist(lengths, bins=50, color="#818cf8", edgecolor='#1f2937', alpha=0.9)
                ax.set_xlabel("Length (bp)")
                ax.set_ylabel("Frequency")
                ax.grid(axis='y', alpha=0.2, linestyle='--')
                st.pyplot(fig3)
            else:
                st.warning("ระบบไม่พบข้อมูลบริเวณ Intergenic ในชุดข้อมูลนี้")

        with c2:
            st.markdown("**2. เปรียบเทียบปริมาณ GC (GC Content Comparison)**")
            gc_coding = [calculate_gc(c_data['seq'][s:e]) for s, e in c_data['cds_regions']]
            gc_nc = [calculate_gc(s) for s in c_data['intergenic_seqs'] if len(s) > 0]
            
            if gc_coding and gc_nc:
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                bp = ax2.boxplot([gc_coding, gc_nc], patch_artist=True)
                ax2.set_xticks([1, 2])
                ax2.set_xticklabels(["Coding Region", "Non-coding Region"])
                for box in bp['boxes']:
                    box.set(color='#34d399', linewidth=2)
                    box.set(facecolor='#065f46')
                for median in bp['medians']:
                    median.set(color='white', linewidth=2)
                ax2.set_ylabel("GC Percentage (%)")
                ax2.grid(axis='y', alpha=0.2, linestyle='--')
                st.pyplot(fig2)

        st.divider()
        
        # 3. Sliding Window
        st.markdown("**3. ความแปรปรวนของปริมาณ GC ตามแนวสายโครโมโซม (Sliding Window)**")
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
        st.markdown(f"**4. ความถี่และสัดส่วนการพบกรดอะมิโน (Amino Acid Composition) - ตรวจพบรหัสโปรตีนจำนวน {c_data['total_proteins']:,} ลำดับ**")
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
                labels={"Count": "ความถี่ที่พบ (หน่วย)"}
            )
            fig_aa.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=380)
            st.plotly_chart(fig_aa, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("⚠️ ระบบไม่พบข้อมูลการแปลรหัสโปรตีน (Translation Feature) ในข้อมูลส่วนนี้")

        # --- 🤖 ส่วนเสริม: ระบบวิเคราะห์และถามตอบด้วย AI รายโครโมโซม ---
        st.divider()
        st.subheader("🧬 ระบบผู้ช่วยวิเคราะห์ทางชีววิทยาด้วยปัญญาประดิษฐ์ (AI Assistant)")
        ai_col1, ai_col2 = st.columns([1, 2])
        
        with ai_col1:
            st.markdown("ระบบจะประมวลผลสถิติภาพรวมของจีโนมและวิเคราะห์เปรียบเทียบเจาะจงเฉพาะส่วนที่ท่านเลือก ท่านสามารถระบุคำถามเพิ่มเติมเพื่อการวิเคราะห์เฉพาะทางได้")
            user_question = st.text_input("💡 ระบุคำถามหรือสมมติฐานทางชีววิทยา:", placeholder="เช่น ปริมาณกรดอะมิโน Serine ที่สูงขึ้นมีความหมายทางวิวัฒนาการอย่างไร?")
            run_ai = st.button("✨ ดำเนินการวิเคราะห์ด้วย AI")
            
        with ai_col2:
            if run_ai:
                if not api_key:
                    st.error("❌ โปรดระบุ Google API Key ในแถบเมนูด้านซ้ายเพื่อเริ่มต้นกระบวนการวิเคราะห์")
                else:
                    all_chroms_summary = ""
                    for cid, cinfo in data['chromosomes'].items():
                        all_chroms_summary += f"- รหัส {cid}: ความยาว={cinfo['len']:,} bp, GC={cinfo['gc_total']:.2f}%, Non-coding={cinfo['nc_pct']:.2f}%, จำนวนโปรตีน={cinfo['total_proteins']:,} ชนิด\n"
                    
                    prompt = f"""
                    You are an expert Bioinformatics AI Assistant. Analyze the genomic data of this organism.
                    
                    [GLOBAL GENOME CONTEXT]
                    Organism Classification Name: {data['name']}
                    Total Chromosomes in this file: {data['total_chromosomes']}
                    Here is the statistical summary of ALL chromosomes for your baseline comparison:
                    {all_chroms_summary}
                    
                    [TARGET FOCUS]
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
                    2. Use the [GLOBAL GENOME CONTEXT] data only to make meaningful biological comparisons.
                    3. Do not generalize the answer to the whole genome unless making a comparison. Keep the focus tight.
                    4. Answer in scientifically rigorous, clear, formal, and academic Thai language suitable for a research presentation.
                    """
                    
                    response_text = get_ai_response(api_key, prompt)
                    st.markdown("### 📝 รายงานและผลการวิเคราะห์จาก AI")
                    st.info(response_text)
            else:
                st.info("💡 เมื่อระบุ API Key แล้ว โปรดกดปุ่มเพื่อรับรายงานเชิงวิชาการ")

        # 5. Advanced Analysis Section
        st.markdown("---")
        st.subheader("การวิเคราะห์ขั้นสูง: รูปแบบซ้ำและข้อมูลดิบ (Advanced Analysis)")
        
        ac1, ac2 = st.columns(2)
        with ac1:
            st.markdown("#### การค้นหาลำดับเบสซ้ำ (Motif Search)")
            st.caption("สืบค้นรูปแบบลำดับเบสซ้ำในบริเวณ Non-coding DNA")
            
            sc1, sc2 = st.columns(2)
            with sc1: motif_input = st.text_input("รูปแบบลำดับเบส (ตัวอย่าง: AT, G)", value="AT")
            with sc2: threshold_input = st.number_input("จำนวนรอบการทำซ้ำขั้นต่ำ", min_value=3, value=5)
            
            total_repeats = 0
            for s in c_data['intergenic_seqs']:
                total_repeats += find_simple_repeats(s, motif_input, threshold_input)
            
            st.metric(f"จำนวนตำแหน่งที่พบรูปแบบ '{motif_input}' ซ้ำมากกว่า {threshold_input} ครั้ง", f"{total_repeats:,} ตำแหน่ง")

        with ac2:
            st.markdown("#### การส่งออกข้อมูลดิบ (Data Export)")
            st.caption("ดึงข้อมูลลำดับเบสส่วน Non-coding DNA เพื่อนำไปประมวลผลต่อ (รูปแบบ FASTA)")
            
            fasta_str = ""
            for i, seq_segment in enumerate(c_data['intergenic_seqs']):
                if len(seq_segment) > 0:
                    fasta_str += f">Intergenic_{i+1}_{c_data['id']}\n{seq_segment}\n"
            
            st.download_button(
                label="📥 ดาวน์โหลดข้อมูล Non-coding Sequences (.fasta)",
                data=fasta_str,
                file_name=f"{c_data['id']}_junk_dna.fasta",
                mime="text/plain"
            )
            
        if data['total_chromosomes'] > 1:
            st.markdown("---")
            st.markdown(f"### 📊 การประเมินความแตกต่างระหว่างโครโมโซม (Intra-organism Comparison)")
            st.write("ตารางและแผนภาพแสดงการเปรียบเทียบค่าทางสถิติระหว่างโครโมโซมภายในจีโนมเดียวกัน")

            chrom_list = []
            for cid, cinfo in data['chromosomes'].items():
                chrom_list.append({
                    "รหัสอ้างอิง (ID)": cid,
                    "ความยาว (bp)": cinfo['len'],
                    "ปริมาณ GC (%)": cinfo['gc_total'],
                    "สัดส่วนรหัสโปรตีน (%)": cinfo['coding_pct'],
                    "สัดส่วน Non-coding (%)": cinfo['nc_pct']
                })
            df_chroms = pd.DataFrame(chrom_list)
            st.dataframe(df_chroms.style.highlight_max(axis=0, color='#1e40af'), use_container_width=True)

            cc1, cc2 = st.columns(2)
            with cc1:
                st.markdown("**เปรียบเทียบขนาดของโครโมโซม**")
                fig_c1 = px.bar(
                    df_chroms, x="รหัสอ้างอิง (ID)", y="ความยาว (bp)", color="ปริมาณ GC (%)", 
                    template="plotly_dark", color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig_c1, use_container_width=True)
                
            with cc2:
                st.markdown("**ความสัมพันธ์: ขนาดโครโมโซม และ สัดส่วน Non-coding DNA**")
                fig_c2 = px.scatter(
                    df_chroms, x="ความยาว (bp)", y="สัดส่วน Non-coding (%)", color="ปริมาณ GC (%)", 
                    size="ความยาว (bp)", hover_name="รหัสอ้างอิง (ID)", template="plotly_dark",
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig_c2, use_container_width=True)

# ============================================
# MODE B: Multi-File (Comparison)
# ============================================
    elif len(results) > 1:
        st.markdown(f"### การประเมินความสัมพันธ์ระหว่างสายพันธุ์ (จำนวนข้อมูล: {len(results)} ตัวอย่าง)")
        
        df = pd.DataFrame([
            {
                "ชื่อสายพันธุ์ (Organism)": r['name'].split(',')[0],
                "ขนาดจีโนม (bp)": r['len'],
                "สัดส่วนรหัสโปรตีน (%)": r['coding_pct'],
                "สัดส่วน Non-coding (%)": r['nc_pct'],
                "ปริมาณ GC (%)": r['gc_total']
            } for r in results
        ])

        # 1. Summary Table
        st.markdown("#### ตารางสรุปข้อมูลทางสถิติข้ามสายพันธุ์ (Summary Table)")
        st.dataframe(df.style.highlight_max(axis=0, color='#1e40af'), use_container_width=True)

        # --- ส่วนเสริม: รายงานวิเคราะห์เปรียบเทียบข้ามสายพันธุ์ด้วย AI ---
        st.markdown("---")
        st.subheader("🤖 รายงานวิเคราะห์เปรียบเทียบเชิงวิวัฒนาการโดยปัญญาประดิษฐ์ (AI Comparative Insight)")
        run_comp_ai = st.button("📊 สร้างรายงานวิเคราะห์เปรียบเทียบข้ามสายพันธุ์")
        if run_comp_ai:
            if not api_key:
                st.error("❌ โปรดระบุ Google API Key ในแถบเมนูด้านซ้ายก่อนเริ่มการประมวลผลครับ")
            else:
                data_str = df.to_string()
                prompt = f"""
                You are a Bioinformatics expert. Analyze this comparative data table of multiple organisms:
                {data_str}
                
                Please provide a rigorous comparative analysis addressing:
                1. Which organism demonstrates higher genetic complexity or evolutionary advancement based on genome size and coding vs non-coding ratios?
                2. Identify if there's any correlation between GC content variants and environmental adaptations or lifestyle among these species.
                3. Comment on the distribution patterns of Non-coding DNA (Junk DNA).
                Answer clearly in highly formal, academic Thai language suitable for a research paper.
                """
                response_text = get_ai_response(api_key, prompt)
                st.markdown("### 📝 รายงานประเมินความแตกต่างข้ามสายพันธุ์")
                st.info(response_text)

        # 2. Interactive Charts
        st.markdown("---")
        st.markdown("#### ความสัมพันธ์เชิงโครงสร้าง: ขนาดจีโนมและสัดส่วน Non-coding DNA")
        st.caption("ℹ️ โปรดนำเมาส์ชี้ที่จุดพิกัดเพื่อดูชื่อสายพันธุ์ / สามารถใช้เมาส์เลื่อนหรือซูมขยายเพื่อดูรายละเอียด")
        
        fig_scatter = px.scatter(
            df, x="ขนาดจีโนม (bp)", y="สัดส่วน Non-coding (%)", color="ปริมาณ GC (%)", size="ขนาดจีโนม (bp)",
            hover_name="ชื่อสายพันธุ์ (Organism)", color_continuous_scale="Viridis", template="plotly_dark",
            title="Genome Size vs. Non-coding DNA Percentage"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**การประเมินสัดส่วน Non-coding DNA (%)**")
            fig_bar, ax_bar = plt.subplots()
            df_sorted = df.sort_values("สัดส่วน Non-coding (%)", ascending=True)
            ax_bar.barh(df_sorted["ชื่อสายพันธุ์ (Organism)"], df_sorted["สัดส่วน Non-coding (%)"], color="#ac3632")
            ax_bar.set_xlabel("สัดส่วน Non-coding DNA (%)")
            ax_bar.grid(axis='x', linestyle='--', alpha=0.3)
            st.pyplot(fig_bar)

        with c2:
            st.markdown("**การประเมินขนาดจีโนมโดยรวม (Total Genome Size)**")
            fig_bar2, ax_bar2 = plt.subplots()
            df_sorted_len = df.sort_values("ขนาดจีโนม (bp)", ascending=True)
            ax_bar2.barh(df_sorted_len["ชื่อสายพันธุ์ (Organism)"], df_sorted_len["ขนาดจีโนม (bp)"], color="#60a5fa") 
            ax_bar2.set_xlabel("ขนาดรวม (Base pairs)")
            ax_bar2.grid(axis='x', linestyle='--', alpha=0.3)
            st.pyplot(fig_bar2)
