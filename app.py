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
                time.sleep(3) # พัก 3 วินาทีก่อนลองใหม่
                continue
            else:
                err_msg = str(e)
                if "IncompleteRead" in err_msg or "EOF" in err_msg:
                    raise Exception("เซิร์ฟเวอร์ NCBI ตัดการเชื่อมต่อ (IncompleteRead/EOF) โปรดลองใหม่อีกครั้ง")
                raise Exception(f"เกิดข้อผิดพลาดในการดึงข้อมูล: {err_msg}")

def search_ncbi_genomes(query, email):
    Entrez.email = email
    search_term = f"({query}[Organism] OR {query}[All Fields]) AND \"latest refseq\"[filter]"
    
    # ใช้ Safe Wrapper ครอบ esearch และ esummary
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
            raise Exception(f"จีโนมนี้ประกอบด้วยชิ้นส่วนถึง {len(nucl_ids)} ชิ้น แนะนำให้ดาวน์โหลดไฟล์ .gbff จากเว็บมาอัปโหลดเองครับ")
        
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
        seq = str(record.seq).upper()
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
    if not api_key: return "⚠️ กรุณาระบุ Google API Key ในแถบเมนูด้านซ้ายเพื่อใช้งานฟีเจอร์ AI"
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
    
    st.markdown("---")
    st.subheader("🤖 AI Configuration")
    api_key = st.text_input("Google API Key", type="password", help="ใส่ API Key จาก Google AI Studio เพื่อวิเคราะห์ด้วย AI")
    
    st.markdown("---")
    st.subheader("🌐 ดึงข้อมูลจาก NCBI")
    ncbi_email = st.text_input("Email (จำเป็นสำหรับ NCBI)", placeholder="email@example.com")
    
    tab1, tab2 = st.tabs(["🔍 ค้นหาชื่อ", "📝 ระบุ ID โดยตรง"])
    
    with tab1:
        st.write("ค้นหาชื่อสิ่งมีชีวิตในฐานข้อมูล Assembly")
        search_query = st.text_input("พิมพ์ชื่อ (เช่น Yeast, E. coli)", key="search_q")
        
        if st.button("🔍 ค้นหาข้อมูล"):
            if not ncbi_email:
                st.warning("⚠️ กรุณากรอก Email ด้านบนก่อนทำการค้นหาครับ")
            elif search_query:
                with st.spinner(f"กำลังค้นหา '{search_query}'..."):
                    try:
                        results = search_ncbi_genomes(search_query, ncbi_email)
                        st.session_state['ncbi_search_results'] = results
                    except Exception as e:
                        st.error(f"เกิดข้อผิดพลาด: {e}")
        
        if st.session_state.get('ncbi_search_results') is not None:
            res_list = st.session_state['ncbi_search_results']
            if len(res_list) > 0:
                options = {r['id']: r['label'] for r in res_list}
                selected_acc = st.selectbox("พบข้อมูล โปรดเลือกจีโนม:", options=list(options.keys()), format_func=lambda x: options[x])
                
                if st.button("📥 ดาวน์โหลดจีโนมที่เลือก"):
                    with st.spinner(f"กำลังดาวน์โหลดและประมวลผล {selected_acc} (ทยอยโหลดแบบปลอดภัย อาจใช้เวลาสักครู่)..."):
                        try:
                            raw_data = fetch_ncbi(selected_acc, ncbi_email)
                            if not any(item['id'] == selected_acc for item in st.session_state['ncbi_cache']):
                                st.session_state['ncbi_cache'].append({
                                    "id": selected_acc,
                                    "filename": f"NCBI_{selected_acc}.gbff",
                                    "content": raw_data
                                })
                            st.success(f"ดาวน์โหลด {selected_acc} สำเร็จ!")
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))
            else:
                st.info("❌ ไม่พบชื่อจีโนมนี้ในฐานข้อมูล NCBI ลองใช้คำอื่นดูครับ")

    with tab2:
        st.write("ดึงข้อมูลหากทราบรหัส RefSeq / GenBank อยู่แล้ว")
        ncbi_id = st.text_input("RefSeq ID", placeholder="เช่น NC_000913 หรือ GCF_000146045.2", key="manual_id")
        
        if st.button("📥 ดาวน์โหลดด้วย ID"):
            if not ncbi_email or not ncbi_id:
                st.error("กรุณากรอก Email และ RefSeq ID ให้ครบถ้วน")
            else:
                with st.spinner(f"กำลังดาวน์โหลดและประมวลผล {ncbi_id} (ทยอยโหลดแบบปลอดภัย อาจใช้เวลาสักครู่)..."):
                    try:
                        raw_data = fetch_ncbi(ncbi_id.strip(), ncbi_email.strip())
                        if not any(item['id'] == ncbi_id for item in st.session_state['ncbi_cache']):
                            st.session_state
