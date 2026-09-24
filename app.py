import streamlit as st
from Bio import SeqIO, Entrez
from Bio.Seq import Seq
from Bio.Seq import UndefinedSequenceError
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import google.generativeai as genai
import time
import json
import urllib.parse
from scipy.cluster.hierarchy import linkage, dendrogram

# ============================================
# 1. Page Configuration & Custom CSS
# ============================================
st.set_page_config(page_title="Genome Analyzer Pro", layout="wide", page_icon="🧬")

plt.style.use('dark_background')

st.markdown("""
<style>
    .stApp { background-color: #111827; color: #F9FAFB; }
    [data-testid="stSidebar"] { background-color: #1F2937; }
    h1, h2, h3, .main-header { color: #FFFFFF !important; font-family: 'Inter', sans-serif; }
    .sub-header { color: #9CA3AF !important; font-size: 1.1rem; }
    [data-testid="stMetricValue"] { color: #10B981 !important; font-weight: bold; } 
    [data-testid="stMetricLabel"] { color: #D1D5DB !important; }
    .stButton button { width: 100%; border-radius: 8px; background-color: #374151; color: white; border: none; }
    .stButton button:hover { background-color: #4B5563; }
    .chat-bubble-user { background-color: #1E3A8A; padding: 10px 14px; border-radius: 12px; margin-bottom: 8px; }
    .chat-bubble-ai { background-color: #374151; padding: 10px 14px; border-radius: 12px; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

# Initialize Session States
if 'ncbi_cache' not in st.session_state:
    st.session_state['ncbi_cache'] = []
if 'ncbi_search_results' not in st.session_state:
    st.session_state['ncbi_search_results'] = None
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'parsed_results' not in st.session_state:
    st.session_state['parsed_results'] = []
if 'processed_sources' not in st.session_state:
    st.session_state['processed_sources'] = set()

# ============================================
# Helper: Convert Integer to Roman Numerals
# ============================================
def int_to_roman(num):
    val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    syb = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
    roman_num = ''
    i = 0
    while num > 0:
        for _ in range(num // val[i]):
            roman_num += syb[i]
            num -= val[i]
        i += 1
    return roman_num

# ============================================
# 2. Advanced Bioinformatic Logic & Calculations
# ============================================

def calculate_gc(sequence):
    if not sequence: return 0
    return (sequence.count("G") + sequence.count("C")) / len(sequence) * 100

def calculate_gc_skew(sequence):
    if not sequence: return 0
    g = sequence.count("G")
    c = sequence.count("C")
    if (g + c) == 0: return 0
    return (g - c) / (g + c)

def find_orfs(sequence, min_aa_len=100):
    seq_obj = Seq(sequence)
    seq_len = len(sequence)
    orfs = []
    
    # 6 Reading Frames
    frames = []
    for frame in range(3):
        frames.append((frame, str(seq_obj[frame:]), "+"))
    rc_seq = str(seq_obj.reverse_complement())
    for frame in range(3):
        frames.append((frame, rc_seq[frame:], "-"))
        
    start_codon = "ATG"
    stop_codons = {"TAA", "TAG", "TGA"}

    for frame_idx, frame_seq, strand in frames:
        i = 0
        while i < len(frame_seq) - 2:
            codon = frame_seq[i:i+3]
            if codon == start_codon:
                for j in range(i + 3, len(frame_seq) - 2, 3):
                    stop = frame_seq[j:j+3]
                    if stop in stop_codons:
                        aa_len = (j + 3 - i) // 3
                        if aa_len >= min_aa_len:
                            if strand == "+":
                                start_pos = frame_idx + i
                                end_pos = frame_idx + j + 3
                            else:
                                start_pos = seq_len - (frame_idx + j + 3)
                                end_pos = seq_len - (frame_idx + i)
                            orfs.append({
                                "Strand": strand,
                                "Start": start_pos,
                                "End": end_pos,
                                "Length (bp)": end_pos - start_pos,
                                "Protein Length (aa)": aa_len,
                                "Frame": frame_idx + 1
                            })
                        break
            i += 3
    return pd.DataFrame(orfs)

def calculate_rscu(cds_sequences):
    codon_counts = {}
    synonymous_codons = {
        'A': ['GCT', 'GCC', 'GCA', 'GCG'],
        'C': ['TGT', 'TGC'],
        'D': ['GAT', 'GAC'],
        'E': ['GAA', 'GAG'],
        'F': ['TTT', 'TTC'],
        'G': ['GGT', 'GGC', 'GGA', 'GGG'],
        'H': ['CAT', 'CAC'],
        'I': ['ATT', 'ATC', 'ATA'],
        'K': ['AAA', 'AAG'],
        'L': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'],
        'M': ['ATG'],
        'N': ['AAT', 'AAC'],
        'P': ['CCT', 'CCC', 'CCA', 'CCG'],
        'Q': ['CAA', 'CAG'],
        'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
        'S': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'],
        'T': ['ACT', 'ACC', 'ACA', 'ACG'],
        'V': ['GTT', 'GTC', 'GTA', 'GTG'],
        'W': ['TGG'],
        'Y': ['TAT', 'TAC']
    }
    
    for aa, codons in synonymous_codons.items():
        for c in codons: codon_counts[c] = 0

    total_codons = 0
    for seq in cds_sequences:
        for i in range(0, len(seq) - 2, 3):
            codon = seq[i:i+3].upper()
            if codon in codon_counts:
                codon_counts[codon] += 1
                total_codons += 1

    rscu_data = []
    for aa, codons in synonymous_codons.items():
        n_i = len(codons)
        total_aa_count = sum(codon_counts[c] for c in codons)
        for c in codons:
            count = codon_counts[c]
            rscu = (count / (total_aa_count / n_i)) if total_aa_count > 0 else 0
            rscu_data.append({"Amino Acid": aa, "Codon": c, "Count": count, "RSCU": round(rscu, 3)})

    return pd.DataFrame(rscu_data)

def generate_kmer_profile(seq, k=3):
    kmers = {}
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        if "N" not in kmer:
            kmers[kmer] = kmers.get(kmer, 0) + 1
    total = sum(kmers.values())
    return {kmer: count/total for kmer, count in kmers.items()} if total > 0 else {}

def safe_ncbi_call(func, max_retries=5, is_fetch=False, **kwargs):
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
                time.sleep(2)
                continue
            else:
                raise Exception(f"Error fetching NCBI data: {str(e)}")

def fetch_ncbi(acc_id, email):
    Entrez.email = email
    acc_id = acc_id.strip().upper()

    if acc_id.startswith("GCF_") or acc_id.startswith("GCA_"):
        search_rec = safe_ncbi_call(Entrez.esearch, db="assembly", term=acc_id)
        if not search_rec["IdList"]:
            raise Exception(f"Assembly not found: {acc_id}")
        assembly_id = search_rec["IdList"][0]
        link_rec = safe_ncbi_call(Entrez.elink, dbfrom="assembly", db="nucleotide", id=assembly_id)
        if not link_rec[0].get("LinkSetDb"):
            raise Exception(f"No nucleotide data linked to Assembly: {acc_id}")
        nucl_ids = [link["Id"] for link in link_rec[0]["LinkSetDb"][0]["Link"]]
        
        all_data = ""
        batch_size = 5 
        for i in range(0, min(len(nucl_ids), 20), batch_size):
            batch_ids = nucl_ids[i:i+batch_size]
            id_string = ",".join(batch_ids)
            all_data += safe_ncbi_call(Entrez.efetch, is_fetch=True, db="nucleotide", id=id_string, rettype="gbwithparts", retmode="text")
            time.sleep(0.5)
        return all_data
    else:
        return safe_ncbi_call(Entrez.efetch, is_fetch=True, db="nucleotide", id=acc_id, rettype="gbwithparts", retmode="text")

def parse_file_content(file_content, filename):
    records = []
    file_type = "genbank"
    
    if filename.endswith(".fasta") or filename.endswith(".fa"):
        file_type = "fasta"
    
    try:
        records = list(SeqIO.parse(io.StringIO(file_content), file_type))
        if not records and file_type == "genbank":
            records = list(SeqIO.parse(io.StringIO(file_content), "fasta"))
            file_type = "fasta"
    except Exception as e:
        return None, f"Error parsing {filename}: {str(e)}"

    if not records:
        return None, f"No sequences found in {filename}"

    chromosomes_data = {}
    total_len = 0
    total_coding_len = 0
    total_gc = 0
    
    num_records = len(records)
    
    for idx, record in enumerate(records, start=1):
        try:
            seq = str(record.seq).upper()
        except UndefinedSequenceError:
            seq = "N" * len(record)
        
        slen = len(seq)
        total_len += slen
        total_gc += (seq.count("G") + seq.count("C"))
        
        cds_regions = []
        features_list = []
        protein_seqs = []  
        cds_sequences = []

        if file_type == "genbank":
            for f in record.features:
                feat_type = f.type
                start, end = int(f.location.start), int(f.location.end)
                strand = f.location.strand
                gene_name = f.qualifiers.get('gene', f.qualifiers.get('locus_tag', [feat_type]))[0]
                
                features_list.append({
                    "type": feat_type, "start": start, "end": end, 
                    "strand": strand, "name": gene_name
                })
                
                if feat_type == "CDS":
                    cds_regions.append((start, end))
                    cds_seq = seq[start:end]
                    cds_sequences.append(cds_seq)
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

        all_proteins = "".join(protein_seqs)
        aa_list = list("ACDEFGHIKLMNPQRSTVWY")
        aa_dist = {aa: all_proteins.count(aa) for aa in aa_list} if all_proteins else {}

        if num_records > 1:
            roman_idx = int_to_roman(idx)
            chrom_key = f"Chromosome {roman_idx} ({record.id})"
        else:
            chrom_key = record.id

        chromosomes_data[chrom_key] = {
            "id": record.id,
            "desc": record.description,
            "len": slen,
            "seq": seq,
            "features": features_list,
            "cds_regions": cds_regions,
            "cds_seqs": cds_sequences,
            "coding_pct": coding_pct,
            "nc_pct": nc_pct,
            "intergenic_seqs": intergenic_seqs,
            "gc_total": calculate_gc(seq),
            "gc_skew": calculate_gc_skew(seq),
            "aa_dist": aa_dist,
            "total_proteins": len(protein_seqs),
            "kmer_profile": generate_kmer_profile(seq, k=3)
        }

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
    if not api_key: return "Please enter your Google API Key in the left sidebar."
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash') 
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Error: {str(e)}"

# ============================================
# 3. Sidebar: Input & File Controls
# ============================================
with st.sidebar:
    st.title("Genome Analyzer")
    st.caption("Advanced Bioinformatics & AI Suite")
    st.markdown("---")
    
    st.subheader("1. NCBI Database Import")
    ncbi_email = st.text_input("Email (Required)", placeholder="researcher@lab.org")
    
    tab_s1, tab_s2 = st.tabs(["Search Organism", "Accession ID"])
    with tab_s1:
        s_query = st.text_input("Organism Name", placeholder="e.g., Escherichia coli")
        if st.button("Search NCBI"):
            if ncbi_email and s_query:
                with st.spinner("Searching NCBI..."):
                    try:
                        Entrez.email = ncbi_email
                        rec = safe_ncbi_call(Entrez.esearch, db="assembly", term=f"{s_query}[Organism] AND \"latest refseq\"[filter]", retmax=5)
                        ids = rec.get("IdList", [])
                        if ids:
                            sums = safe_ncbi_call(Entrez.esummary, db="assembly", id=",".join(ids))
                            doc_sums = sums.get('DocumentSummarySet', {}).get('DocumentSummary', [])
                            st.session_state['ncbi_search_results'] = [
                                {"id": d.get('AssemblyAccession'), "label": f"{d.get('SpeciesName')} ({d.get('AssemblyAccession')})"}
                                for d in doc_sums
                            ]
                    except Exception as e: st.error(str(e))
        
        if st.session_state.get('ncbi_search_results'):
            opts = {x['id']: x['label'] for x in st.session_state['ncbi_search_results']}
            sel_acc = st.selectbox("Select Result", list(opts.keys()), format_func=lambda x: opts[x])
            if st.button("Import Selected Genome"):
                with st.spinner(f"Fetching {sel_acc}..."):
                    try:
                        raw = fetch_ncbi(sel_acc, ncbi_email)
                        st.session_state['ncbi_cache'].append({"id": sel_acc, "filename": f"{sel_acc}.gbff", "content": raw})
                        st.success("Successfully imported!")
                        st.rerun()
                    except Exception as e: st.error(str(e))

    with tab_s2:
        acc_manual = st.text_input("Accession Code", placeholder="e.g., NC_000913")
        if st.button("Fetch Code"):
            if ncbi_email and acc_manual:
                with st.spinner("Fetching NCBI..."):
                    try:
                        raw = fetch_ncbi(acc_manual, ncbi_email)
                        st.session_state['ncbi_cache'].append({"id": acc_manual, "filename": f"{acc_manual}.gbff", "content": raw})
                        st.success("Loaded!")
                        st.rerun()
                    except Exception as e: st.error(str(e))

    st.markdown("---")
    st.subheader("2. Upload Files")
    uploaded_files = st.file_uploader("Upload .gbff, .gb, .fasta, .fa files", type=["gbff", "gb", "gbk", "fasta", "fa"], accept_multiple_files=True)
    
    if uploaded_files:
        for uf in uploaded_files:
            source_id = f"file_{uf.name}_{uf.size}"
            if source_id not in st.session_state['processed_sources']:
                content = uf.getvalue().decode("utf-8", errors="ignore")
                data, err = parse_file_content(content, uf.name)
                if data:
                    st.session_state['parsed_results'].append(data)
                    st.session_state['processed_sources'].add(source_id)
                elif err:
                    st.error(err)

    if st.session_state['ncbi_cache']:
        for item in st.session_state['ncbi_cache']:
            source_id = f"ncbi_{item['id']}"
            if source_id not in st.session_state['processed_sources']:
                data, err = parse_file_content(item['content'], item['filename'])
                if data:
                    st.session_state['parsed_results'].append(data)
                    st.session_state['processed_sources'].add(source_id)
                elif err:
                    st.error(err)

    if st.session_state['parsed_results']:
        st.markdown("---")
        st.subheader("✏️ Edit Uploaded Organisms")
        to_delete = []
        for idx, org in enumerate(st.session_state['parsed_results']):
            col_name, col_del = st.columns([4, 1])
            with col_name:
                new_name = st.text_input(f"Organism #{idx+1}", value=org['name'], key=f"org_name_{idx}")
                st.session_state['parsed_results'][idx]['name'] = new_name
            with col_del:
                st.write("")
                st.write("")
                if st.button("🗑️", key=f"del_org_{idx}"):
                    to_delete.append(idx)
        
        if to_delete:
            for d_idx in sorted(to_delete, reverse=True):
                st.session_state['parsed_results'].pop(d_idx)
            st.rerun()

    st.markdown("---")
    st.subheader("3. AI Key Configuration")
    api_key = st.text_input("Google AI Studio Key", type="password")

# ============================================
# 4. Main Application Interface
# ============================================
st.markdown('<h1 class="main-header">Genome Analysis & AI Workspace</h1>', unsafe_allow_html=True)

results = st.session_state['parsed_results']

if not results:
    st.info("👈 Please upload genome data files or search NCBI in the sidebar to start analysis.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Linear Feature Tracks", "Active")
    c2.metric("GC Skew & Codon RSCU", "Active")
    c3.metric("AI Multi-turn Chatbot", "Ready")
else:
    tab_single, tab_comp, tab_ai, tab_export = st.tabs([
        "🔬 Single Genome Analysis", 
        "📊 Comparative Genomics & Synteny", 
        "🤖 AI Interactive Assistant", 
        "📥 Data & Report Export"
    ])

    # ============================================
    # TAB 1: Single Genome Deep Dive
    # ============================================
    with tab_single:
        selected_organism = st.selectbox("Select Genome Sample", options=[r['name'] for r in results], key="s_org")
        data = next(r for r in results if r['name'] == selected_organism)
        
        st.subheader(f"Genome Analysis: {data['name']}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Genome Size", f"{data['len']:,} bp")
        m2.metric("GC Content", f"{data['gc_total']:.2f}%")
        m3.metric("Coding Ratio (CDS)", f"{data['coding_pct']:.2f}%")
        m4.metric("Non-coding Ratio", f"{data['nc_pct']:.2f}%")
        
        chrom_ids = list(data['chromosomes'].keys())
        chrom_options = ["All Chromosomes (Combined)"] + chrom_ids if len(chrom_ids) > 1 else chrom_ids
        selected_chrom_id = st.selectbox("Select Sequence / Chromosome", chrom_options)
        
        if selected_chrom_id == "All Chromosomes (Combined)":
            combined_seq = "".join([c['seq'] for c in data['chromosomes'].values()])
            combined_cds_seqs = [cds for c in data['chromosomes'].values() for cds in c['cds_seqs']]
            combined_features = []
            
            offset = 0
            for cid, cinfo in data['chromosomes'].items():
                for f in cinfo['features']:
                    f_copy = f.copy()
                    f_copy['start'] += offset
                    f_copy['end'] += offset
                    f_copy['chrom'] = cid
                    combined_features.append(f_copy)
                offset += cinfo['len']
            
            c_data = {
                'id': 'ALL',
                'desc': 'All Chromosomes Combined',
                'len': len(combined_seq),
                'seq': combined_seq,
                'features': combined_features,
                'cds_seqs': combined_cds_seqs,
                'is_combined': True
            }
        else:
            c_data = data['chromosomes'][selected_chrom_id]
            c_data['is_combined'] = False
        
        st.markdown("---")
        st.markdown("### 1. Interactive Genome & Feature Track Browser")
        
        features = c_data['features']
        if features:
            df_feat = pd.DataFrame(features)
            fig_track = go.Figure()
            
            fig_track.add_trace(go.Scatter(
                x=[0, c_data['len']], y=[0, 0], mode='lines',
                line=dict(color='#6B7280', width=4), hoverinfo='none', name='Genome'
            ))
            
            feat_subset = df_feat.head(200)
            for idx, row in feat_subset.iterrows():
                y_pos = 1 if row['strand'] == 1 else -1
                color = '#10B981' if row['type'] == 'CDS' else '#F59E0B'
                chrom_label = f"<br>Chr: {row['chrom']}" if 'chrom' in row else ""
                fig_track.add_trace(go.Scatter(
                    x=[row['start'], row['end']], y=[y_pos, y_pos],
                    mode='lines+markers', line=dict(color=color, width=8),
                    name=row['type'],
                    hovertemplate=f"Feature: {row['name']}<br>Type: {row['type']}{chrom_label}<br>Span: {row['start']:,} - {row['end']:,} bp<extra></extra>"
                ))
            
            fig_track.update_layout(
                title="Linear Gene Map (Green: CDS | Yellow: RNA/Other Features)",
                xaxis_title="Position (bp)", yaxis=dict(showticklabels=False, range=[-2, 2]),
                template="plotly_dark", height=250, showlegend=False,
                xaxis=dict(rangeslider=dict(visible=True))
            )
            st.plotly_chart(fig_track, use_container_width=True)
        else:
            st.info("No feature annotation data available for linear track visualization.")

        st.markdown("### 2. GC Content & GC Skew Sliding Window")
        win_size = st.slider("Window Size (bp)", min_value=500, max_value=50000, value=2000, step=500)
        
        seq = c_data['seq']
        positions, gc_vals, skew_vals = [], [], []
        
        # ปรับ range ให้ครอบคลุมอย่างน้อย 1 window
        step = max(1, win_size)
        for i in range(0, len(seq) - win_size + 1, step):
            sub = seq[i:i+win_size]
            positions.append(i)
            gc_vals.append(calculate_gc(sub))
            skew_vals.append(calculate_gc_skew(sub))

        if positions:
            fig_skew = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("GC Content (%)", "GC Skew (G-C)/(G+C)"))
            fig_skew.add_trace(go.Scatter(x=positions, y=gc_vals, line=dict(color='#818CF8')), row=1, col=1)
            fig_skew.add_trace(go.Scatter(x=positions, y=skew_vals, line=dict(color='#34D399')), row=2, col=1)
            fig_skew.update_layout(template="plotly_dark", height=400, showlegend=False)
            st.plotly_chart(fig_skew, use_container_width=True)
        else:
            st.warning("Sequence length is shorter than the selected window size. Please decrease the window size.")

        st.markdown("### 3. Whole Sequence Polar View (Circos-style)")
        polar_df = pd.DataFrame({'Position': positions, 'GC': gc_vals, 'Skew': skew_vals})
        
        # เช็คว่ามีข้อมูลก่อนวาด Polar Plot
        if not polar_df.empty and c_data['len'] > 0:
            polar_df_plot = polar_df.copy()
            # แปลงตำแหน่ง Base Pair ให้เป็นมุม 0 - 360 องศา
            polar_df_plot['Angle'] = (polar_df_plot['Position'] / c_data['len']) * 360
            
            fig_polar = px.line_polar(
                polar_df_plot, 
                r="GC", 
                theta="Angle", 
                template="plotly_dark", 
                color_discrete_sequence=['#F43F5E']
            )
            fig_polar.update_layout(
                height=450,
                polar=dict(
                    angularaxis=dict(
                        direction="clockwise",  
                        rotation=90,            
                        tickmode='array',
                        tickvals=[0, 90, 180, 270],
                        ticktext=['0%', '25%', '50%', '75%']
                    )
                )
            )
            st.plotly_chart(fig_polar, use_container_width=True)
        else:
            st.info("💡 Insufficient data to render Polar View. Try reducing the Window Size slider above.")

        st.markdown("---")
        st.markdown("### 6. External BLAST Action")
        blast_seq = c_data['seq'][:500]
        blast_url = f"https://blast.ncbi.nlm.nih.gov/Blast.cgi?QUERY={urllib.parse.quote(blast_seq)}&PROGRAM=blastn&DATABASE=nr&CMD=Put"
        st.markdown(f'<a href="{blast_url}" target="_blank"><button style="padding:10px; background-color:#2563EB; color:white; border-radius:8px; border:none; cursor:pointer;">🚀 Send First 500bp to NCBI BLASTn</button></a>', unsafe_allow_html=True)

    # ============================================
    # TAB 2: Comparative Genomics & Synteny
    # ============================================
    with tab_comp:
        if len(results) < 2:
            st.warning("Please upload or import at least 2 genome samples to enable comparative analysis.")
        else:
            st.subheader("Inter-Species Comparative Dashboard")
            
            comp_df = pd.DataFrame([
                {
                    "Organism": r['name'],
                    "Size (bp)": r['len'],
                    "GC%": r['gc_total'],
                    "Coding%": r['coding_pct'],
                    "Chromosomes": r['total_chromosomes']
                } for r in results
            ])
            st.dataframe(comp_df, use_container_width=True)

            st.markdown("---")
            col_synt, col_phylo = st.columns(2)
            
            with col_synt:
                st.markdown("### 1. Synteny Dotplot Matrix")
                org1 = st.selectbox("Genome A", [r['name'] for r in results], index=0)
                org2 = st.selectbox("Genome B", [r['name'] for r in results], index=min(1, len(results)-1))
                
                seq1 = list(next(r for r in results if r['name'] == org1)['chromosomes'].values())[0]['seq'][:5000]
                seq2 = list(next(r for r in results if r['name'] == org2)['chromosomes'].values())[0]['seq'][:5000]
                
                if st.button("Generate Dotplot"):
                    k = 10
                    matches_x, matches_y = [], []
                    kmers1 = {seq1[i:i+k]: i for i in range(len(seq1)-k)}
                    for j in range(len(seq2)-k):
                        kmer = seq2[j:j+k]
                        if kmer in kmers1:
                            matches_x.append(kmers1[kmer])
                            matches_y.append(j)
                    
                    fig_dot = go.Figure(data=go.Scatter(x=matches_x, y=matches_y, mode='markers', marker=dict(size=3, color='#818CF8')))
                    fig_dot.update_layout(title="Dotplot Synteny Match", xaxis_title=org1, yaxis_title=org2, template="plotly_dark", height=400)
                    st.plotly_chart(fig_dot, use_container_width=True)

            with col_phylo:
                st.markdown("### 2. K-mer Distance Clustering (Dendrogram)")
                if len(results) >= 2:
                    kmer_profiles = []
                    names = [r['name'] for r in results]
                    for r in results:
                        first_seq = list(r['chromosomes'].values())[0]['seq']
                        kmer_profiles.append(generate_kmer_profile(first_seq, k=3))
                    
                    df_kmers = pd.DataFrame(kmer_profiles).fillna(0)
                    dist_matrix = np.corrcoef(df_kmers)
                    
                    fig_phy, ax_phy = plt.subplots(figsize=(6, 4))
                    linked = linkage(dist_matrix, 'single')
                    dendrogram(linked, labels=names, orientation='top', ax=ax_phy)
                    ax_phy.set_title("Genetic Proximity Dendrogram")
                    st.pyplot(fig_phy)

    # ============================================
    # TAB 3: Interactive AI Assistant
    # ============================================
    with tab_ai:
        st.subheader("🤖 AI Genomics Assistant")
        st.caption("Ask continuous questions or execute quick preset biological prompts.")
        
        p_col1, p_col2, p_col3 = st.columns(3)
        preset_prompt = None
        if p_col1.button("Analyze Horizontal Gene Transfer"):
            preset_prompt = "Examine the GC Skew and GC Content variations to identify potential horizontal gene transfer regions."
        if p_col2.button("Assess Thermal/Environmental Adaptation"):
            preset_prompt = "Evaluate GC Content and Amino Acid usage bias to infer thermal or environmental adaptations."
        if p_col3.button("Comparative Evolutionary Summary"):
            preset_prompt = "Summarize the key evolutionary trade-offs between coding density and non-coding regions across loaded genomes."

        for msg in st.session_state['chat_history']:
            role_class = "chat-bubble-user" if msg['role'] == 'user' else "chat-bubble-ai"
            st.markdown(f'<div class="{role_class}"><b>{msg["role"].capitalize()}:</b> {msg["content"]}</div>', unsafe_allow_html=True)

        user_input = st.chat_input("Ask any question about your genomic dataset...")
        prompt_to_run = user_input or preset_prompt

        if prompt_to_run:
            st.session_state['chat_history'].append({"role": "user", "content": prompt_to_run})
            
            context = f"Loaded Samples Count: {len(results)}\n"
            for r in results:
                context += f"- Organism: {r['name']}, Size: {r['len']}bp, GC: {r['gc_total']:.2f}%, Coding: {r['coding_pct']:.2f}%\n"

            full_prompt = f"Context Data:\n{context}\nUser Question: {prompt_to_run}\nProvide a rigorous scientific response."
            
            with st.spinner("AI is thinking..."):
                ai_response = get_ai_response(api_key, full_prompt)
                st.session_state['chat_history'].append({"role": "assistant", "content": ai_response})
                st.rerun()

    # ============================================
    # TAB 4: Data & Report Export
    # ============================================
    with tab_export:
        st.subheader("📥 Data & Report Export")
        st.caption("Download summary statistics and analytical outputs in CSV or JSON formats.")
        
        export_list = []
        for r in results:
            export_list.append({
                "Organism Name": r['name'],
                "Filename": r['filename'],
                "Total Chromosomes": r['total_chromosomes'],
                "Genome Length (bp)": r['len'],
                "GC Content (%)": round(r['gc_total'], 2),
                "Coding Ratio (%)": round(r['coding_pct'], 2),
                "Non-coding Ratio (%)": round(r['nc_pct'], 2)
            })
        
        df_export = pd.DataFrame(export_list)
        st.markdown("### Summary Report Table")
        st.dataframe(df_export, use_container_width=True)
        
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv_data = df_export.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📄 Download Summary Table (CSV)",
                data=csv_data,
                file_name="genome_summary_report.csv",
                mime="text/csv"
            )
        
        with col_dl2:
            json_data = json.dumps(export_list, indent=4)
            st.download_button(
                label="📦 Download Summary JSON",
                data=json_data,
                file_name="genome_summary_report.json",
                mime="application/json"
            )
