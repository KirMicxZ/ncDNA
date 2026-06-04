# ============================================
    # MODE A: Single File (Deep Dive)
    # ============================================
    if len(results) == 1:
        data = results[0]
        st.markdown(f"### ผลการวิเคราะห์: {data['name']}")
        st.caption(f"File: {data['filename']} | จำนวนโครโมโซมที่พบ: {data['total_chromosomes']} โครโมโซม")
        
        # 💡 [ส่วนที่เพิ่มใหม่] แผงข้อมูลสรุปภาพรวมทั้งจีโนม (Whole Genome Summary)
        st.markdown("#### 🌍 สรุปภาพรวมทั้งจีโนม (Whole Genome Summary)")
        wg1, wg2, wg3, wg4 = st.columns(4)
        wg1.metric("Total Genome Size", f"{data['len']:,} bp")
        wg2.metric("Total GC Content", f"{data['gc_total']:.2f}%")
        wg3.metric("Overall Coding DNA", f"{data['coding_pct']:.2f}%")
        wg4.metric("Overall Junk DNA", f"{data['nc_pct']:.2f}%")
        st.divider()
        # ----------------------------------------------------
        
        chrom_ids = list(data['chromosomes'].keys())
        
        if 'selected_chrom_id' not in st.session_state or st.session_state.selected_chrom_id not in chrom_ids:
            st.session_state.selected_chrom_id = chrom_ids[0]
            
        if len(chrom_ids) > 1:
            st.markdown("### 🧬 แผนผังโครโมโซม (Chromosome Map)")
            # ... (โค้ดแผนผังแท่งโครโมโซมเดิมของคุณ) ...
