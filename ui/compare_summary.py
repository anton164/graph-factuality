import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path

st.set_page_config(layout="wide")

st.title('Comparison of Summaries')

ref_loc = st.text_input("Location of reference dataset", "results/refs")
summary_1_loc = st.text_input("Location of generated summary", "results/output-us")
summary_2_loc = st.text_input("Location of generated summary (second)", "results/output-them")

all_files = os.listdir(Path(ref_loc))

file_selected = st.selectbox("Select file to compare", all_files)
file_number = file_selected.strip(".ref")

col1, col2, col3 = st.columns(3)

with col1:
    st.header("Reference")
    with open(ref_loc + "/" + file_selected, 'r') as f:
        ref_text = f.read()
    st.write(ref_text)

with col2:
    st.header("Summary 1")
    load_sum_1 = st.checkbox('Load summary')
    if load_sum_1:
        with open(summary_1_loc + "/" + file_number + ".dec", 'r') as f:
            sum_text = f.read()
        st.write(sum_text)

with col3:
    st.header("Summary 2")
    load_sum_2 = st.checkbox('Load summary 2')
    if load_sum_2:
        with open(summary_2_loc + "/" + file_number + ".dec", 'r') as f:
            sum_text = f.read()
        st.write(sum_text)