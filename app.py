import streamlit as st
import pandas as pd
import numpy as np
from outlier import compute_outlier_scores, detect_numeric_columns, detect_categorical_columns, label_outliers
from utils import read_tabular, to_excel_bytes, to_parquet_bytes

st.set_page_config(page_title='Outlier Explorer', layout='wide')

st.title('Outlier Explorer — interactive outlier analysis')

uploaded = st.file_uploader('Upload CSV / Parquet / Excel', type=['csv','parquet','xlsx','xls','txt'])
use_dask = st.checkbox('Use Dask for large files (only with filesystem paths)', value=False)

if uploaded is not None:
    try:
        # `uploaded` is a file-like; pass use_dask but utils will only use dask for path strings
        df = read_tabular(uploaded, uploaded.name, use_dask=use_dask)
    except Exception as e:
        st.error(f'Failed to read file: {e}')
        st.stop()

    st.sidebar.header('Settings')
    numeric_cols = detect_numeric_columns(df)
    categorical_cols = detect_categorical_columns(df)
    st.sidebar.write(f'Numeric: {len(numeric_cols)} — Categorical: {len(categorical_cols)}')

    method = st.sidebar.selectbox('Outlier method', ['iqr','zscore'])
    show_cols = st.sidebar.multiselect('Columns to analyze', options=numeric_cols + categorical_cols if (numeric_cols+categorical_cols) else df.columns.tolist(), default=numeric_cols)
    st.sidebar.markdown('---')
    st.sidebar.write('Visual thresholds')
    iqr_mult = st.sidebar.slider('IQR multiplier (visual hint)', 0.5, 5.0, 1.5, 0.1)
    z_thresh = st.sidebar.slider('Z-score threshold (visual hint)', 0.5, 6.0, 3.0, 0.1)
    weight_num = st.sidebar.slider('Weight: numeric', 0.0, 2.0, 1.0, 0.1)
    weight_cat = st.sidebar.slider('Weight: categorical', 0.0, 2.0, 0.5, 0.1)

    scores = compute_outlier_scores(df, method=method, numeric_cols=numeric_cols, categorical_cols=categorical_cols, weight_numeric=weight_num, weight_categorical=weight_cat)
    df['_outlier_score'] = scores['combined']

    rank_n = st.sidebar.number_input('Top N outliers to show', min_value=1, max_value=10000, value=50)
    descending = st.sidebar.checkbox('Descending rank (highest scores first)', value=True)

    st.header('Preview and Filters')
    col1, col2 = st.columns([3,1])

    with col2:
        st.metric('Rows', len(df))
        st.metric('Columns', len(df.columns)-1)
        outlier_cut = st.slider('Minimum combined outlier score to mark', float(scores['combined'].min() if not scores['combined'].isnull().all() else 0.0), float(scores['combined'].max() if not scores['combined'].isnull().all() else 1.0), float(scores['combined'].quantile(0.95)), step=0.01)
        filter_outliers = st.checkbox('Filter to outliers only (score >= cut)', value=False)

    if filter_outliers:
        df_view = df[df['_outlier_score'] >= outlier_cut]
    else:
        df_view = df.copy()

    # label outliers (boolean)
    df_view['_is_outlier'] = label_outliers(df_view, score_series=df_view['_outlier_score'], threshold=outlier_cut)

    # Styling: highlight numeric cells in outlier rows
    def highlight_row(row):
        if row['_is_outlier']:
            return ['background-color: #ffe6e6' if (col in numeric_cols or col in categorical_cols) else '' for col in row.index]
        return ['' for _ in row.index]

    # Use pandas Styler
    sty = df_view.style.apply(highlight_row, axis=1)
    # emphasize outlier score column
    sty = sty.format({"_outlier_score": "{:.4f}"})

    st.dataframe(sty, use_container_width=True)

    st.header('Ranking')
    ranked = df.sort_values('_outlier_score', ascending=not descending).head(rank_n)
    st.dataframe(ranked, use_container_width=True)

    st.header('Export results')
    c1, c2 = st.columns(2)
    with c1:
        to_excel = st.button('Download filtered as Excel')
    with c2:
        to_parquet = st.button('Download filtered as Parquet')

    export_df = df_view.drop(columns=[])

    if to_excel:
        b = to_excel_bytes(export_df)
        st.download_button('Download Excel', data=b, file_name='outliers.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    if to_parquet:
        b = to_parquet_bytes(export_df)
        st.download_button('Download Parquet', data=b, file_name='outliers.parquet', mime='application/octet-stream')

    st.sidebar.markdown('---')
    st.sidebar.write('Tip: tweak weights, thresholds and columns, then export results.')

else:
    st.info('Start by uploading a dataset (CSV/Excel/Parquet). A small sample is included in the repo.')
