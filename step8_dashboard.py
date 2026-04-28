"""
Job Recommendation System Dashboard

Run with:
    streamlit run step8_dashboard.py
"""

import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Job Recommendation Dashboard | Gamage Recruiters",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1565C0;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #546E7A;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f0f4ff;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #1565C0;
    }
    .skill-tag {
        display: inline-block;
        background: #e3f2fd;
        color: #1565C0;
        border-radius: 12px;
        padding: 2px 10px;
        margin: 2px;
        font-size: 0.82rem;
        font-weight: 500;
    }
    .rank-badge {
        background: #1565C0;
        color: white;
        border-radius: 50%;
        width: 28px;
        height: 28px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .section-divider {
        border-top: 2px solid #e0e0e0;
        margin: 1.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING 
# ══════════════════════════════════════════════════════════════════════════════

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
OUTPUTS = os.path.join(BASE, 'outputs')
MODELS  = os.path.join(BASE, 'models')


@st.cache_data(show_spinner="Loading data…")
def load_data():
    df_c    = pd.read_csv(os.path.join(DATA, 'candidates_preprocessed.csv'))
    df_j    = pd.read_csv(os.path.join(DATA, 'job_postings_preprocessed.csv'))
    df_recs = pd.read_csv(os.path.join(OUTPUTS, 'step6_recommendations.csv'))

    # Evaluation results (produced by Step 7)
    eval_path = os.path.join(OUTPUTS, 'step7_eval_results.json')
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            eval_results = json.load(f)
    else:
        eval_results = None

    return df_c, df_j, df_recs, eval_results


@st.cache_data(show_spinner="Loading score matrices…")
def load_matrices():
    matrices = {}
    names = {
        'hybrid':    'step6_hybrid_score_matrix.npy',
        'tfidf':     'step6_tfidf_cosine_matrix.npy',
        'w2v':       'step6_w2v_cosine_matrix.npy',
        'euclidean': 'step6_euc_sim_matrix.npy',
        'pearson':   'step6_pearson_sim_matrix.npy',
    }
    for key, fname in names.items():
        path = os.path.join(MODELS, fname)
        if os.path.exists(path):
            matrices[key] = np.load(path)
    return matrices


df_c, df_j, df_recs, eval_results = load_data()
score_matrices = load_matrices()

METRIC_LABELS = {
    'hybrid':    'Hybrid Score',
    'tfidf':     'TF-IDF Cosine',
    'w2v':       'Word2Vec Cosine',
    'euclidean': 'Euclidean Sim',
    'pearson':   'Pearson Sim',
}
METRIC_COLORS = {
    'hybrid':    '#1565C0',
    'tfidf':     '#2196F3',
    'w2v':       '#9C27B0',
    'euclidean': '#4CAF50',
    'pearson':   '#FF9800',
}

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_recommendations(candidate_id: str, metric: str = 'hybrid', top_n: int = 5) -> pd.DataFrame:
    """Return top-N job recommendations for a candidate using the chosen metric."""
    if metric not in score_matrices:
        return df_recs[df_recs['candidate_id'] == candidate_id].head(top_n)

    cand_row = df_c[df_c['candidate_id'] == candidate_id]
    if cand_row.empty:
        return pd.DataFrame()

    i = cand_row.index[0]
    scores = score_matrices[metric][i]
    top_idx = np.argsort(scores)[::-1][:top_n]

    rows = []
    for rank, idx in enumerate(top_idx, 1):
        job = df_j.iloc[idx]
        # Skill overlap
        try:
            cand_skills = set(s.strip().lower() for s in str(cand_row.iloc[0]['skills']).split(','))
            job_skills  = set(s.strip().lower() for s in str(job['required_skills']).split(','))
            overlap_pct = round(len(cand_skills & job_skills) / max(len(job_skills), 1) * 100, 1)
        except Exception:
            overlap_pct = 0.0

        rows.append({
            'Rank':            rank,
            'job_id':          job.get('job_id', ''),
            'Job Title':       job.get('job_title', ''),
            'Company':         job.get('company_name', ''),
            'Industry':        job.get('industry', ''),
            'Location':        job.get('location', ''),
            'Level':           job.get('experience_level', ''),
            'Salary (LKR)':    f"LKR {int(job.get('salary_min_lkr', 0)):,}–{int(job.get('salary_max_lkr', 0)):,}",
            'Required Skills': job.get('required_skills', ''),
            'Score':           round(float(scores[idx]), 4),
            'Skill Overlap %': overlap_pct,
        })
    return pd.DataFrame(rows)


def render_skill_tags(skills_str: str) -> str:
    skills = [s.strip() for s in str(skills_str).split(',') if s.strip()]
    return ' '.join(f'<span class="skill-tag">{s}</span>' for s in skills)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/office-worker.png", width=60)
    st.markdown("### Gamage Recruiters")
    st.markdown("**Job Recommendation System**")
    st.markdown("---")

    active_tab = st.radio(
        "Navigate",
        ["Candidate Explorer", "Model Evaluation", "Bulk Recommendations", "About"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Filters**")

    domain_filter = st.multiselect(
        "Domain / Industry",
        options=sorted(df_c['primary_domain'].dropna().unique()),
        default=[],
        placeholder="All domains",
    )
    level_filter = st.multiselect(
        "Experience Level",
        options=sorted(df_c['experience_level'].dropna().unique()),
        default=[],
        placeholder="All levels",
    )
    location_filter = st.multiselect(
        "Location",
        options=sorted(df_c['location'].dropna().unique()),
        default=[],
        placeholder="All locations",
    )

    st.markdown("---")
    st.markdown("**Recommendation Settings**")
    selected_metric = st.selectbox(
        "Similarity Metric",
        options=list(METRIC_LABELS.keys()),
        format_func=lambda x: METRIC_LABELS[x],
        index=0,
    )
    top_n = st.slider("Top N Recommendations", min_value=1, max_value=20, value=5)

# ══════════════════════════════════════════════════════════════════════════════
# APPLY SIDEBAR FILTERS
# ══════════════════════════════════════════════════════════════════════════════

df_filtered = df_c.copy()
if domain_filter:
    df_filtered = df_filtered[df_filtered['primary_domain'].isin(domain_filter)]
if level_filter:
    df_filtered = df_filtered[df_filtered['experience_level'].isin(level_filter)]
if location_filter:
    df_filtered = df_filtered[df_filtered['location'].isin(location_filter)]

# ══════════════════════════════════════════════════════════════════════════════
# MAIN HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<p class="main-header">Job Recommendation Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Gamage Recruiters · Data Science Internship Project</p>', unsafe_allow_html=True)

# Top-level KPIs
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Candidates", f"{len(df_c):,}")
col2.metric("Total Jobs", f"{len(df_j):,}")
col3.metric("Domains", df_c['primary_domain'].nunique())
col4.metric("Locations", df_c['location'].nunique())
col5.metric("Similarity Metric", METRIC_LABELS[selected_metric].split()[0])

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB: CANDIDATE EXPLORER
# ══════════════════════════════════════════════════════════════════════════════

if "Candidate Explorer" in active_tab:
    st.subheader("Candidate Explorer")

    # Candidate picker
    cand_options = df_filtered[['candidate_id', 'name', 'primary_domain', 'experience_level', 'location']].copy()
    cand_options['display'] = (
        cand_options['candidate_id'] + ' — ' +
        cand_options['name'] + ' | ' +
        cand_options['primary_domain'] + ' | ' +
        cand_options['experience_level'] + ' | ' +
        cand_options['location']
    )

    if cand_options.empty:
        st.warning("No candidates match the selected filters. Please adjust the sidebar filters.")
        st.stop()

    selected_display = st.selectbox(
        "Select a Candidate",
        options=cand_options['display'].tolist(),
        index=0,
    )
    selected_cid = selected_display.split(' — ')[0]
    cand_info = df_c[df_c['candidate_id'] == selected_cid].iloc[0]

    # ── Candidate profile card ───────────────────────────────────────────────
    st.markdown("#### Candidate Profile")
    c1, c2, c3 = st.columns([2, 2, 3])

    with c1:
        st.markdown(f"**Name:** {cand_info['name']}")
        st.markdown(f"**ID:** `{cand_info['candidate_id']}`")
        st.markdown(f"**Domain:** {cand_info.get('primary_domain', '—')}")
        st.markdown(f"**Level:** {cand_info.get('experience_level', '—')}")

    with c2:
        st.markdown(f"**Location:** {cand_info.get('location', '—')}")
        st.markdown(f"**Education:** {cand_info.get('education_level', '—')}")
        st.markdown(f"**University:** {cand_info.get('university', '—')}")
        yoe = cand_info.get('years_of_experience', cand_info.get('experience_years', '—'))
        st.markdown(f"**Years of Exp.:** {yoe}")

    with c3:
        st.markdown("**Skills:**")
        st.markdown(render_skill_tags(cand_info.get('skills', '')), unsafe_allow_html=True)
        pref_loc = cand_info.get('preferred_locations', cand_info.get('preferred_location', '—'))
        st.markdown(f"**Preferred Locations:** {pref_loc}")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Recommendations ──────────────────────────────────────────────────────
    st.markdown(f"#### Top {top_n} Job Recommendations — *{METRIC_LABELS[selected_metric]}*")

    recs_df = get_recommendations(selected_cid, metric=selected_metric, top_n=top_n)

    if recs_df.empty:
        st.info("No recommendations available. Run Steps 5 & 6 first.")
    else:
        # Score bar chart
        fig_bar = px.bar(
            recs_df,
            x='Score',
            y='Job Title',
            orientation='h',
            color='Score',
            color_continuous_scale=['#bbdefb', '#1565C0'],
            text='Score',
            hover_data=['Company', 'Industry', 'Location', 'Level', 'Salary (LKR)', 'Skill Overlap %'],
            labels={'Score': 'Similarity Score'},
        )
        fig_bar.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig_bar.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            coloraxis_showscale=False,
            height=max(300, top_n * 45),
            margin=dict(l=10, r=80, t=30, b=10),
            title=f"Similarity Scores — {METRIC_LABELS[selected_metric]}",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Detailed cards
        for _, row in recs_df.iterrows():
            with st.expander(
                f"#{int(row['Rank'])}  {row['Job Title']}  —  {row['Company']}  "
                f"({row['Location']})  |  Score: {row['Score']:.4f}  "
                f"| Skill Overlap: {row['Skill Overlap %']}%"
            ):
                rc1, rc2 = st.columns(2)
                with rc1:
                    st.markdown(f"**Industry:** {row['Industry']}")
                    st.markdown(f"**Level:** {row['Level']}")
                    st.markdown(f"**Salary:** {row['Salary (LKR)']}")
                with rc2:
                    st.markdown(f"**Similarity Score:** `{row['Score']:.4f}`")
                    st.markdown(f"**Skill Overlap:** `{row['Skill Overlap %']}%`")
                st.markdown("**Required Skills:**")
                st.markdown(render_skill_tags(row['Required Skills']), unsafe_allow_html=True)

        # Export button
        st.download_button(
            label="Download Recommendations (CSV)",
            data=recs_df.to_csv(index=False).encode(),
            file_name=f"recommendations_{selected_cid}.csv",
            mime="text/csv",
        )

    # ── Metric comparison for this candidate ────────────────────────────────
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### Metric Comparison for This Candidate")

    i = df_c[df_c['candidate_id'] == selected_cid].index[0]
    comp_rows = []
    for m_key, m_matrix in score_matrices.items():
        top_scores = np.sort(m_matrix[i])[::-1][:top_n]
        comp_rows.append({
            'Metric': METRIC_LABELS[m_key],
            'Top-1 Score': round(float(top_scores[0]), 4),
            f'Avg Top-{top_n} Score': round(float(top_scores.mean()), 4),
            'color': METRIC_COLORS[m_key],
        })
    comp_df = pd.DataFrame(comp_rows)

    fig_comp = px.bar(
        comp_df, x='Metric', y=f'Avg Top-{top_n} Score',
        color='Metric',
        color_discrete_map={METRIC_LABELS[k]: METRIC_COLORS[k] for k in METRIC_COLORS},
        text=f'Avg Top-{top_n} Score',
        title=f"Avg Top-{top_n} Score across Metrics (Candidate: {cand_info['name']})",
    )
    fig_comp.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig_comp.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig_comp, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB: MODEL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

elif "Model Evaluation" in active_tab:
    st.subheader("Model Evaluation — Step 7 Results")

    if eval_results is None:
        st.warning(
            "Evaluation results not found. Please run the **Step 7 notebook** first "
            "to generate `../outputs/step7_eval_results.json`."
        )
        st.stop()

    K_OPTIONS = [1, 3, 5, 10]
    k_sel = st.select_slider("Select K", options=K_OPTIONS, value=5)

    # ── Summary metrics table ────────────────────────────────────────────────
    st.markdown(f"#### Summary Table at K = {k_sel}")
    sum_rows = []
    for m_key, res in eval_results.items():
        sum_rows.append({
            'Metric':          res['label'],
            f'Precision@{k_sel}': round(res['precision'][str(k_sel)], 4),
            f'Recall@{k_sel}':    round(res['recall'][str(k_sel)],    4),
            f'F1@{k_sel}':        round(res['f1'][str(k_sel)],        4),
            f'NDCG@{k_sel}':      round(res['ndcg'][str(k_sel)],      4),
            f'HitRate@{k_sel}':   round(res['hit_rate'][str(k_sel)],  4),
            'MAP':                round(res['MAP'],                    4),
        })
    sum_df = pd.DataFrame(sum_rows).sort_values('MAP', ascending=False)

    def color_max(col):
        is_max = col == col.max()
        return ['background-color: #c8e6c9; font-weight: bold' if v else '' for v in is_max]

    num_cols = [c for c in sum_df.columns if c != 'Metric']
    st.dataframe(sum_df.style.apply(color_max, subset=num_cols), use_container_width=True)

    st.download_button(
        "Download Summary CSV",
        data=sum_df.to_csv(index=False).encode(),
        file_name="step7_eval_summary.csv",
        mime="text/csv",
    )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Precision & Recall curves ────────────────────────────────────────────
    st.markdown("#### Precision@K and Recall@K")
    col_p, col_r = st.columns(2)

    for col_widget, eval_key, title in [
        (col_p, 'precision', 'Precision@K'),
        (col_r, 'recall',    'Recall@K'),
    ]:
        fig = go.Figure()
        for m_key, res in eval_results.items():
            vals = [res[eval_key][str(k)] for k in K_OPTIONS]
            fig.add_trace(go.Scatter(
                x=K_OPTIONS, y=vals,
                mode='lines+markers',
                name=res['label'],
                line=dict(color=res['color'], width=3 if m_key == 'hybrid' else 1.5,
                          dash='solid' if m_key == 'hybrid' else 'dash'),
                marker=dict(size=8),
            ))
        fig.update_layout(
            title=title, xaxis_title='K', yaxis_title=eval_key.capitalize(),
            height=350, legend=dict(font=dict(size=10)),
        )
        col_widget.plotly_chart(fig, use_container_width=True)

    # ── MAP bar chart ────────────────────────────────────────────────────────
    st.markdown("#### Mean Average Precision (MAP)")
    map_df = pd.DataFrame([
        {'Metric': res['label'], 'MAP': res['MAP'], 'color': res['color']}
        for res in eval_results.values()
    ]).sort_values('MAP', ascending=False)

    fig_map = px.bar(
        map_df, x='Metric', y='MAP',
        color='Metric',
        color_discrete_map={res['label']: res['color'] for res in eval_results.values()},
        text='MAP',
        title='MAP — Mean Average Precision by Metric',
    )
    fig_map.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig_map.update_layout(showlegend=False, height=380)
    st.plotly_chart(fig_map, use_container_width=True)

    # ── NDCG@K ──────────────────────────────────────────────────────────────
    st.markdown("#### NDCG@K — Ranking Quality")
    fig_ndcg = go.Figure()
    for m_key, res in eval_results.items():
        vals = [res['ndcg'][str(k)] for k in K_OPTIONS]
        fig_ndcg.add_trace(go.Scatter(
            x=K_OPTIONS, y=vals,
            mode='lines+markers',
            name=res['label'],
            line=dict(color=res['color'], width=3 if m_key == 'hybrid' else 1.5,
                      dash='solid' if m_key == 'hybrid' else 'dash'),
            marker=dict(symbol='square', size=8),
        ))
    fig_ndcg.update_layout(
        xaxis_title='K', yaxis_title='NDCG',
        height=380, legend=dict(font=dict(size=10)),
    )
    st.plotly_chart(fig_ndcg, use_container_width=True)

    # ── Hit Rate ─────────────────────────────────────────────────────────────
    st.markdown("#### Hit Rate@K — % Candidates with ≥1 Relevant Job")
    fig_hit = go.Figure()
    for m_key, res in eval_results.items():
        vals = [res['hit_rate'][str(k)] * 100 for k in K_OPTIONS]
        fig_hit.add_trace(go.Scatter(
            x=K_OPTIONS, y=vals,
            mode='lines+markers',
            name=res['label'],
            line=dict(color=res['color'], width=3 if m_key == 'hybrid' else 1.5,
                      dash='solid' if m_key == 'hybrid' else 'dash'),
            marker=dict(symbol='triangle-up', size=9),
        ))
    fig_hit.update_layout(
        xaxis_title='K', yaxis_title='Hit Rate (%)',
        height=380, yaxis=dict(ticksuffix='%'), legend=dict(font=dict(size=10)),
    )
    st.plotly_chart(fig_hit, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB: BULK RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════

elif "Bulk Recommendations" in active_tab:
    st.subheader("Bulk Recommendations Table")

    st.markdown(
        f"Showing top **{top_n}** recommendations per candidate "
        f"using **{METRIC_LABELS[selected_metric]}**."
    )

    # Filter controls
    col_f1, col_f2, col_f3 = st.columns(3)
    search_name = col_f1.text_input("Search by name", placeholder="e.g. Kasun")
    domain_bulk = col_f2.selectbox("Domain", ['All'] + sorted(df_c['primary_domain'].dropna().unique()))
    level_bulk  = col_f3.selectbox("Level",  ['All'] + sorted(df_c['experience_level'].dropna().unique()))

    df_bulk = df_filtered.copy()
    if search_name:
        df_bulk = df_bulk[df_bulk['name'].str.contains(search_name, case=False, na=False)]
    if domain_bulk != 'All':
        df_bulk = df_bulk[df_bulk['primary_domain'] == domain_bulk]
    if level_bulk != 'All':
        df_bulk = df_bulk[df_bulk['experience_level'] == level_bulk]

    df_bulk = df_bulk.head(100)   # cap at 100 candidates for performance

    st.info(f"Displaying {len(df_bulk)} candidate(s). (Capped at 100 for performance.)")

    if df_bulk.empty:
        st.warning("No candidates match. Try adjusting the filters.")
    else:
        all_recs = []
        progress = st.progress(0)
        for idx_prog, (_, row) in enumerate(df_bulk.iterrows()):
            recs = get_recommendations(row['candidate_id'], metric=selected_metric, top_n=top_n)
            if not recs.empty:
                recs.insert(0, 'Candidate Name', row['name'])
                recs.insert(0, 'candidate_id',   row['candidate_id'])
                all_recs.append(recs)
            progress.progress((idx_prog + 1) / len(df_bulk))
        progress.empty()

        if all_recs:
            df_all = pd.concat(all_recs, ignore_index=True)

            # Wide pivot view (1 row per candidate, Rank 1-5 side by side)
            st.markdown("#### Wide View — One Row per Candidate")
            wide_rows = []
            for cid, grp in df_all.groupby('candidate_id'):
                cand_name = grp['Candidate Name'].iloc[0]
                w = {'ID': cid, 'Name': cand_name}
                for _, rec in grp.iterrows():
                    r = int(rec['Rank'])
                    w[f'#{r} Job'] = rec['Job Title']
                    w[f'#{r} Company'] = rec['Company']
                    w[f'#{r} Score'] = rec['Score']
                wide_rows.append(w)
            df_wide = pd.DataFrame(wide_rows)
            st.dataframe(df_wide, use_container_width=True, height=400)

            st.download_button(
                "Download Wide Table (CSV)",
                data=df_wide.to_csv(index=False).encode(),
                file_name="bulk_recommendations_wide.csv",
                mime="text/csv",
            )

            st.markdown("#### Long View — All Recommendations")
            display_cols = ['candidate_id', 'Candidate Name', 'Rank', 'Job Title', 'Company',
                            'Industry', 'Location', 'Level', 'Score', 'Skill Overlap %']
            st.dataframe(df_all[display_cols], use_container_width=True, height=500)

            st.download_button(
                "Download Full Long Table (CSV)",
                data=df_all.to_csv(index=False).encode(),
                file_name="bulk_recommendations_long.csv",
                mime="text/csv",
            )

    # ── Score heatmap for first 20 candidates ────────────────────────────────
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### Score Heatmap — Top Candidates × Top Jobs")

    if selected_metric in score_matrices and len(df_c) > 0:
        n_show = min(20, len(df_c))
        n_jobs_show = min(20, len(df_j))
        sample_idx = df_filtered.index[:n_show].tolist()
        heat_matrix = score_matrices[selected_metric][np.ix_(sample_idx, range(n_jobs_show))]
        cand_labels = df_c.loc[sample_idx, 'name'].tolist()
        job_labels  = df_j['job_title'].iloc[:n_jobs_show].tolist()

        fig_heat = px.imshow(
            heat_matrix,
            labels=dict(x="Job", y="Candidate", color="Score"),
            x=job_labels,
            y=cand_labels,
            color_continuous_scale='Blues',
            title=f"Score Heatmap — {METRIC_LABELS[selected_metric]} (first {n_show} candidates × {n_jobs_show} jobs)",
            aspect='auto',
        )
        fig_heat.update_layout(height=500)
        st.plotly_chart(fig_heat, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB: ABOUT
# ══════════════════════════════════════════════════════════════════════════════

elif "About" in active_tab:
    st.subheader("About This Dashboard")

    st.markdown("""
### Job Recommendation System
**Gamage Recruiters — Data Science Internship Project**

This dashboard is the deliverable of an internship project to build an 
end-to-end job recommendation system for the Sri Lankan recruitment market.

---
#### Pipeline Overview

| Step | Description | Status |
|------|-------------|--------|
| Step 2 | Synthetic dataset generation (500 candidates, 200 jobs) | Done |
| Step 3 | Text preprocessing & feature engineering | Done |
| Step 4 | Content-based recommendation engine | Done |
| Step 5 | TF-IDF & Word2Vec text feature extraction | Done |
| Step 6 | Multi-metric similarity matching & hybrid scoring | Done |
| **Step 7** | **Evaluation — Precision@K, Recall@K, MAP, NDCG** | Done |
| **Step 8** | **This dashboard** | Done |

---
#### Hybrid Scoring Formula

```
Hybrid Score = 0.55 × TF-IDF Cosine   (primary keyword signal)
             + 0.20 × Word2Vec Cosine  (semantic signal)
             + Experience Bonus        (0.15 / 0.07 / 0)
             + Domain Bonus            (0.10 / 0)
             + Location Bonus          (0.05 / 0)
```

---
#### Evaluation Proxy
No historical application data was available, so **domain match**  
(candidate `primary_domain` == job `industry`) is used as a binary relevance signal.

---
#### How to Run

```bash
pip install streamlit plotly pandas numpy
streamlit run step8_dashboard.py
```
""")

    # Dataset stats
    st.markdown("---")
    st.markdown("#### Dataset Statistics")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Candidate Domain Distribution**")
        dom_counts = df_c['primary_domain'].value_counts().reset_index()
        dom_counts.columns = ['Domain', 'Count']
        fig_dom = px.pie(dom_counts, names='Domain', values='Count',
                         hole=0.4, color_discrete_sequence=px.colors.qualitative.Set2)
        fig_dom.update_layout(height=350)
        st.plotly_chart(fig_dom, use_container_width=True)

    with c2:
        st.markdown("**Job Industry Distribution**")
        ind_counts = df_j['industry'].value_counts().reset_index()
        ind_counts.columns = ['Industry', 'Count']
        fig_ind = px.bar(ind_counts, x='Count', y='Industry', orientation='h',
                         color='Count', color_continuous_scale='Blues')
        fig_ind.update_layout(height=350, coloraxis_showscale=False,
                               yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_ind, use_container_width=True)


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:#9E9E9E; font-size:0.82rem;'>"
    "Job Recommendation System"
    "</center>",
    unsafe_allow_html=True,
)
