import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def render_custom_css():
    st.markdown("""
    <style>
    /* Force White Background and Dark High-Contrast Text Everywhere */
    html, body, .stApp, div[data-testid="stAppViewContainer"], section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* Force All Headers, Labels, Paragraphs, Spans, Lists, Strong tags to Dark Text */
    h1, h2, h3, h4, h5, h6, p, span, label, div, li, ul, ol, strong, em {
        color: #0f172a !important;
    }

    /* Fix Blockquotes Visibility (`> text`) */
    blockquote {
        border-left: 4px solid #0f4c81 !important;
        background-color: #f8fafc !important;
        color: #0f172a !important;
        padding: 12px 18px !important;
        border-radius: 6px !important;
        margin: 14px 0 !important;
    }
    blockquote p, blockquote strong, blockquote span {
        color: #0f172a !important;
        font-size: 0.95rem !important;
    }

    /* Fix Inline Code Tags (`code`) Visibility */
    code {
        background-color: #f1f5f9 !important;
        color: #0f4c81 !important;
        border: 1px solid #cbd5e1 !important;
        padding: 2px 7px !important;
        border-radius: 4px !important;
        font-weight: 600 !important;
        font-family: monospace !important;
    }

    /* Fix Streamlit Tabs Text Visibility */
    button[data-baseweb="tab"] {
        color: #475569 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #0f4c81 !important;
        border-bottom-color: #0f4c81 !important;
        font-weight: 700 !important;
    }
    button[data-baseweb="tab"] p, button[data-baseweb="tab"] span {
        color: inherit !important;
    }

    /* Fix Captions, Subtexts, and Muted Labels */
    .stCaption, div[data-testid="stCaptionContainer"] p, div[data-testid="stCaptionContainer"] span {
        color: #475569 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }

    /* Fix Alert Banners (st.info, st.success, st.warning, st.error) */
    div[data-testid="stAlert"] {
        background-color: #f8fafc !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        padding: 12px 16px !important;
    }
    div[data-testid="stAlert"] p, div[data-testid="stAlert"] span, div[data-testid="stAlert"] div {
        color: #0f172a !important;
        font-weight: 600 !important;
    }

    /* Text inputs & Textarea */
    textarea, input, select {
        color: #0f172a !important;
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
    }
    textarea::placeholder, input::placeholder {
        color: #64748b !important;
    }

    /* Top Medical Header Bar */
    .header-card {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-top: 4px solid #0f4c81 !important;
        border-radius: 8px !important;
        padding: 18px 24px !important;
        margin-bottom: 20px !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04) !important;
    }
    .header-title {
        color: #0f4c81 !important;
        font-size: 1.85rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
        letter-spacing: -0.5px !important;
    }
    .header-subtitle {
        color: #334155 !important;
        font-size: 0.92rem !important;
        margin-top: 4px !important;
        font-weight: 500 !important;
    }

    /* Professional Clinical KPI Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 16px;
        margin-bottom: 22px;
    }
    .kpi-card {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-top: 4px solid #2563eb !important;
        border-radius: 8px !important;
        padding: 16px 18px !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04) !important;
    }
    .kpi-card.warning { border-top-color: #d97706 !important; }
    .kpi-card.danger { border-top-color: #dc2626 !important; }
    .kpi-card.success { border-top-color: #059669 !important; }
    
    .kpi-label {
        font-size: 0.78rem !important;
        text-transform: uppercase !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px !important;
        color: #475569 !important;
    }
    .kpi-value {
        font-size: 1.75rem !important;
        font-weight: 800 !important;
        color: #0f172a !important;
        margin-top: 4px !important;
    }
    .kpi-subtext {
        font-size: 0.80rem !important;
        color: #334155 !important;
        font-weight: 500 !important;
        margin-top: 2px !important;
    }

    /* Clinical Section Container Cards */
    .section-card {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        padding: 20px !important;
        margin-bottom: 20px !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04) !important;
    }
    .section-title {
        font-size: 1.10rem !important;
        font-weight: 700 !important;
        color: #0f4c81 !important;
        margin-bottom: 14px !important;
        border-bottom: 2px solid #e2e8f0 !important;
        padding-bottom: 8px !important;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Streamlit UI Button Adjustments */
    .stButton>button {
        background-color: #0f4c81 !important;
        color: #ffffff !important;
        border-radius: 6px !important;
        border: none !important;
        font-weight: 600 !important;
        padding: 8px 18px !important;
    }
    .stButton>button * {
        color: #ffffff !important;
    }
    .stButton>button:hover {
        background-color: #1e40af !important;
    }
    
    /* Selectbox, Input Box Styling */
    div[data-baseweb="select"], div[data-baseweb="input"] {
        background-color: #ffffff !important;
        border-radius: 6px !important;
        border: 1px solid #cbd5e1 !important;
    }
    div[data-baseweb="select"] *, div[data-baseweb="input"] * {
        color: #0f172a !important;
    }

    /* Dataframe and Tables */
    div[data-testid="stTable"], div[data-testid="stDataFrame"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
    }
    div[data-testid="stTable"] *, div[data-testid="stDataFrame"] * {
        color: #0f172a !important;
    }
    </style>
    """, unsafe_allow_html=True)

def plot_tissue_composition_donut(tissue_breakdown):
    """
    Generate clean, light-themed Plotly Donut Chart for tissue composition.
    """
    labels = ["Granulation Tissue", "Fibrinous Slough", "Necrotic Eschar", "Epithelialization"]
    values = [
        tissue_breakdown.get("Granulation_Pct", 40.0),
        tissue_breakdown.get("Slough_Pct", 30.0),
        tissue_breakdown.get("Necrotic_Pct", 15.0),
        tissue_breakdown.get("Epithelial_Pct", 15.0)
    ]
    colors = ["#dc2626", "#d97706", "#334155", "#ec4899"]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color="#ffffff", width=2)),
        textinfo="label+percent",
        hoverinfo="label+value+percent",
        textfont=dict(color="#0f172a", size=12, family="sans-serif")
    )])

    fig.update_layout(
        title=dict(text="Wound Bed Tissue Distribution (%)", font=dict(color="#0f172a", size=14, family="sans-serif")),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#0f172a", family="sans-serif"),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, font=dict(color="#0f172a")),
        margin=dict(l=20, r=20, t=40, b=50),
        height=320
    )
    return fig

def plot_longitudinal_trajectory(trajectory_dict):
    """
    Generate clean, light-themed Plotly line chart with confidence bounds for longitudinal healing forecasts.
    """
    weeks = trajectory_dict.get("weeks", [0, 2, 4, 6, 8, 10, 12])
    healing = trajectory_dict.get("healing_pct", [0, 20, 40, 60, 75, 88, 95])
    upper = trajectory_dict.get("healing_upper", [0, 28, 48, 68, 83, 95, 100])
    lower = trajectory_dict.get("healing_lower", [0, 12, 32, 52, 67, 81, 90])
    inf_risk = trajectory_dict.get("infection_risk_pct", [45, 38, 30, 22, 15, 10, 6])

    fig = go.Figure()

    # Confidence Interval Fill Area
    fig.add_trace(go.Scatter(
        x=weeks + weeks[::-1],
        y=upper + lower[::-1],
        fill="toself",
        fillcolor="rgba(37, 99, 235, 0.12)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        name="95% Confidence Interval"
    ))

    # Projected Healing Trajectory Line
    fig.add_trace(go.Scatter(
        x=weeks,
        y=healing,
        mode="lines+markers",
        name="Projected Healing %",
        line=dict(color="#1d4ed8", width=3),
        marker=dict(size=8, color="#1d4ed8")
    ))

    # Infection Risk Curve Line
    fig.add_trace(go.Scatter(
        x=weeks,
        y=inf_risk,
        mode="lines+markers",
        name="Infection Risk %",
        line=dict(color="#dc2626", width=2.5, dash="dash"),
        marker=dict(size=6, color="#dc2626")
    ))

    fig.update_layout(
        title=dict(text="Longitudinal Healing Trajectory & Infection Forecast", font=dict(color="#0f172a", size=15, family="sans-serif")),
        xaxis=dict(title=dict(text="Timeline (Weeks)", font=dict(color="#0f172a")), gridcolor="#e2e8f0", dtick=2, tickfont=dict(color="#0f172a")),
        yaxis=dict(title=dict(text="Percentage (%)", font=dict(color="#0f172a")), gridcolor="#e2e8f0", range=[0, 105], tickfont=dict(color="#0f172a")),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#0f172a", family="sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#0f172a")),
        margin=dict(l=40, r=30, t=50, b=40),
        height=360
    )
    return fig

def plot_digital_twin_radar(baseline_state, simulated_state=None):
    """
    Generate clean light-themed Polar Radar chart comparing baseline vs simulated Patient Digital Twin state.
    """
    categories = [
        "Cellular Proliferation",
        "Microvascular Perfusion",
        "Collagen Synthesis",
        "Moisture Balance",
        "Biofilm Control",
        "Low MMP Enzyme Activity"
    ]

    def state_to_vals(st):
        return [
            st.get("Cellular_Proliferation_Factor", 50.0),
            st.get("Microvascular_Perfusion", 50.0),
            st.get("Collagen_Synthesis_Rate", 50.0),
            st.get("Moisture_Balance_Index", 50.0),
            100.0 - st.get("Bacterial_Biofilm_Density", 50.0),
            100.0 - st.get("MMP_Activity", 50.0)
        ]

    v_base = state_to_vals(baseline_state)

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=v_base,
        theta=categories,
        fill="toself",
        name="Baseline Biological State",
        fillcolor="rgba(217, 119, 6, 0.15)",
        line=dict(color="#d97706", width=2)
    ))

    if simulated_state is not None:
        v_sim = state_to_vals(simulated_state)
        fig.add_trace(go.Scatterpolar(
            r=v_sim,
            theta=categories,
            fill="toself",
            name="Simulated Post-Intervention",
            fillcolor="rgba(5, 150, 105, 0.20)",
            line=dict(color="#059669", width=2.5)
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], color="#0f172a", gridcolor="#e2e8f0", tickfont=dict(color="#0f172a")),
            angularaxis=dict(color="#0f172a", gridcolor="#e2e8f0", tickfont=dict(color="#0f172a")),
            bgcolor="#ffffff"
        ),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#0f172a", family="sans-serif"),
        title=dict(text="Patient Digital Twin Biological State Radar", font=dict(color="#0f172a", size=14, family="sans-serif")),
        legend=dict(font=dict(color="#0f172a")),
        margin=dict(l=40, r=40, t=40, b=40),
        height=380
    )
    return fig

def plot_shap_waterfall(impact_df):
    """
    Generate horizontal bar chart for clinical SHAP feature importances on white background.
    """
    df_sorted = impact_df.sort_values(by="SHAP_Value", ascending=True)

    colors = ["#ef4444" if v > 0 else "#10b981" for v in df_sorted["SHAP_Value"]]

    fig = go.Figure(go.Bar(
        x=df_sorted["SHAP_Value"],
        y=df_sorted["Feature"],
        orientation="h",
        marker=dict(color=colors),
        text=[f"{v:+.1f} days" for v in df_sorted["SHAP_Value"]],
        textposition="outside",
        textfont=dict(color="#0f172a", size=12)
    ))

    fig.update_layout(
        title=dict(text="SHAP Clinical Driver Attribution (+ Days Delay)", font=dict(color="#0f172a", size=14, family="sans-serif")),
        xaxis=dict(title=dict(text="Impact on Estimated Healing Time (Days)", font=dict(color="#0f172a")), gridcolor="#e2e8f0", tickfont=dict(color="#0f172a")),
        yaxis=dict(color="#0f172a", tickfont=dict(color="#0f172a")),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#0f172a", family="sans-serif"),
        margin=dict(l=160, r=40, t=40, b=40),
        height=360
    )
    return fig
