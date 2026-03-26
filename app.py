import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp
import pingouin as pg

# ===============================
# STYLE
# ===============================
def apply_prism_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 1.5,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"]
    })

def prism_palette(n):
    base = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2","#937860","#DA8BC3"]
    return [base[i % len(base)] for i in range(n)]

# ===============================
# UI
# ===============================
st.set_page_config(layout="wide")
st.title("📊 iRATco Statistical Analyzer")

if "groups" not in st.session_state:
    st.session_state.groups = [{"name": "Group 1", "data": ""}]

if st.button("➕ Add Group"):
    st.session_state.groups.append({"name": f"Group {len(st.session_state.groups)+1}", "data": ""})

all_data = []

for i, g in enumerate(st.session_state.groups):

    st.markdown(f"### Group {i+1}")
    col1, col2 = st.columns([1,2])

    with col1:
        name = st.text_input("Name", g["name"], key=f"name{i}")
    with col2:
        data = st.text_area("Data", g["data"], key=f"data{i}")

    st.session_state.groups[i]["name"] = name
    st.session_state.groups[i]["data"] = data

    if data.strip():
        try:
            vals = [float(x) for x in data.split()]
            for v in vals:
                all_data.append({"Group": name, "Value": v})
        except:
            st.error(f"Error in group {i+1}")

# ===============================
# DATAFRAME
# ===============================
if all_data:

    df = pd.DataFrame(all_data)
    group_order = [g["name"] for g in st.session_state.groups if g["data"].strip()]

    df["Group"] = pd.Categorical(df["Group"], categories=group_order, ordered=True)

    st.dataframe(df, use_container_width=True)

    grouped = df.groupby("Group")["Value"]
    group_values = [grouped.get_group(g).values for g in group_order]

    # ===============================
    # NORMALITY
    # ===============================
    normal_flags = []
    for g in group_order:
        stat, p = stats.shapiro(grouped.get_group(g))
        normal_flags.append(p > 0.05)

    all_normal = all(normal_flags)

    # ===============================
    # VARIANCE
    # ===============================
    if len(group_values) == 2:
        stat, p = stats.levene(*group_values)
        var_equal = p > 0.05
    else:
        stat, p = stats.levene(*group_values)
        var_equal = p > 0.05

    # ===============================
    # TEST SELECTION
    # ===============================
    k = len(group_values)

    if k == 2:
        if all_normal and var_equal:
            test = "t-test"
            stat, p_value = stats.ttest_ind(*group_values)
        elif all_normal:
            test = "Welch t-test"
            stat, p_value = stats.ttest_ind(*group_values, equal_var=False)
        else:
            test = "Mann-Whitney"
            stat, p_value = stats.mannwhitneyu(*group_values)
        posthoc_df = None

    else:
        if all_normal and var_equal:
            test = "ANOVA"
            stat, p_value = stats.f_oneway(*group_values)
            posthoc = pairwise_tukeyhsd(df["Value"], df["Group"])
            posthoc_df = pd.DataFrame(data=posthoc.summary().data[1:], columns=posthoc.summary().data[0])

        elif all_normal:
            test = "Welch ANOVA"
            res = pg.welch_anova(dv="Value", between="Group", data=df)
            p_value = res["p-unc"][0]
            posthoc_df = pg.pairwise_gameshowell(dv="Value", between="Group", data=df)

        else:
            test = "Kruskal"
            stat, p_value = stats.kruskal(*group_values)
            posthoc_df = sp.posthoc_dunn(df, val_col="Value", group_col="Group", p_adjust="bonferroni")

    st.write(f"### 🧠 Test Used: {test}")
    st.write(f"p-value: {round(p_value,5)}")

    if posthoc_df is not None:
        st.write("### 📌 Post-hoc Result")
        st.dataframe(posthoc_df)

    # ===============================
    # LETTER GROUPING (BASED ON P)
    # ===============================
    letters = {g: "" for g in group_order}

    if posthoc_df is not None and k > 2:

        alpha = 0.05
        current_letter = "a"

        for g in group_order:
            letters[g] = current_letter

            for g2 in group_order:
                if g != g2:
                    try:
                        p = posthoc_df.loc[g, g2] if test=="Kruskal" else None
                        if p and p < alpha:
                            current_letter = chr(ord(current_letter)+1)
                    except:
                        pass

    # ===============================
    # PLOT
    # ===============================
    apply_prism_style()
    colors = prism_palette(len(group_order))

    fig, ax = plt.subplots()

    means = grouped.mean()
    stds = grouped.std()

    x = np.arange(len(group_order))

    ax.bar(x, means.values, yerr=stds.values, capsize=5, color=colors, edgecolor="black")

    for i, g in enumerate(group_order):
        y = grouped.get_group(g)
        jitter = np.random.normal(i, 0.04, size=len(y))
        ax.scatter(jitter, y, color="gray", s=30)

        if letters[g]:
            ax.text(i, max(y)+stds[g]*1.2, letters[g], ha='center', fontsize=14)

    ax.set_xticks(x)
    ax.set_xticklabels(group_order, rotation=45, ha='right')
    ax.set_ylabel("Value")

    st.pyplot(fig)
