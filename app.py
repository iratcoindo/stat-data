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
    # LETTER GROUPING (CLD - VALID)
    # ===============================
    letters = {g: "" for g in group_order}

    if posthoc_df is not None and k > 2:

        alpha = 0.05

        # ===============================
        # BUILD P-VALUE MATRIX
        # ===============================
        p_matrix = pd.DataFrame(
            np.ones((k, k)),
            index=group_order,
            columns=group_order
        )

        for i in group_order:
            for j in group_order:
                if i == j:
                    continue

                try:
                    if test == "Kruskal":
                        p = posthoc_df.loc[i, j]

                    elif test == "ANOVA":
                        row = posthoc_df[
                            ((posthoc_df['group1']==i) & (posthoc_df['group2']==j)) |
                            ((posthoc_df['group1']==j) & (posthoc_df['group2']==i))
                        ]
                        p = row['p-adj'].values[0]

                    elif test == "Welch ANOVA":
                        row = posthoc_df[
                            ((posthoc_df['A']==i) & (posthoc_df['B']==j)) |
                            ((posthoc_df['A']==j) & (posthoc_df['B']==i))
                        ]
                        p = row['pval'].values[0]

                    else:
                        p = 1

                    p_matrix.loc[i, j] = p

                except:
                    p_matrix.loc[i, j] = 1

        # ===============================
        # SORT GROUP BY MEAN
        # ===============================
        means = grouped.mean().sort_values(ascending=False)
        sorted_groups = list(means.index)

        # ===============================
        # ASSIGN LETTERS
        # ===============================
        letter_list = []

        for g in sorted_groups:

            placed = False

            for group_set in letter_list:
                conflict = False

                for existing in group_set:
                    if p_matrix.loc[g, existing] < alpha:
                        conflict = True
                        break

                if not conflict:
                    group_set.append(g)
                    placed = True
                    break

            if not placed:
                letter_list.append([g])

        # assign letters
        alphabet = list("abcdefghijklmnopqrstuvwxyz")

        for idx, group_set in enumerate(letter_list):
            for g in group_set:
                letters[g] += alphabet[idx]
