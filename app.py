import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp
import pingouin as pg

# ===============================
# CLD FUNCTION (R multcompLetters equivalent)
# ===============================
def cld_from_pmatrix(p_matrix, alpha=0.05):
    groups = list(p_matrix.columns)

    letters = {g: "" for g in groups}
    letter_list = []

    for g in groups:
        placed = False

        for i, group_set in enumerate(letter_list):
            conflict = False

            for existing in group_set:
                if p_matrix.loc[g, existing] <= alpha:
                    conflict = True
                    break

            if not conflict:
                group_set.append(g)
                letters[g] += chr(97 + i)  # a, b, c...
                placed = True
                break

        if not placed:
            letter_list.append([g])
            letters[g] += chr(97 + len(letter_list) - 1)

    return letters

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
    # NORMALITY (SHAPIRO-WILK)
    # ===============================
    st.markdown("### 🧪 Normality Test (Shapiro-Wilk)")

    normality_results = []
    normal_flags = []

    for g in group_order:
        values = grouped.get_group(g)

        stat, p = stats.shapiro(values)

        normal = p > 0.05
        normal_flags.append(normal)

        normality_results.append({
            "Group": g,
            "W statistic": round(stat, 4),
            "p-value": round(p, 4),
            "Normal": "Yes" if normal else "No"
        })

    normality_df = pd.DataFrame(normality_results)
    st.dataframe(normality_df, use_container_width=True)

    all_normal = all(normal_flags)

    # ===============================
    # VARIANCE
    # ===============================
    st.markdown("### ⚖️ Variance Homogeneity Test")

    if len(group_values) == 2:
        # F-test
        var1 = np.var(group_values[0], ddof=1)
        var2 = np.var(group_values[1], ddof=1)

        F = var1 / var2 if var1 > var2 else var2 / var1

        # pendekatan kasar (rule of thumb)
        var_equal = F < 4

        st.write(f"F statistic: {round(F,4)}")
        st.write(f"Equal variance: {'Yes' if var_equal else 'No'}")

    else:
        # Bartlett test
        stat, p = stats.bartlett(*group_values)

        var_equal = p > 0.05

        st.write(f"Bartlett statistic: {round(stat,4)}")
        if p < 0.0001:
            st.write("p-value: < 0.0001")
        else:
            st.write(f"p-value: {round(p,4)}")
        st.write(f"Equal variance: {'Yes' if var_equal else 'No'}")

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

            # aman untuk berbagai versi / kondisi
            if "p-unc" in res.columns:
                p_value = res["p-unc"].values[0]
            elif "p-uncorrected" in res.columns:
                p_value = res["p-uncorrected"].values[0]
            else:
                st.warning("Welch ANOVA gagal, fallback ke Kruskal")
                stat, p_value = stats.kruskal(*group_values)
                test = "Kruskal"
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
# ===============================
# LETTER GROUPING (FINAL SAFE)
# ===============================
letters = {g: "" for g in group_order}

if posthoc_df is not None and k > 2:

    # ubah semua jadi string
    posthoc_df.index = posthoc_df.index.astype(str)
    posthoc_df.columns = posthoc_df.columns.astype(str)

    group_order_str = [str(g) for g in group_order]

    # ===============================
    # MATCH GROUP YANG ADA DI MATRIX
    # ===============================
    valid_groups = [g for g in group_order_str if g in posthoc_df.index]

    if len(valid_groups) < 2:
        st.error("Group names do not match posthoc matrix")
        st.write("Group order:", group_order_str)
        st.write("Posthoc index:", list(posthoc_df.index))
        st.stop()

    # ambil hanya yang valid
    p_matrix = posthoc_df.loc[valid_groups, valid_groups]

    # ===============================
    # CLD
    # ===============================
    letters = cld_from_pmatrix(p_matrix, alpha=0.05)

    # isi ke semua group (yang tidak ada tetap kosong)
    letters = {g: letters.get(str(g), "") for g in group_order}

    # DEBUG
    st.write("Valid groups:", valid_groups)
    st.write("P-matrix:")
    st.dataframe(p_matrix)
    st.write("Letters:", letters)
    # ===============================
    # PLOT
    # ===============================
    apply_prism_style()

    means = grouped.mean()
    stds = grouped.std()
    ns = grouped.count()
    ses = stds / np.sqrt(ns)

    # pilih error bar
    error_type = st.selectbox("Error Bar", ["SD", "SE"])
    errors = stds if error_type == "SD" else ses

    colors = prism_palette(len(group_order))
    x = np.arange(len(group_order))

    fig, ax = plt.subplots()

    # BARPLOT
    ax.bar(
        x,
        means.values,
        yerr=errors.values,
        capsize=6,
        color=colors,
        edgecolor="black",
        linewidth=1.5,
        alpha=0.8
    )

    # scatter (jitter)
    for i, g in enumerate(group_order):
        y = grouped.get_group(g)
        jitter = np.random.normal(i, 0.04, size=len(y))

        ax.scatter(
            jitter,
            y,
            color="gray",
            s=35,
            zorder=3
        )

    # ===============================
    # LETTER DISPLAY
    # ===============================
    y_max = df["Value"].max()
    y_range = y_max - df["Value"].min()

    for i, g in enumerate(group_order):
        if letters[g]:
            ax.text(
                i,
                y_max + y_range*0.1,
                letters[g],
                ha='center',
                fontsize=14
            )

    # ===============================
    # AXIS
    # ===============================
    ax.set_xticks(x)
    ax.set_xticklabels(group_order, rotation=45, ha='right')
    ax.set_ylabel("Value")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ===============================
    # TEST LABEL (AUTO)
    # ===============================
    if test == "ANOVA":
        test_label = "ANOVA"
    elif test == "Welch ANOVA":
        test_label = "Welch ANOVA"
    elif test == "Kruskal":
        test_label = "Kruskal-Wallis"
    elif test == "t-test":
        test_label = "t-test"
    elif test == "Welch t-test":
        test_label = "Welch t-test"
    elif test == "Mann-Whitney":
        test_label = "Mann-Whitney U"
    else:
        test_label = test

    # format p-value (biar rapi seperti jurnal)
    if p_value < 0.0001:
        p_text = "p < 0.0001"
    else:
        p_text = f"p = {round(p_value,4)}"

    # ===============================
    # POSISI TEKS
    # ===============================
    y_max = df["Value"].max()
    y_min = df["Value"].min()
    y_range = y_max - y_min

    ax.text(
        ax.get_xlim()[0],                  # kiri plot
        y_max + y_range*0.2,               # di atas plot
        f"{test_label}, {p_text}",
        ha='left',
        va='bottom',
        fontsize=13,
        fontweight='bold'
    )
    
    st.pyplot(fig)
