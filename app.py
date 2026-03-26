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
    # LETTER GROUPING (IMPROVED)
    # ===============================
    letters = {}
    alphabet = list("abcdefghijklmnopqrstuvwxyz")

    sorted_groups = means.sort_values(ascending=False).index

    threshold = errors.mean()

    group_letters = []

    for i, g in enumerate(sorted_groups):

        assigned = False

        for j, existing in enumerate(group_letters):
            # cek apakah bisa masuk grup ini
            conflict = False

            for eg in existing:
                if abs(means[g] - means[eg]) > threshold:
                    conflict = True
                    break

            if not conflict:
                existing.append(g)
                letters[g] = alphabet[j]
                assigned = True
                break

        if not assigned:
            group_letters.append([g])
            letters[g] = alphabet[len(group_letters)-1]

    # ===============================
    # VISUALIZATION (PRISM STYLE FINAL)
    # ===============================
    import matplotlib.pyplot as plt
    import numpy as np

    EDGE_COLOR = "black"   # abu gelap
    JITTER_COLOR = "gray"
    ALPHA_FILL = 0.7

    apply_prism_style()
    colors = prism_palette(len(groups))

    fig, ax = plt.subplots()

    # ===============================
    # GLOBAL Y SYSTEM (PENTING)
    # ===============================
    y_min = df["Value"].min()
    y_max = df["Value"].max()
    y_range = y_max - y_min

    offset_letter = y_range * 0.08
    offset_pval   = y_range * 0.1
    offset_top    = y_range * 0.11

    y_letter = y_max + offset_letter
    y_pval   = y_letter + offset_pval

    # ===============================
    # BOXPLOT
    # ===============================
    if plot_type == "Boxplot":

        data = [df[df["Group"] == g]["Value"] for g in groups]

        box = ax.boxplot(
            data,
            patch_artist=True,
            showfliers=False,
            widths=0.6
        )

        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(ALPHA_FILL)
            patch.set_edgecolor(EDGE_COLOR)
            patch.set_linewidth(1.5)

        for element in ['whiskers', 'caps', 'medians']:
            for item in box[element]:
                item.set_color(EDGE_COLOR)
                item.set_linewidth(1.5)

        # scatter
        for i, g in enumerate(groups):
            y = df[df["Group"] == g]["Value"]
            x = np.random.normal(i+1, 0.04, size=len(y))

            ax.scatter(
                x, y,
                color=JITTER_COLOR,
                s=35,              # lebih kecil
                linewidth=0,
                zorder=3
            )

        ax.set_xticklabels(groups,
            rotation=45,
            ha='right'   # penting supaya tidak numpuk
        )

    # ===============================
    # BARPLOT
    # ===============================
    else:

        x = np.arange(len(groups))

        ax.bar(
            x,
            means.values,
            yerr=errors.values,
            capsize=6,
            color=colors,
            edgecolor=EDGE_COLOR,
            linewidth=1.5,
            alpha=ALPHA_FILL
        )

        for i, g in enumerate(groups):
            y = df[df["Group"] == g]["Value"]
            x_jitter = np.random.normal(i, 0.04, size=len(y))

            ax.scatter(
                x_jitter,
                y,
                color=JITTER_COLOR,
                s=35,              # lebih kecil
                linewidth=0,
                zorder=3
            )

        ax.set_xticks(x)
        ax.set_xticklabels(groups,
            rotation=45,
            ha='right'   # penting supaya tidak numpuk
        )

    # ===============================
    # SIGNIFICANCE LETTER (RATA)
    # ===============================
    if p_value < 0.05:
        for i, g in enumerate(groups):

            xpos = i+1 if plot_type == "Boxplot" else i

            ax.text(
                xpos,
                y_letter,
                letters[g],
                ha='center',
                fontsize=14
            )

    # ===============================
    # P VALUE (KIRI ATAS)
    # ===============================
    ax.text(
        ax.get_xlim()[0],   # kiri plot
        y_pval,             # posisi Y yang sudah kita hitung
        f"p val = {round(p_value,4)}",
        ha='left',
        va='bottom',
        fontsize=14
    )

    # ===============================
    # AXIS LIMIT (BIAR GA NUBRUK)
    # ===============================
    ax.set_ylim(
        y_min-(y_range*0.2),
        y_letter + offset_pval + (offset_top*0.2)
    )

    # ===============================
    # CLEAN STYLE (PRISM)
    # ===============================
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for spine in ax.spines.values():
        spine.set_color(EDGE_COLOR)
        spine.set_linewidth(1.5)

    ax.set_ylabel("Value")

    st.pyplot(fig)
