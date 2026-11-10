import html as html_lib
import io
import math
import os
from fractions import Fraction
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Analyse – Tableaux statistiques", layout="wide")

# Interface style « tableau noir » (boutons et onglets)
st.markdown("""
<style>
.stButton>button, .stDownloadButton>button {
  background:#f4f1e8; color:#1f3a30; font-weight:700;
  border:2px dashed #cfd8cf; border-radius:10px;
}
.stButton>button:hover, .stDownloadButton>button:hover {
  background:#ffd86b; color:#1f3a30; border-color:#fff;
}
button[data-baseweb="tab"] { color:#e8efe9; font-weight:600; }
button[data-baseweb="tab"][aria-selected="true"] { color:#ffd86b; }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Lecture du fichier
# ----------------------------------------------------------------------------
def clean_value(x):
    """Texte propre : 1.0 -> '1', espaces retirés, vide -> NaN."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (float, np.floating)) and float(x).is_integer():
        return str(int(x))
    s = str(x).strip()
    return s if s != "" else np.nan


def _src_bytes(src):
    """Contenu brut + nom du fichier (chemin local ou fichier téléversé)."""
    if hasattr(src, "getvalue"):
        return src.getvalue(), src.name
    return Path(src).read_bytes(), str(src)


def _read_csv_fast(data):
    """CSV : séparateur détecté sur la 1re ligne, puis lecture avec le moteur C (rapide)."""
    text = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    first = text.split("\n", 1)[0]
    sep = max([",", ";", "\t", "|"], key=first.count)
    return pd.read_csv(io.StringIO(text), header=None, sep=sep)


def load_table(src, sheet):
    """Lit la feuille et détecte seul le tableau : ignore les lignes et colonnes
    vides autour, la première ligne restante contient les noms de colonnes."""
    name = getattr(src, "name", str(src)).lower()
    if name.endswith((".csv", ".txt")):
        data = src.getvalue() if hasattr(src, "getvalue") else Path(src).read_bytes()
        raw = _read_csv_fast(data)
    else:
        raw = pd.read_excel(src, sheet_name=sheet, header=None)
    raw = raw.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    if raw.empty:
        return raw
    header = [str(c).strip() if not pd.isna(c) else f"Colonne {i + 1}" for i, c in enumerate(raw.iloc[0])]
    df = raw.iloc[1:].copy()
    df.columns = header
    return df.reset_index(drop=True)


@st.cache_data(show_spinner="Lecture du fichier…")
def load_table_cached(data, name, sheet):
    """Le fichier n'est relu que s'il a changé (le cache dépend de son contenu)."""
    buf = io.BytesIO(data)
    buf.name = name
    return load_table(buf, sheet)


@st.cache_data(show_spinner=False)
def sheet_names_cached(data, name):
    buf = io.BytesIO(data)
    buf.name = name
    return sheet_names(buf)


@st.cache_data(show_spinner=False)
def prepare_data(df, variables, id_col):
    """Nettoyage des valeurs (mis en cache : fait une seule fois par fichier)."""
    data = df.copy()
    for v in variables:
        data[v] = data[v].map(clean_value)
    if id_col != "(aucune)":
        data.index = data[id_col].astype(str)
        data.index.name = id_col
    return data


MAX_MODS = 30        # au-delà, une colonne n'est pas une variable qualitative (identifiant, valeur continue…)
MAX_SHOW_ROWS = 2000  # lignes affichées à l'écran (le calcul utilise toutes les lignes)


def show_df(df, cmap=None):
    """Affiche un grand tableau sans ralentir le navigateur : seulement les premières lignes."""
    n = len(df)
    if n > MAX_SHOW_ROWS:
        st.caption(f"Affichage des {MAX_SHOW_ROWS} premières lignes sur {n} (tous les calculs utilisent les {n} lignes).")
        df = df.head(MAX_SHOW_ROWS)
    if cmap and df.size <= 100_000:
        try:
            st.dataframe(df.style.background_gradient(cmap=cmap, axis=None), width="stretch")
            return
        except Exception:
            pass
    st.dataframe(df, width="stretch")


def sheet_names(src):
    name = getattr(src, "name", str(src)).lower()
    if name.endswith((".csv", ".txt")):
        return [None]
    return pd.ExcelFile(src).sheet_names


# ----------------------------------------------------------------------------
# Gestion et Tri des Modalités
# ----------------------------------------------------------------------------
def get_sorted_mods(mods, t):
    """ترتيب الموداليات: إذا كانت ثنائية (Oui/Non) تصبح Oui ثم Non، وإذا كانت غير ذلك تبقى حسب اختيار المستخدم."""
    n = len(mods)
    if t == "Nominale" and n == 2:
        oui_mods = [m for m in mods if "oui" in str(m).strip().lower()]
        non_mods = [m for m in mods if "non" in str(m).strip().lower()]
        other_mods = [m for m in mods if m not in oui_mods and m not in non_mods]
        if oui_mods and non_mods:
            return oui_mods + non_mods + other_mods
    return mods


def build_disjunctive(df, config):
    """Tableau disjonctif complet : uniquement des 0/1."""
    blocks = []
    for var, mods in config.items():
        b = pd.DataFrame({m: (df[var] == m).astype(int) for m in mods})
        b.columns = pd.MultiIndex.from_product([[var], mods], names=["Variable", "Modalité"])
        blocks.append(b)
    return pd.concat(blocks, axis=1)


def build_coded(df, config, types):
    """Données codées (Tableau de codage) مع الترتيب التراكمي للـ Ordinale (ex: 1110, 1111). Version vectorisée."""
    blocks = []
    for var, mods in config.items():
        n = len(mods)
        t = types[var]
        if t == "Ordinale":
            labels = [f"≥ {m}" for m in mods]
            pos = df[var].map({m: i + 1 for i, m in enumerate(mods)}).fillna(0).astype(int)
            cols = [(pos >= i).astype(int).to_numpy() for i in range(1, n + 1)]
        else:
            labels = [f"{m} (Bit)" for m in mods] if n == 2 else list(mods)
            sv = df[var].astype(str).str.strip().str.lower()
            cols = [(sv == str(m).strip().lower()).astype(int).to_numpy() for m in mods]
        arr = np.column_stack(cols) if cols else np.zeros((len(df), 0), dtype=int)
        b = pd.DataFrame(arr, index=df.index,
                         columns=pd.MultiIndex.from_product([[var], labels], names=["Variable", "Bit"]))
        blocks.append(b)
    return pd.concat(blocks, axis=1)


def build_contingency_matrix(df, var1, var2, order1, order2):
    """Tableau de contingence (croisement de deux variables)."""
    ct = pd.crosstab(df[var1], df[var2], margins=True, margins_name="Total")

    existing_index = [m for m in order1 if m in ct.index]
    if "Total" in ct.index:
        existing_index.append("Total")
    existing_cols = [m for m in order2 if m in ct.columns]
    if "Total" in ct.columns:
        existing_cols.append("Total")

    try:
        ct = ct.reindex(index=existing_index, columns=existing_cols, fill_value=0)
    except Exception:
        pass

    return ct


# ----------------------------------------------------------------------------
# Affichage : couleurs et tableaux groupés
# ----------------------------------------------------------------------------
PALETTE = ["#fde2e4", "#dcefe6", "#dbe7fd", "#fff1c1", "#e6dcff", "#d3f0fa", "#ffe0c2", "#e3f5d1"]


def shade(frac, rgb):
    """Fond coloré : blanc -> couleur `rgb` selon frac (0 à 1), le texte reste lisible."""
    frac = max(0.0, min(1.0, float(frac)))
    t = 0.08 + 0.62 * frac
    r, g, b = [int(255 - (255 - c) * t) for c in rgb]
    return f"background-color: rgb({r},{g},{b})"


def cellmap(df, fn):
    return df.map(fn) if hasattr(df, "map") else df.applymap(fn)


def unique_labels(labels):
    seen, out = {}, []
    for lab in labels:
        if lab in seen:
            seen[lab] += 1
            out.append(f"{lab}_{seen[lab]}")
        else:
            seen[lab] = 1
            out.append(lab)
    return out


def group_spans(labels):
    """[(étiquette, début, longueur)] pour des étiquettes consécutives identiques."""
    labels, spans, i = list(labels), [], 0
    while i < len(labels):
        j = i
        while j < len(labels) and labels[j] == labels[i]:
            j += 1
        spans.append((labels[i], i, j - i))
        i = j
    return spans


def grouped_table_html(df, cell_css, colors, max_height=560):
    """Tableau HTML dont les lignes ET les colonnes sont groupées par variable :
    le nom de la variable est écrit une seule fois, au-dessus (colonnes) et à gauche (lignes)
    de ses modalités."""
    esc = html_lib.escape
    r_var, r_mod = list(df.index.get_level_values(0)), list(df.index.get_level_values(1))
    c_var, c_mod = list(df.columns.get_level_values(0)), list(df.columns.get_level_values(1))
    r_spans, c_spans = group_spans(r_var), group_spans(c_var)
    r_starts = {s for _, s, _ in r_spans}
    c_starts = {s for _, s, _ in c_spans}
    thick = "3px solid #374151"
    thin = "1px solid #d1d5db"
    base = "padding:6px 10px;text-align:center;box-sizing:border-box;"

    def bg(var):
        return f"background:{colors.get(var, '#f3f4f6')};"

    h = [
        f'<div style="overflow:auto;max-height:{max_height}px;border:1px solid #e5e7eb;border-radius:8px;">',
        '<table style="border-collapse:separate;border-spacing:0;font-family:sans-serif;font-size:14px;'
        'width:max-content;min-width:100%;color:#111827;">',
    ]
    # --- ligne d'en-tête 1 : noms des variables (fusionnés) ---
    h.append("<tr>")
    h.append(
        f'<th rowspan="2" colspan="2" style="{base}background:#e5e7eb;position:sticky;top:0;left:0;'
        f'z-index:4;border-bottom:{thick};border-right:{thick};"></th>'
    )
    for var, start, n in c_spans:
        h.append(
            f'<th colspan="{n}" style="{base}{bg(var)}font-weight:700;height:34px;position:sticky;top:0;'
            f'z-index:3;border-left:{thick};border-bottom:{thin};">{esc(str(var))}</th>'
        )
    h.append("</tr>")
    # --- ligne d'en-tête 2 : modalités ---
    h.append("<tr>")
    for j, (var, mod) in enumerate(zip(c_var, c_mod)):
        left = f"border-left:{thick};" if j in c_starts else ""
        h.append(
            f'<th style="{base}{bg(var)}font-weight:600;min-width:64px;position:sticky;top:34px;z-index:3;'
            f'{left}border-bottom:{thick};">{esc(str(mod))}</th>'
        )
    h.append("</tr>")
    # --- corps ---
    values = df.values
    for i in range(len(df)):
        top = f"border-top:{thick};" if i in r_starts else ""
        h.append("<tr>")
        if i in r_starts:
            n = next(n for _, s, n in r_spans if s == i)
            h.append(
                f'<th rowspan="{n}" style="{base}{bg(r_var[i])}font-weight:700;min-width:90px;'
                f'position:sticky;left:0;z-index:2;border-top:{thick};border-right:{thin};">{esc(str(r_var[i]))}</th>'
            )
        h.append(
            f'<th style="{base}{bg(r_var[i])}font-weight:600;min-width:110px;position:sticky;left:90px;z-index:2;'
            f'{top}border-right:{thick};">{esc(str(r_mod[i]))}</th>'
        )
        for j in range(len(c_var)):
            left = f"border-left:{thick};" if j in c_starts else ""
            h.append(
                f'<td style="{base}{cell_css(i, j, values[i][j])}{top}{left}border-bottom:{thin};">'
                f"{esc(str(values[i][j]))}</td>"
            )
        h.append("</tr>")
    h.append("</table></div>")
    return "".join(h)


def _hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def burt_css(burt, colors, rgb=(124, 58, 237)):
    """Couleur des cellules de Burt :
    - blocs « même variable » (diagonale de blocs) : couleur de la variable (celle de son en-tête),
      la diagonale elle-même un peu plus foncée et en gras ;
    - autres blocs : intensité selon l'effectif."""
    vmax = max(float(np.max(burt.values)), 1.0)
    r_var = list(burt.index.get_level_values(0))
    c_var = list(burt.columns.get_level_values(0))

    def css(i, j, v):
        if r_var[i] == c_var[j]:
            base = colors.get(r_var[i], "#f3f4f6")
            if i == j:
                r, g, b = [int(c * 0.82) for c in _hex_rgb(base)]
                return f"background:rgb({r},{g},{b});font-weight:700;"
            return f"background:{base};color:#6b7280;"
        return shade(v / vmax, rgb) + ";"

    return css


def show_html(html_str, fallback_df):
    try:
        st.html(html_str)
    except Exception:
        try:
            st.markdown(html_str, unsafe_allow_html=True)
        except Exception:
            st.dataframe(fallback_df, width="stretch")


# ----------------------------------------------------------------------------
# Tables de semblance (ressemblance) et de dissemblance entre individus
# ----------------------------------------------------------------------------
SIM_INDICES = {
    "Concordance simple  (a+d)/p": "smc",
    "Jaccard  a/(a+b+c)": "jac",
    "Dice  2a/(2a+b+c)": "dice",
    "Nombre de concordances  (a+d)": "match",
}
DIS_INDICES = {
    "Distance de Hamming  (b+c) = nombre de bits différents": "ham",
    "1 − Concordance simple": "smc",
    "1 − Jaccard": "jac",
    "1 − Dice": "dice",
}


def pair_counts(X):
    """Pour chaque paire d'individus : a = 1 communs, b/c = 1 chez un seul, d = 0 communs, p = nb de colonnes."""
    X = np.asarray(X, dtype=float)
    p = X.shape[1]
    ones = X.sum(axis=1)
    a = X @ X.T
    b = ones[:, None] - a
    c = ones[None, :] - a
    d = p - a - b - c
    return a, b, c, d, p


FRAC_MODES = ["Fractions", "Fractions simplifiées", "Décimaux"]


def sim_parts(X, kind):
    """Numérateur et dénominateur (entiers) de l'indice de ressemblance pour chaque paire d'individus."""
    a, b, c, d, p = pair_counts(X)
    if kind == "smc":
        num, den = a + d, np.full(a.shape, float(p))
    elif kind == "jac":
        num, den = a, a + b + c
    elif kind == "dice":
        num, den = 2 * a, 2 * a + b + c
    else:  # nombre de concordances (entier)
        num, den = a + d, np.ones(a.shape)
    zero = den == 0  # deux vecteurs entièrement nuls : identiques -> ressemblance 1
    num = np.where(zero, 1.0, num)
    den = np.where(zero, 1.0, den)
    return num.astype(int), den.astype(int)


def dis_parts(X, kind):
    """Numérateur et dénominateur de l'indice de dissemblance (Hamming = entier, sinon 1 - ressemblance)."""
    if kind == "ham":
        a, b, c, d, p = pair_counts(X)
        return (b + c).astype(int), np.ones(a.shape, dtype=int)
    num, den = sim_parts(X, kind)
    return den - num, den


def frac_text(n, d, mode):
    n, d = int(n), int(d)
    if mode == "Décimaux":
        return str(n) if d == 1 else f"{n / d:.3f}"
    if mode == "Fractions simplifiées":
        g = math.gcd(n, d) or 1
        n, d = n // g, d // g
    if n == 0:
        return "0"
    if d == 1:
        return str(n)
    if n == d:
        return "1"
    return f"{n}/{d}"


def frac_labels(num, den, mode):
    return [[frac_text(num[i, j], den[i, j], mode) for j in range(num.shape[1])] for i in range(num.shape[0])]


# ----------------------------------------------------------------------------
# Choix de l'ordre d'une variable ordinale
# ----------------------------------------------------------------------------
def _order_add(key, m):
    st.session_state[key].append(m)


def _order_move(key, i, step):
    lst = st.session_state[key]
    j = i + step
    if 0 <= j < len(lst):
        lst[i], lst[j] = lst[j], lst[i]


def _order_first(key, i):
    lst = st.session_state[key]
    lst.insert(0, lst.pop(i))


def _order_remove(key, i):
    st.session_state[key].pop(i)


def _order_clear(key):
    st.session_state[key] = []


def _order_reverse(key):
    st.session_state[key] = st.session_state[key][::-1]


def order_editor(box, key, mods):
    """Ordre d'une variable ordinale, modifiable à tout moment :
    un clic sur une modalité l'ajoute à la fin ; ⬆ ⬇ la déplacent (on peut donc en mettre une en premier) ;
    ✖ la retire ; « Tout effacer » recommence depuis le début ; « Inverser » renverse l'ordre."""
    if key not in st.session_state:
        st.session_state[key] = []
    st.session_state[key] = [m for m in st.session_state[key] if m in mods]
    cur = st.session_state[key]
    rest = [m for m in mods if m not in cur]

    if rest:
        box.caption("Cliquez les modalités dans l'ordre croissant :")
        for s in range(0, len(rest), 6):
            chunk = rest[s : s + 6]
            for col, m in zip(box.columns(len(chunk)), chunk):
                col.button(f"＋ {m}", key=f"{key}_add_{m}", on_click=_order_add, args=(key, m))
    for i, m in enumerate(cur):
        c0, c1, c2, c3, c4 = box.columns([5, 1, 1, 1, 1])
        c0.markdown(f"**{i + 1}.** {m}")
        c1.button("⏫", key=f"{key}_top_{i}", on_click=_order_first, args=(key, i), disabled=(i == 0), help="Mettre en premier")
        c2.button("⬆", key=f"{key}_up_{i}", on_click=_order_move, args=(key, i, -1), disabled=(i == 0), help="Monter d'un cran")
        c3.button("⬇", key=f"{key}_dn_{i}", on_click=_order_move, args=(key, i, 1), disabled=(i == len(cur) - 1), help="Descendre d'un cran")
        c4.button("✖", key=f"{key}_rm_{i}", on_click=_order_remove, args=(key, i), help="Retirer")
    if cur:
        e1, e2, _ = box.columns([2, 2, 4])
        e1.button("↺ Tout effacer", key=f"{key}_clr", on_click=_order_clear, args=(key,))
        e2.button("⇅ Inverser", key=f"{key}_rev", on_click=_order_reverse, args=(key,))
    return list(cur)


# ----------------------------------------------------------------------------
# Ordre d'une variable ordinale par glisser-déposer
# ----------------------------------------------------------------------------
_SORT_DIR = Path(__file__).resolve().parent / "_sortable_modalites"

_SORT_HTML = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
  :root { --fg:#111827; --chip:#eef2ff; --border:#c7d2fe; --accent:#4f46e5; }
  html, body { margin:0; padding:0; background:transparent; }
  body { padding:4px 2px 8px 2px; font-family:"Source Sans Pro", "Segoe UI", sans-serif; color:var(--fg); }
  #list { display:flex; flex-wrap:wrap; gap:10px; align-items:center; min-height:44px; }
  .chip { display:flex; align-items:center; gap:8px; padding:8px 12px; border-radius:10px;
          background:var(--chip); border:1px solid var(--border); cursor:grab; user-select:none;
          font-size:15px; box-shadow:0 1px 2px rgba(0,0,0,.08); }
  .chip:hover { border-color:var(--accent); }
  .chip.dragging { opacity:.35; }
  .rank { background:var(--accent); color:#fff; border-radius:999px; min-width:22px; height:22px;
          display:inline-flex; align-items:center; justify-content:center; font-size:12px; font-weight:700; }
  .grip { opacity:.45; letter-spacing:-2px; }
</style>
</head>
<body>
<div id="list"></div>
<script>
  var list = document.getElementById("list");
  var order = [];
  var dragged = null;

  function send(type, data) {
    var msg = { isStreamlitMessage: true, type: type };
    for (var k in data) { msg[k] = data[k]; }
    window.parent.postMessage(msg, "*");
  }
  function setHeight() {
    send("streamlit:setFrameHeight", { height: document.body.scrollHeight + 6 });
  }
  function currentOrder() {
    return Array.prototype.map.call(list.children, function (c) { return c.dataset.name; });
  }
  function refreshRanks() {
    Array.prototype.forEach.call(list.children, function (c, i) {
      c.querySelector(".rank").textContent = i + 1;
    });
  }
  function commit() {
    var now = currentOrder();
    refreshRanks();
    if (JSON.stringify(now) !== JSON.stringify(order)) {
      order = now;
      send("streamlit:setComponentValue", { value: now, dataType: "json" });
    }
  }
  function build(items) {
    list.innerHTML = "";
    items.forEach(function (name) {
      var el = document.createElement("div");
      el.className = "chip";
      el.draggable = true;
      el.dataset.name = name;
      el.innerHTML = '<span class="rank"></span><span class="label"></span><span class="grip">&#8942;&#8942;</span>';
      el.querySelector(".label").textContent = name;
      el.addEventListener("dragstart", function (e) {
        dragged = el;
        el.classList.add("dragging");
        e.dataTransfer.effectAllowed = "move";
        e.dataTransfer.setData("text/plain", name);
      });
      el.addEventListener("dragend", function () {
        el.classList.remove("dragging");
        dragged = null;
        commit();
      });
      el.addEventListener("dragover", function (e) {
        e.preventDefault();
        if (!dragged || dragged === el) { return; }
        var r = el.getBoundingClientRect();
        var before;
        if (e.clientY < r.top) { before = true; }
        else if (e.clientY > r.bottom) { before = false; }
        else { before = e.clientX < r.left + r.width / 2; }
        list.insertBefore(dragged, before ? el : el.nextSibling);
        refreshRanks();
      });
      list.appendChild(el);
    });
    refreshRanks();
  }
  list.addEventListener("dragover", function (e) { e.preventDefault(); });
  list.addEventListener("drop", function (e) { e.preventDefault(); });

  function applyTheme(t) {
    if (!t) { return; }
    var s = document.documentElement.style;
    if (t.textColor) { s.setProperty("--fg", t.textColor); }
    if (t.secondaryBackgroundColor) { s.setProperty("--chip", t.secondaryBackgroundColor); }
    if (t.primaryColor) { s.setProperty("--accent", t.primaryColor); }
    s.setProperty("--border", "rgba(128,128,128,.35)");
  }

  window.addEventListener("message", function (e) {
    if (!e.data || e.data.type !== "streamlit:render") { return; }
    if (dragged) { return; }
    applyTheme(e.data.theme);
    var items = (e.data.args && e.data.args.items) || [];
    if (JSON.stringify(items) !== JSON.stringify(order)) {
      order = items.slice();
      build(order);
    }
    setHeight();
  });

  send("streamlit:componentReady", { apiVersion: 1 });
  setHeight();
</script>
</body>
</html>
"""


@st.cache_resource
def _declare_sortable():
    """Déclare le composant glisser-déposer (le fichier HTML est créé à côté de app.py). False si impossible."""
    try:
        import streamlit.components.v1 as components

        _SORT_DIR.mkdir(exist_ok=True)
        f = _SORT_DIR / "index.html"
        if not f.exists() or f.read_text(encoding="utf-8") != _SORT_HTML:
            f.write_text(_SORT_HTML, encoding="utf-8")
        return components.declare_component("modalites_sortable", path=str(_SORT_DIR))
    except Exception:
        return False


def order_sortable(box, key, mods):
    """Ordre d'une variable ordinale : on glisse les modalités les unes parmi les autres.
    Toutes les modalités sont toujours classées ; l'ordre affiché (1, 2, 3…) est celui utilisé."""
    comp = _declare_sortable()
    if not comp:
        return order_editor(box, key, mods)  # solution de secours : boutons
    state_key = f"dnd_{key}"
    cur = [m for m in st.session_state.get(state_key, []) if m in mods]
    cur += [m for m in mods if m not in cur]
    with box:
        res = comp(items=cur, key=key, default=cur)
    if isinstance(res, list) and sorted(res) == sorted(mods):
        cur = list(res)
    st.session_state[state_key] = cur
    return cur


# ----------------------------------------------------------------------------
# Semblance / dissemblance entre modalités (mêmes lignes et colonnes que la table de Burt)
# ----------------------------------------------------------------------------
def matrix_css(values, rgb, vmax):
    """Couleur selon la valeur numérique ; le texte affiché peut être une fraction."""
    vmax = max(float(vmax), 1e-9)

    def css(i, j, v):
        s = shade(values[i][j] / vmax, rgb) + ";"
        return s + "font-weight:700;" if i == j else s

    return css


def proximity_tab(data, config, variables, colors, export_sheets, prefix, indices, parts_fn, rgb, int_kind, sheet):
    """Tableau modalités x modalités, groupé par variable comme la table de Burt.
    Chaque modalité est un vecteur 0/1 sur les n individus (colonne du tableau disjonctif)."""
    sel = st.multiselect(
        "Variables à utiliser",
        variables,
        default=variables,
        key=f"{prefix}_vars_{abs(hash(tuple(variables)))}",
    )
    c_b, c_c = st.columns([3, 2])
    idx_name = c_b.selectbox("Indice", list(indices), key=f"{prefix}_idx")
    mode = c_c.radio("Affichage", FRAC_MODES, key=f"{prefix}_mode", horizontal=True)
    if not sel:
        st.info("Choisissez au moins une variable.")
        return
    Z = build_disjunctive(data, {v: config[v] for v in sel})
    kind = indices[idx_name]
    num, den = parts_fn(Z.values.T, kind)
    labels = pd.DataFrame(frac_labels(num, den, mode), index=Z.columns, columns=Z.columns)
    labels.index.names = [None, None]
    labels.columns.names = [None, None]
    vmax = Z.shape[0] if kind == int_kind else 1
    show_html(grouped_table_html(labels, matrix_css(num / den, rgb, vmax), colors), labels)
    export_sheets[sheet] = (labels, True)


def flat(cols):
    return [f"{a} | {b}" for a, b in cols]


def to_excel(sheets):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, (frame, keep_index) in sheets.items():
            f = frame.copy()
            if isinstance(f.columns, pd.MultiIndex):
                f.columns = flat(f.columns)
            if isinstance(f.index, pd.MultiIndex):
                f.index = flat(f.index)
            f.to_excel(xw, sheet_name=name[:31], index=keep_index)
    return buf.getvalue()


# ----------------------------------------------------------------------------
# Nuages N(I) et N(J) : inertie, variance, covariance, corrélation, convergence
# ----------------------------------------------------------------------------
def burt_block(data, config, v1, v2):
    """Bloc (v1, v2) de la table de Burt = Z1ᵀ Z2 = tableau de contingence (sans totaux)."""
    Z1 = build_disjunctive(data, {v1: config[v1]})
    Z2 = build_disjunctive(data, {v2: config[v2]})
    B = Z1.values.T @ Z2.values
    return pd.DataFrame(B, index=config[v1], columns=config[v2])


def cloud_moments(prof, w, center, labels):
    """Variance, covariance et corrélation des coordonnées d'un nuage (métrique D^-1 du cours).
    prof   : profils (un point par ligne), w : poids des points, center : profil moyen (centre de gravité g).
    Var(k)    = Σ_p w_p (prof_pk − center_k)² / center_k
    Cov(k,k') = Σ_p w_p (prof_pk − center_k)(prof_pk' − center_k') / sqrt(center_k · center_k')
    Cor(k,k') = Cov(k,k') / (σ_k · σ_k')"""
    D = prof - center[None, :]
    cov = (D * w[:, None]).T @ D / np.sqrt(np.outer(center, center))
    var = np.diag(cov).copy()
    sd = np.sqrt(var)
    den = np.outer(sd, sd)
    cor = np.divide(cov, den, out=np.zeros_like(cov), where=den > 0)
    return (
        pd.Series(var, index=labels, name="Variance"),
        pd.DataFrame(cov, index=labels, columns=labels),
        pd.DataFrame(cor, index=labels, columns=labels),
    )


def nuages_inertie(N):
    """Nuages N(I) (profils-lignes) et N(J) (profils-colonnes), inerties, variances (valeurs propres),
    covariances et corrélations entre modalités."""
    N = N.loc[N.sum(axis=1) > 0, N.sum(axis=0) > 0]
    n = float(N.values.sum())
    F = N.values / n
    r, c = F.sum(axis=1), F.sum(axis=0)          # poids f(i.) et f(.j)

    prof_I = F / r[:, None]                       # points de N(I) : f(ij)/f(i.)
    prof_J = (F / c[None, :]).T                   # points de N(J) : f(ij)/f(.j)
    d2_I = ((prof_I - c[None, :]) ** 2 / c[None, :]).sum(axis=1)   # d² du χ² au centre de gravité g
    d2_J = ((prof_J - r[None, :]) ** 2 / r[None, :]).sum(axis=1)
    in_I, in_J = r * d2_I, c * d2_J               # inertie d'un point = poids × d²

    S = (F - np.outer(r, c)) / np.sqrt(np.outer(r, c))
    lam = (np.linalg.svd(S, compute_uv=False) ** 2)[: min(N.shape) - 1]   # variances des axes
    E = np.outer(N.sum(axis=1), N.sum(axis=0)) / n
    chi2 = float(((N.values - E) ** 2 / E).sum())

    tI = pd.DataFrame({"Poids f(i.)": r, "d² au centre G": d2_I, "Inertie": in_I,
                       "Contribution %": 100 * in_I / in_I.sum()}, index=N.index)
    tJ = pd.DataFrame({"Poids f(.j)": c, "d² au centre G": d2_J, "Inertie": in_J,
                       "Contribution %": 100 * in_J / in_J.sum()}, index=N.columns)
    prI = pd.DataFrame(prof_I, index=N.index, columns=N.columns)
    prJ = pd.DataFrame(prof_J, index=N.columns, columns=N.index)

    # N(I) : coordonnées = colonnes j  -> Var(j), Cov(j,j'), Cor(j,j')
    var_I, cov_I, cor_I = cloud_moments(prof_I, r, c, list(N.columns))
    # N(J) : coordonnées = lignes i    -> Var(i), Cov(i,i'), Cor(i,i')
    var_J, cov_J, cor_J = cloud_moments(prof_J, c, r, list(N.index))

    return dict(N=N, n=n, tI=tI, tJ=tJ, prI=prI, prJ=prJ, lam=lam, chi2=chi2,
                inertia_I=float(in_I.sum()), inertia_J=float(in_J.sum()),
                var_I=var_I, cov_I=cov_I, cor_I=cor_I,
                var_J=var_J, cov_J=cov_J, cor_J=cor_J)


def _cor_style(df):
    try:
        return df.round(4).style.background_gradient(cmap="RdBu_r", axis=None, vmin=-1, vmax=1)
    except Exception:
        return df.round(4)


import re


def _tex(s):
    """Texte sûr pour \\text{...} dans LaTeX."""
    return "\\text{" + re.sub(r"([_&%#$])", r"\\\1", str(s)) + "}"


def _fr(a, b):
    return "\\frac{" + str(int(a)) + "}{" + str(int(b)) + "}"


def cloud_detail(M, point_labels, coord_labels):
    """Nuage dont les points sont les LIGNES de M (effectifs) et les coordonnées les COLONNES.
    N(I) : M = N ; N(J) : M = Nᵀ."""
    M = np.asarray(M, dtype=int)
    n = int(M.sum())
    a, b = M.sum(axis=1), M.sum(axis=0)
    w, c = a / n, b / n                      # poids des points, centre de gravité g
    prof = M / a[:, None]                    # profils
    D = prof - c[None, :]
    d2 = (D ** 2 / c[None, :]).sum(axis=1)   # d² du χ² entre le point et g
    return dict(M=M, n=n, a=a, b=b, w=w, c=c, prof=prof, D=D, d2=d2, inertia=w * d2,
                pl=list(point_labels), cl=list(coord_labels))


def _dev(det, p, k):
    return "\\left(" + _fr(det["M"][p, k], det["a"][p]) + "-" + _fr(det["b"][k], det["n"]) + "\\right)"


def latex_point_inertia(det, p):
    n, a, b = det["n"], det["a"], det["b"]
    terms = ["\\frac{" + _dev(det, p, k) + "^2}{" + _fr(b[k], n) + "}" for k in range(len(b))]
    return ("I(" + _tex(det["pl"][p]) + ")=" + _fr(a[p], n) + "\\left[" + "+".join(terms) + "\\right]="
            + _fr(a[p], n) + "\\times " + f"{det['d2'][p]:.4f}" + "=" + f"{det['inertia'][p]:.5f}")


def latex_var(det, k):
    n, a, b, w, c, D = det["n"], det["a"], det["b"], det["w"], det["c"], det["D"]
    terms = [_fr(a[p], n) + "\\cdot\\frac{" + _dev(det, p, k) + "^2}{" + _fr(b[k], n) + "}" for p in range(len(a))]
    val = float(np.sum(w * D[:, k] ** 2 / c[k]))
    return "\\mathrm{Var}(" + _tex(det["cl"][k]) + ")=" + "+".join(terms) + "=" + f"{val:.5f}"


def latex_cov_cor(det, k1, k2):
    n, a, b, w, c, D = det["n"], det["a"], det["b"], det["w"], det["c"], det["D"]
    terms = [
        _fr(a[p], n) + "\\cdot\\frac{" + _dev(det, p, k1) + _dev(det, p, k2) + "}{\\sqrt{"
        + _fr(b[k1], n) + "\\cdot" + _fr(b[k2], n) + "}}"
        for p in range(len(a))
    ]
    cov = float(np.sum(w * D[:, k1] * D[:, k2]) / np.sqrt(c[k1] * c[k2]))
    v1 = float(np.sum(w * D[:, k1] ** 2 / c[k1]))
    v2 = float(np.sum(w * D[:, k2] ** 2 / c[k2]))
    cor = cov / math.sqrt(v1 * v2) if v1 * v2 > 0 else 0.0
    A, B = _tex(det["cl"][k1]), _tex(det["cl"][k2])
    l1 = "\\mathrm{Cov}(" + A + "," + B + ")=" + "+".join(terms) + "=" + f"{cov:.5f}"
    l2 = ("\\mathrm{Cor}(" + A + "," + B + ")=\\frac{\\mathrm{Cov}(" + A + "," + B + ")}{\\sigma_{" + A + "}\\cdot\\sigma_{" + B
          + "}}=\\frac{" + f"{cov:.5f}" + "}{\\sqrt{" + f"{v1:.5f}" + "}\\cdot\\sqrt{" + f"{v2:.5f}" + "}}=" + f"{cor:.4f}")
    return l1, l2


def cloud_table(det, wlabel):
    """Tableau du nuage : profils (coordonnées), poids, d² au centre g, inertie ; dernière ligne = g."""
    df = pd.DataFrame(det["prof"], index=det["pl"], columns=det["cl"])
    df[wlabel] = det["w"]
    df["d²(point, g)"] = det["d2"]
    df["Inertie = poids × d²"] = det["inertia"]
    g = pd.DataFrame([list(det["c"]) + [float(det["w"].sum()), np.nan, float(det["inertia"].sum())]],
                     index=["g (centre de gravité)"], columns=df.columns)
    return pd.concat([df, g])


def distance_matrix(det):
    """d²(p,q) = Σ_k (prof[p,k] − prof[q,k])² / c[k]  (métrique du χ², M = D^-1 du centre g)."""
    P, c = det["prof"], det["c"]
    n = len(det["pl"])
    d2 = np.zeros((n, n))
    for p in range(n):
        for q in range(n):
            d2[p, q] = float(np.sum((P[p] - P[q]) ** 2 / c))
    return pd.DataFrame(d2, index=det["pl"], columns=det["pl"])


def latex_distance(det, p, q):
    n, a, b = det["n"], det["a"], det["b"]
    M, c = det["M"], det["c"]
    A, B = _tex(det["pl"][p]), _tex(det["pl"][q])
    terms = []
    for k in range(len(b)):
        diff = "\\left(" + _fr(M[p, k], a[p]) + "-" + _fr(M[q, k], a[q]) + "\\right)"
        terms.append("\\frac{" + diff + "^2}{" + _fr(b[k], n) + "}")
    val = float(np.sum((det["prof"][p] - det["prof"][q]) ** 2 / c))
    return "d^2(" + A + "," + B + ")=" + "+".join(terms) + "=" + f"{val:.5f}"


# ---- Distances exactes (fractions) ----
def exact_distance_table(det):
    """d²(p,q) en fractions exactes."""
    M, a, b, n = det["M"], det["a"], det["b"], det["n"]
    m = len(det["pl"])
    D = [[Fraction(0)] * m for _ in range(m)]
    for p in range(m):
        for q in range(m):
            s = Fraction(0)
            for k in range(len(b)):
                diff = Fraction(int(M[p, k]), int(a[p])) - Fraction(int(M[q, k]), int(a[q]))
                s += diff * diff / Fraction(int(b[k]), int(n))
            D[p][q] = s
    return D


LATEX_I = (r"I_{N(I)}=\sum_{i=1}^{n} f_{i.}\,\Big\|f_j^{i}-f_{.j}\Big\|^2_{M=D^{-1}_{f.j}}"
           r"=\sum_{i=1}^{n} f_{i.}\sum_{j=1}^{p}\frac{\left(f_j^{i}-f_{.j}\right)^2}{f_{.j}}")
LATEX_J = (r"I_{N(J)}=\sum_{j=1}^{p} f_{.j}\,\Big\|f_i^{j}-f_{i.}\Big\|^2_{M=D^{-1}_{f.i}}"
           r"=\sum_{j=1}^{p} f_{.j}\sum_{i=1}^{n}\frac{\left(f_i^{j}-f_{i.}\right)^2}{f_{i.}}")


def render_cloud(title, det, general_latex, wlabel, dist_latex):
    st.markdown("### " + title)
    st.latex(general_latex)
    tbl = cloud_table(det, wlabel)
    st.dataframe(tbl.round(4), width="stretch")
    with st.expander("Calcul détaillé de l'inertie, point par point", expanded=True):
        for p in range(len(det["pl"])):
            st.latex(latex_point_inertia(det, p))
        st.latex("I=" + "+".join(f"{t:.5f}" for t in det["inertia"]) + f"={det['inertia'].sum():.5f}")

    dist = distance_matrix(det)
    w = det["w"]
    check = 0.5 * float(np.sum(np.outer(w, w) * dist.values))
    st.caption(f"Vérification : ½ Σ f_p f_q d²(p,q) = {check:.5f} = inertie du nuage ({det['inertia'].sum():.5f}).")
    return tbl, dist


# ---- Interprétation de la corrélation + graphiques ----
def interp_cor(r):
    a = abs(r)
    return "Faible" if a <= 0.3 else ("Moyenne" if a <= 0.7 else "Forte")


def interpretation_table(cor):
    names, rows = list(cor.index), []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = float(cor.iloc[i, j])
            sens = "positive" if r > 0.0005 else ("négative" if r < -0.0005 else "nulle")
            rows.append({"Modalité A": names[i], "Modalité B": names[j],
                         "Corrélation": round(r, 4), "Intensité": interp_cor(r), "Sens": sens})
    return pd.DataFrame(rows)


def _int_color(v):
    c = {"Faible": "#fecaca", "Moyenne": "#fde68a", "Forte": "#bbf7d0"}.get(v, "")
    return f"background-color:{c};color:#111827;font-weight:700" if c else ""


def chart_moments(var, cov, cor, kind):
    if kind == "Variance":
        df = var.rename_axis("Modalité").reset_index()
        df.columns = ["Modalité", "Variance"]
        base = alt.Chart(df).encode(
            x=alt.X("Modalité:N", sort=list(df["Modalité"]), axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Variance:Q"))
        return alt.layer(
            base.mark_bar(opacity=0.35, color="#f59e0b"),
            base.mark_line(point=alt.OverlayMarkDef(size=90), color="#f59e0b"),
            base.mark_text(dy=-10, fontWeight="bold").encode(text=alt.Text("Variance:Q", format=".4f")),
        ).properties(height=340)
    m = cov if kind == "Covariance" else cor
    df = m.rename_axis("A").reset_index().melt(id_vars="A", var_name="B", value_name="valeur")
    vmax = max(float(np.abs(df["valeur"]).max()), 1e-9) if kind == "Covariance" else 1.0
    base = alt.Chart(df).encode(x=alt.X("B:N", sort=list(m.columns), title=None),
                                y=alt.Y("A:N", sort=list(m.index), title=None))
    rect = base.mark_rect().encode(
        color=alt.Color("valeur:Q", scale=alt.Scale(scheme="redblue", domain=[-vmax, vmax], reverse=True)),
        tooltip=["A", "B", alt.Tooltip("valeur:Q", format=".4f")])
    txt = base.mark_text(fontWeight="bold").encode(
        text=alt.Text("valeur:Q", format=".2f"),
        color=alt.condition(f"abs(datum.valeur) > {0.6 * vmax}", alt.value("white"), alt.value("black")))
    return (rect + txt).properties(height=340)


def render_moments(title, det, var, cov, cor, key, general_latex):
    st.markdown("#### " + title)
    st.latex(general_latex)
    cA, cB = st.columns([1, 2])
    cA.markdown("Variance")
    cA.dataframe(var.round(5).to_frame(), width="stretch")
    cB.markdown("Covariance")
    cB.dataframe(cov.round(5), width="stretch")
    st.markdown("Corrélation")
    st.dataframe(_cor_style(cor), width="stretch")
    with st.expander("Calcul détaillé de la variance", expanded=True):
        for k in range(len(det["cl"])):
            st.latex(latex_var(det, k))
        vals = [float(np.sum(det["w"] * det["D"][:, k] ** 2 / det["c"][k])) for k in range(len(det["cl"]))]
        st.latex("\\sum \\mathrm{Var}=" + "+".join(f"{v:.5f}" for v in vals) + f"={sum(vals):.5f}")
    if len(det["cl"]) >= 2:
        with st.expander("Calcul détaillé de Cov et Cor entre deux modalités", expanded=True):
            c1, c2 = st.columns(2)
            m1 = c1.selectbox("Modalité A", det["cl"], index=0, key=key + "_A")
            m2 = c2.selectbox("Modalité B", det["cl"], index=1, key=key + "_B")
            if m1 == m2:
                st.warning("Choisissez deux modalités différentes.")
            else:
                l1, l2 = latex_cov_cor(det, det["cl"].index(m1), det["cl"].index(m2))
                st.latex(l1)
                st.latex(l2)

    st.markdown("**Interprétation de la corrélation**")
    st.caption("|r| de 0 à 0,3 : faible · de 0,3 à 0,7 : moyenne · au-delà de 0,7 jusqu'à 1 : forte")
    it = interpretation_table(cor)
    if not it.empty:
        sty = it.style
        sty = sty.map(_int_color, subset=["Intensité"]) if hasattr(sty, "map") else sty.applymap(_int_color, subset=["Intensité"])
        st.dataframe(sty, width="stretch")


def tab_nuages(data, config, variables, export_sheets):
    st.caption("Analyse des nuages N(I) et N(J) entre deux variables, à partir du bloc de la table de Burt.")
    if len(variables) < 2:
        st.info("Sélectionnez au moins deux variables.")
        return
    a, b = st.columns(2)
    v1 = a.selectbox("Variable 1 (lignes → nuage N(I))", variables, index=0, key="nu_v1")
    v2 = b.selectbox("Variable 2 (colonnes → nuage N(J))", variables, index=1, key="nu_v2")
    if v1 == v2:
        st.warning("Veuillez choisir deux variables différentes.")
        return
    N = burt_block(data, config, v1, v2)
    res = nuages_inertie(N)
    if min(res["N"].shape) < 2:
        st.warning("Il faut au moins 2 modalités non vides dans chaque variable.")
        return
    Nc = res["N"]
    rows, cols = list(Nc.index), list(Nc.columns)

    st.markdown("**Bloc de Burt (effectifs)**")
    st.dataframe(Nc, width="stretch")

    det_I = cloud_detail(Nc.values, rows, cols)     # points = modalités de v1, coordonnées = modalités de v2
    det_J = cloud_detail(Nc.values.T, cols, rows)   # points = modalités de v2, coordonnées = modalités de v1

    DIST_I = (r"d^2\left(f_j^{i},f_j^{i'}\right)=\left\|f_j^{i}-f_j^{i'}\right\|^2_{M=D^{-1}_{f.j}}"
              r"=\sum_{j=1}^{p}\frac{\left(f_j^{i}-f_j^{i'}\right)^2}{f_{.j}}")
    DIST_J = (r"d^2\left(f_i^{j},f_i^{j'}\right)=\left\|f_i^{j}-f_i^{j'}\right\|^2_{M=D^{-1}_{f_i.}}"
              r"=\sum_{i=1}^{n}\frac{\left(f_i^{j}-f_i^{j'}\right)^2}{f_{i.}}")
    tbl_I, dist_I = render_cloud(f"Table du nuage N(I) : points = modalités de {v1}", det_I, LATEX_I, "Poids f(i.)", DIST_I)
    tbl_J, dist_J = render_cloud(f"Table du nuage N(J) : points = modalités de {v2}", det_J, LATEX_J, "Poids f(.j)", DIST_J)

    st.markdown("### Variance, covariance et corrélation")
    st.latex(r"\mathrm{Var}(x)=\sum_i \tfrac1n (x_i-\bar x)^2\qquad \mathrm{Cov}(x,y)=\sum_i \tfrac1n (x_i-\bar x)(y_i-\bar y)"
             r"\qquad \mathrm{Cor}(x,y)=\frac{\mathrm{Cov}(x,y)}{\sigma_x\,\sigma_y}")
    lat_var_J = (r"\mathrm{Var}(i)=\sum_{j} f_{.j}\,\frac{\left(f_i^{j}-f_{i.}\right)^2}{f_{i.}}\qquad "
                 r"\mathrm{Cov}(i,i')=\sum_{j} f_{.j}\,\frac{\left(f_i^{j}-f_{i.}\right)\left(f_{i'}^{j}-f_{i'.}\right)}{\sqrt{f_{i.}\,f_{i'.}}}")
    lat_var_I = (r"\mathrm{Var}(j)=\sum_{i} f_{i.}\,\frac{\left(f_j^{i}-f_{.j}\right)^2}{f_{.j}}\qquad "
                 r"\mathrm{Cov}(j,j')=\sum_{i} f_{i.}\,\frac{\left(f_j^{i}-f_{.j}\right)\left(f_{j'}^{i}-f_{.j'}\right)}{\sqrt{f_{.j}\,f_{.j'}}}")
    render_moments(f"Dans N(J) : coordonnées = modalités de {v1}", det_J, res["var_J"], res["cov_J"], res["cor_J"],
                   f"mJ_{v1}_{v2}", lat_var_J)
    render_moments(f"Dans N(I) : coordonnées = modalités de {v2}", det_I, res["var_I"], res["cov_I"], res["cor_I"],
                   f"mI_{v1}_{v2}", lat_var_I)

    st.markdown("### Inertie et convergence")
    st.latex("I_{N(I)}=I_{N(J)}=\\sum_i \\mathrm{Var}(i)=\\sum_j \\mathrm{Var}(j)=\\sum_k \\lambda_k=\\frac{\\chi^2}{n}")
    lines = [
        "I_{N(I)}&=" + f"{res['inertia_I']:.5f}",
        "I_{N(J)}&=" + f"{res['inertia_J']:.5f}",
        "\\sum_i \\mathrm{Var}(i)&=" + "+".join(f"{v:.5f}" for v in res["var_J"]) + f"={res['var_J'].sum():.5f}",
        "\\sum_j \\mathrm{Var}(j)&=" + "+".join(f"{v:.5f}" for v in res["var_I"]) + f"={res['var_I'].sum():.5f}",
        "\\sum_k \\lambda_k&=" + "+".join(f"{v:.5f}" for v in res["lam"]) + f"={res['lam'].sum():.5f}",
        "\\chi^2&=n\\cdot I=" + str(int(res["n"])) + "\\times" + f"{res['inertia_I']:.5f}" + f"={res['chi2']:.4f}"
        + "\\;\\Rightarrow\\;\\frac{\\chi^2}{n}=" + f"{res['chi2'] / res['n']:.5f}",
    ]
    st.latex("\\begin{aligned}" + "\\\\".join(lines) + "\\end{aligned}")
    ok = (
        math.isclose(res["inertia_I"], res["inertia_J"], abs_tol=1e-9)
        and math.isclose(res["inertia_I"], res["lam"].sum(), abs_tol=1e-9)
        and math.isclose(res["inertia_I"], res["var_J"].sum(), abs_tol=1e-9)
        and math.isclose(res["inertia_I"], res["var_I"].sum(), abs_tol=1e-9)
    )
    if ok:
        st.success("Convergence vérifiée : I N(I) = I N(J) = Σ Var = Σ λₖ = χ²/n.")
    else:
        st.error("Les inerties ne sont pas égales (vérifiez les données).")

    lam = res["lam"]
    var_tab = pd.DataFrame({
        "Valeur propre λ (variance de l'axe)": lam,
        "% d'inertie": 100 * lam / lam.sum(),
        "% cumulé": 100 * np.cumsum(lam) / lam.sum(),
    }, index=[f"Axe {k + 1}" for k in range(len(lam))])
    st.dataframe(var_tab.round(4), width="stretch")
    st.bar_chart(var_tab["Valeur propre λ (variance de l'axe)"])

    export_sheets["Nuage_NI"] = (tbl_I, True)
    export_sheets["Nuage_NJ"] = (tbl_J, True)
    export_sheets["Dist_NI"] = (dist_I, True)
    export_sheets["Dist_NJ"] = (dist_J, True)
    export_sheets["Cov_NJ"] = (res["cov_J"], True)
    export_sheets["Cor_NJ"] = (res["cor_J"], True)
    export_sheets["Cov_NI"] = (res["cov_I"], True)
    export_sheets["Cor_NI"] = (res["cor_I"], True)
    export_sheets["Inertie_variance"] = (var_tab, True)


# ----------------------------------------------------------------------------
# Histogramme empilé à partir de la table de Burt (comme sur le croquis)
# ----------------------------------------------------------------------------
HIST_COLORS = ["#ef4444", "#f59e0b", "#22c55e", "#3b82f6", "#a855f7", "#14b8a6", "#ec4899", "#84cc16"]


def stacked_bars(df, xcol, ycol, mods_c, x_order):
    """Une barre par modalité de la variable 1 ; les modalités de la variable 2 sont empilées
    et leur nom est écrit dans chaque bloc (comme sur le croquis)."""
    d = pd.DataFrame({
        "Groupe": df[xcol].astype(str),
        "Modalité": df["Modalité"].astype(str),
        "Valeur": df[ycol].astype(float),
    })
    mods_c = [str(m) for m in mods_c]
    x_order = [str(m) for m in x_order]
    d["ordre"] = d["Modalité"].map({m: i for i, m in enumerate(mods_c)})
    d = d.sort_values(["Groupe", "ordre"])
    d["y1"] = d.groupby("Groupe")["Valeur"].cumsum()
    d["y0"] = d["y1"] - d["Valeur"]
    d["mid"] = (d["y0"] + d["y1"]) / 2
    is_pct = ycol == "Pourcentage"
    ngr = d["Groupe"].nunique()
    many = ngr > 6   # beaucoup de barres : on n'écrit que la valeur (le nom des modalités est dans la légende)
    if many:
        d["label"] = [f"{v:.0f}%" if is_pct else f"{int(round(v))}" for v in d["Valeur"]]
    else:
        d["label"] = [f"{m} · {v:.1f}%" if is_pct else f"{m} · {int(round(v))}"
                      for m, v in zip(d["Modalité"], d["Valeur"])]
    fsize = 13 if ngr <= 6 else (11 if ngr <= 12 else 9)
    ymax = max(float(d["y1"].max()), 1e-9)
    color = alt.Color("Modalité:N", sort=mods_c,
                      scale=alt.Scale(domain=mods_c, range=HIST_COLORS[: len(mods_c)]),
                      legend=alt.Legend(title="Modalité"))
    x = alt.X("Groupe:N", sort=x_order, title=xcol, axis=alt.Axis(labelAngle=0, labelFontSize=14))
    base = alt.Chart(d)
    bars = base.mark_bar(stroke="#111827", strokeWidth=1, width=alt.RelativeBandSize(0.85)).encode(
        x=x, y=alt.Y("y0:Q", title=ycol), y2="y1:Q", color=color,
        tooltip=["Groupe", "Modalité", "Valeur"])
    txt = (base.transform_filter(f"datum.Valeur >= {(0.04 if not many else 0.06) * ymax}")
           .mark_text(color="white", fontWeight="bold", fontSize=fsize)
           .encode(x=x, y="mid:Q", text="label:N"))
    return alt.layer(bars, txt).properties(height=440)


def tab_histogramme(data, config, variables):
    if len(variables) < 2:
        st.info("Veuillez sélectionner au moins deux variables.")
        return
    a, b, c = st.columns(3)
    v1 = a.selectbox("Variable 1 (ex. Oui/Non)", variables, index=0, key="hist_v1")
    v2 = b.selectbox("Variable 2 (ex. Jamais…Toujours)", variables, index=1, key="hist_v2")
    if v1 == v2:
        st.warning("Veuillez choisir deux variables différentes.")
        return
    mods1, mods2 = config[v1], config[v2]
    m1 = c.selectbox(f"Modalité de {v1}", ["(toutes)"] + list(mods1), key="hist_m1")
    pct = st.checkbox("Afficher en pourcentage (profil-ligne)", value=False, key="hist_pct")

    B = burt_block(data, config, v1, v2)
    if pct:
        B = B.div(B.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100
    ylab = "Pourcentage" if pct else "Effectif"

    if m1 != "(toutes)":
        df_plot = pd.DataFrame({v1: [m1] * len(mods2), "Modalité": list(mods2),
                                ylab: B.loc[m1, mods2].values.astype(float)})
        chart = stacked_bars(df_plot, v1, ylab, mods2, [m1])
    else:
        rows = []
        for r_ in mods1:
            for c_ in mods2:
                rows.append({v1: r_, "Modalité": c_, ylab: float(B.loc[r_, c_])})
        chart = stacked_bars(pd.DataFrame(rows), v1, ylab, mods2, mods1)
    st.altair_chart(chart, use_container_width=True)
    st.dataframe(B.round(1), width="stretch")

    # --- Courbes : variance / covariance / corrélation (même onglet) ---
    st.markdown("### Courbes : variance, covariance, corrélation")
    Nraw = burt_block(data, config, v1, v2)
    Nf = Nraw.loc[Nraw.sum(axis=1) > 0, Nraw.sum(axis=0) > 0]
    if min(Nf.shape) < 2:
        st.info("Il faut au moins 2 modalités non vides dans chaque variable pour ces courbes.")
        return
    res = nuages_inertie(Nraw)
    kind = st.radio("Courbe", ["Variance", "Covariance", "Corrélation"], horizontal=True, key="hist_kind")
    k1, k2 = st.columns(2)
    k1.markdown(f"**N(J) : coordonnées = modalités de {v1}**")
    k1.altair_chart(chart_moments(res["var_J"], res["cov_J"], res["cor_J"], kind), use_container_width=True)
    k2.markdown(f"**N(I) : coordonnées = modalités de {v2}**")
    k2.altair_chart(chart_moments(res["var_I"], res["cov_I"], res["cor_I"], kind), use_container_width=True)


# ----------------------------------------------------------------------------
# Table de distance (onglet à part, présentation normale)
# ----------------------------------------------------------------------------
LATEX_DIST_I = (r"d^2\left(f_j^{i},f_j^{i'}\right)=\left\|f_j^{i}-f_j^{i'}\right\|^2_{M=D^{-1}_{f.j}}"
                r"=\sum_{j=1}^{p}\frac{\left(f_j^{i}-f_j^{i'}\right)^2}{f_{.j}}")
LATEX_DIST_J = (r"d^2\left(f_i^{j},f_i^{j'}\right)=\left\|f_i^{j}-f_i^{j'}\right\|^2_{M=D^{-1}_{f_i.}}"
                r"=\sum_{i=1}^{n}\frac{\left(f_i^{j}-f_i^{j'}\right)^2}{f_{i.}}")


def _frac_str(f):
    return "0" if f == 0 else (str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}")


def tab_distance(data, config, variables, export_sheets):
    st.caption("Distance du χ² entre deux points d'un nuage : N(I) = modalités de la variable 1, N(J) = modalités de la variable 2.")
    if len(variables) < 2:
        st.info("Sélectionnez au moins deux variables.")
        return
    a, b = st.columns(2)
    v1 = a.selectbox("Variable 1 (nuage N(I))", variables, index=0, key="di_v1")
    v2 = b.selectbox("Variable 2 (nuage N(J))", variables, index=1, key="di_v2")
    if v1 == v2:
        st.warning("Veuillez choisir deux variables différentes.")
        return
    N = burt_block(data, config, v1, v2)
    N = N.loc[N.sum(axis=1) > 0, N.sum(axis=0) > 0]
    if min(N.shape) < 2:
        st.warning("Il faut au moins 2 modalités non vides dans chaque variable.")
        return
    det_I = cloud_detail(N.values, list(N.index), list(N.columns))
    det_J = cloud_detail(N.values.T, list(N.columns), list(N.index))

    for title, det, lat, key, sheet in [
        (f"Tableau de distance N(I) : modalités de {v1}", det_I, LATEX_DIST_I, "I", "Dist_NI"),
        (f"Tableau de distance N(J) : modalités de {v2}", det_J, LATEX_DIST_J, "J", "Dist_NJ"),
    ]:
        st.markdown("### " + title)
        st.latex(lat)
        dist = distance_matrix(det)
        st.dataframe(dist.round(4).style.background_gradient(cmap="Reds", axis=None), width="stretch")

        D = exact_distance_table(det)
        m = len(D)
        fr = pd.DataFrame([[_frac_str(D[i][j]) if j <= i else "" for j in range(m)] for i in range(m)],
                          index=det["pl"], columns=det["pl"])
        st.markdown("**Valeurs exactes (fractions)**")
        st.dataframe(fr, width="stretch")

        with st.expander("Calcul détaillé d'une distance", expanded=True):
            c1, c2 = st.columns(2)
            p1 = c1.selectbox("Point A", det["pl"], index=0, key=f"di_A_{key}")
            p2 = c2.selectbox("Point B", det["pl"], index=1, key=f"di_B_{key}")
            if p1 == p2:
                st.warning("Choisissez deux points différents.")
            else:
                st.latex(latex_distance(det, det["pl"].index(p1), det["pl"].index(p2)))
        with st.expander("Toutes les distances, calcul par calcul", expanded=False):
            for p in range(m):
                for q in range(p + 1, m):
                    st.latex(latex_distance(det, p, q))
        export_sheets[sheet] = (dist, True)


# ----------------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------------
@st.fragment(run_every=3)
def watch_file(path):
    """Toutes les 3 s : si le fichier a été enregistré dans Excel, on recharge tout."""
    try:
        m = os.path.getmtime(path)
    except OSError:
        return
    if m != st.session_state.get("mtime"):
        st.rerun()


def main():
    st.title("Analyse – Tableaux statistiques")

    with st.sidebar:
        st.header("Paramètres")
        mode = st.radio(
            "Source des données",
            ["Fichier sur l'ordinateur (mise à jour auto)", "Téléverser un fichier"],
        )
        local = mode.startswith("Fichier")
        if local:
            path = st.text_input("Chemin du fichier Excel", value="cars.csv")
            if not path or not os.path.isfile(path):
                st.error(f"Fichier introuvable : {os.path.abspath(path) if path else ''}")
                st.stop()
            src = path
            st.caption(f"Lecture de : {os.path.abspath(path)}")
            m_before = os.path.getmtime(path)
            st.session_state.setdefault("mtime", m_before)
            auto = st.checkbox("Actualisation automatique (3 s)", value=True)
            st.button("Actualiser depuis Excel")
            if auto:
                watch_file(path)
            out_path = Path(path).resolve().with_name(Path(path).stem + "_resultats.xlsx")
        else:
            up = st.file_uploader("Charger un fichier (Excel / CSV)", type=["xlsx", "xlsm", "xls", "csv"])
            if up is None:
                st.stop()
            src = up
            out_path = Path("resultats.xlsx").resolve()
        sheets = [None]
        try:
            src_bytes, src_name = _src_bytes(src)
            sheets = sheet_names_cached(src_bytes, src_name)
        except Exception as e:
            st.error(f"Impossible d'ouvrir le fichier : {e}")
            st.stop()
        sheet = st.selectbox("Feuille", sheets) if sheets[0] is not None else None

    try:
        df = load_table_cached(src_bytes, src_name, sheet)
    except Exception as e:
        st.error(f"Erreur de lecture (le fichier est peut-être en cours d'enregistrement) : {e}")
        st.stop()
    if local:
        st.session_state["mtime"] = m_before
    if df.empty:
        st.warning("Le tableau est vide. Vérifiez la feuille choisie.")
        st.stop()

    # --- Colonnes ------------------------------------------------------------
    st.subheader("Colonnes")
    c1, c2 = st.columns(2)
    id_col = c1.selectbox(
        "Colonne identifiant (ex. Chercheur)", ["(aucune)"] + list(df.columns), index=1 if len(df.columns) > 1 else 0
    )
    candidates = [c for c in df.columns if c != id_col]
    default_vars = [c for c in candidates if df[c].nunique(dropna=True) <= MAX_MODS]
    variables = c2.multiselect("Variables à analyser", candidates, default=default_vars)
    if not variables:
        st.info("Choisissez au moins une variable.")
        st.stop()

    data = prepare_data(df, variables, id_col)
    too_many = [v for v in variables if data[v].nunique() > MAX_MODS]
    if too_many:
        st.error(f"Ces colonnes ont plus de {MAX_MODS} modalités différentes (identifiant ? valeur continue ?) : "
                 + ", ".join(too_many) + ". Retirez-les des variables à analyser.")
        st.stop()

    # --- Type de chaque variable ---------------------------------------------
    st.subheader("Type de chaque variable")
    st.caption("Nominale : l'ordre n'a pas d'importance. Ordinale : glissez les modalités pour les ranger du plus petit au plus grand.")

    types, config, problems = {}, {}, []
    for v in variables:
        mods_found = sorted(data[v].dropna().unique().tolist(), key=lambda x: (len(x), x) if x.isdigit() else (0, x))
        box = st.container(border=True)
        a, b = box.columns([1, 3])
        t = a.radio(f"**{v}**", ["Nominale", "Ordinale"], key=f"type_{v}")
        types[v] = t
        if t == "Nominale":
            config[v] = get_sorted_mods(mods_found, t)
            b.write("Modalités : " + ", ".join(config[v]))
        else:
            order = order_sortable(b, f"order_{v}_{abs(hash(tuple(mods_found)))}", mods_found)
            if len(order) != len(mods_found):
                problems.append(v)
                b.warning(f"Ordre incomplet : {len(order)}/{len(mods_found)} modalités choisies.")
            else:
                b.success("Ordre : " + "  <  ".join(order))
            config[v] = order

    if problems:
        st.info("Complétez l'ordre des variables ordinales pour afficher les résultats : " + ", ".join(problems))
        st.stop()

    # --- Résultats ---------------------------------------------------------
    disj = build_disjunctive(data, config)
    coded = build_coded(data, config, types)
    # Les tableaux affichés sont ajoutés ici au fur et à mesure, puis exportés en Excel.
    export_sheets = {
        "Donnees": (data[variables], True),
        "Tbl_codage": (coded, True),
        "Disjonctif": (disj, True),
    }

    st.divider()
    st.subheader("Résultats")
    t1, t2, t3, t4, t4_vis, t4_nu, t4_dist, t5, t5_sem, t5_dis, t6 = st.tabs(
        [
            "Données",
            "Tableau de codage",
            "Tableau disjonctif (0/1)",
            "Table de contingence",
            "Histogramme",
            "Nuages & inertie",
            "Table de distance",
            "Table de Burt",
            "Table de semblance",
            "Table de dissemblance",
            "Exporter",
        ]
    )

    with t1:
        show_df(data[variables])
        st.caption(f"{len(data)} individus, {len(variables)} variables.")

    with t2:
        st.caption("Chaque réponse remplacée par son code (logique cumulée pour l'ordinal : ex. souvent -> 1110, toujours -> 1111).")
        show_df(coded, "YlGn")

    with t3:
        st.caption("Chaque ligne = un individu ; 1 si la modalité est choisie, sinon 0.")
        show_df(disj, "Blues")

    with t4:
        st.caption("Croisement des fréquences entre deux variables.")
        if len(variables) < 2:
            st.info("Sélectionnez au moins deux variables pour afficher le tableau de contingence.")
        else:
            col_v1, col_v2 = st.columns(2)
            sel_v1 = col_v1.selectbox("Première variable (Lignes)", variables, index=0)
            sel_v2 = col_v2.selectbox("Deuxième variable (Colonnes)", variables, index=1 if len(variables) > 1 else 0)

            if sel_v1 == sel_v2:
                st.warning("Veuillez choisir deux variables différentes.")
            else:
                ct_display = build_contingency_matrix(data, sel_v1, sel_v2, config[sel_v1], config[sel_v2])
                st.dataframe(
                    ct_display.style.background_gradient(cmap="Oranges", axis=None),
                    width="stretch"
                )
                export_sheets["Contingence"] = (ct_display, True)

    with t4_vis:
        st.subheader("Histogramme (à partir de la table de Burt)")
        tab_histogramme(data, config, variables)

    with t4_nu:
        st.subheader("Nuages N(I) et N(J) : inertie, variance, covariance, corrélation")
        tab_nuages(data, config, variables, export_sheets)

    with t4_dist:
        st.subheader("Table de distance")
        tab_distance(data, config, variables, export_sheets)

    with t5:
        st.caption(
            "B = ZᵀZ (Z = tableau disjonctif complet). La diagonale donne les effectifs des modalités. "
            "Choisissez les variables à inclure : les modalités sont groupées par variable en lignes et en colonnes."
        )
        burt_vars = st.multiselect(
            "Variables de la table de Burt",
            variables,
            default=variables,
            key=f"burt_vars_{abs(hash(tuple(variables)))}",
        )
        if not burt_vars:
            st.info("Choisissez au moins une variable.")
        else:
            disj_b = build_disjunctive(data, {v: config[v] for v in burt_vars})
            burt = pd.DataFrame(disj_b.values.T @ disj_b.values, index=disj_b.columns, columns=disj_b.columns)
            burt.index.names = [None, None]
            burt.columns.names = [None, None]
            group_colors = {v: PALETTE[i % len(PALETTE)] for i, v in enumerate(variables)}
            show_html(grouped_table_html(burt, burt_css(burt, group_colors), group_colors), burt)
            export_sheets["Burt"] = (burt, True)

    group_colors = {v: PALETTE[i % len(PALETTE)] for i, v in enumerate(variables)}

    with t5_sem:
        st.caption(
            "Ressemblance entre les modalités (mêmes lignes et colonnes que la table de Burt, groupées par variable). "
            "Chaque modalité est un vecteur 0/1 sur les n individus : a = individus qui ont les deux modalités, "
            "b et c = qui n'ont que l'une des deux, d = qui n'ont aucune, p = n."
        )
        proximity_tab(data, config, variables, group_colors, export_sheets, "sem",
                      SIM_INDICES, sim_parts, (34, 197, 94), "match", "Semblance")

    with t5_dis:
        st.caption(
            "Dissemblance entre les modalités : plus la valeur est grande, plus elles sont différentes "
            "(Hamming = nombre d'individus qui n'ont que l'une des deux)."
        )
        proximity_tab(data, config, variables, group_colors, export_sheets, "dis",
                      DIS_INDICES, dis_parts, (239, 68, 68), "ham", "Dissemblance")

    with t6:
        st.subheader("Exporter tous les tableaux")
        n_ind = len(data)
        incl = st.checkbox("Inclure les tableaux des individus (Données, codage, disjonctif)",
                           value=(n_ind <= 20000), key="exp_incl", disabled=(n_ind > 1_000_000))
        if n_ind > 20000:
            st.caption(f"{n_ind} individus : l'export des tableaux d'individus peut prendre plusieurs minutes.")
        if st.button("Préparer le fichier Excel"):
            skip = () if (incl and n_ind <= 1_000_000) else ("Donnees", "Tbl_codage", "Disjonctif")
            to_save = {k: v for k, v in export_sheets.items() if k not in skip}
            with st.spinner("Création du fichier Excel…"):
                st.session_state["xls_bytes"] = to_excel(to_save)
            try:
                out_path.write_bytes(st.session_state["xls_bytes"])
                st.success(f"Résultats enregistrés : {out_path}")
            except PermissionError:
                st.warning(f"Fermez {out_path.name} dans Excel pour qu'il puisse être mis à jour.")
            except Exception as e:
                st.warning(f"Enregistrement automatique impossible : {e}")
        if "xls_bytes" in st.session_state:
            st.download_button(
                "Télécharger résultats.xlsx",
                st.session_state["xls_bytes"],
                "resultats.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


main()
