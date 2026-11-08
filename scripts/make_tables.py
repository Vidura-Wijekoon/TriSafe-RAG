"""Generate every table and figure in the paper from results/*/results.json.

    python scripts/make_tables.py            # writes paper/trust2027/generated/*.tex and figures

Nothing in the paper's result tables is typed by hand: the LaTeX files
produced here are \\input into main.tex.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "paper" / "trust2027" / "generated"
FIG = ROOT / "paper" / "trust2027" / "figures"
LANGS = ["en", "si", "ta", "mixed"]
LANG_LABEL = {"en": "en", "si": "si", "ta": "ta", "mixed": "mixed"}


def f(x, nd=3, pct=False):
    if x is None:
        return "--"
    if pct:
        return f"{100 * x:.0f}"
    return f"{x:.{nd}f}"


def pval(p):
    if p is None:
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def ci(entry, nd=2):
    if not entry or entry.get("ci_lo") is None:
        return "--"
    return f"[{entry['ci_lo']:.{nd}f}, {entry['ci_hi']:.{nd}f}]"


def load(backend):
    p = ROOT / "results" / backend / "results.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def tex_escape(s: str) -> str:
    return s.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


# --------------------------------------------------------------------------- #
def table_main(R, name, caption_extra=""):
    """Headline table: answer rate, answerable accuracy, selective accuracy,
    unanswerable abstention, abstention F1, ECE, AUROC with bootstrap CIs."""
    order = ["B1_vanilla", "B2_grounding_only", "B3_abstention_only", "TriSafe_default", "TriSafe_lexical",
             "TriSafe_rrf", "TriSafe_devtuned", "TriSafe_perlang", "TriSafe_platt"]
    boot = R["bootstrap_test"]
    rows = []
    for k in order:
        if k not in R["systems"]:
            continue
        S = R["systems"][k]
        o = S["test"]["overall"]
        b = boot.get(k, {})
        catch = f"{f(o.get('catch_rate_wrong'),2)}/{f(o.get('retain_rate_correct'),2)}"
        rows.append(
            f"{tex_escape(S['label'])} & {f(o['answer_rate'],2)} & {f(o['acc_answerable'],3)} {ci(b.get('acc_answerable'))} & "
            f"{f(o['selective_accuracy'],3)} & {f(o['unanswerable_abstention'],2)} & {catch} & {f(o['abstention_f1'],3)} & "
            f"{f(o['ece_draft'],3)} {ci(b.get('ece_draft'))} & {f(o['auroc_draft'],3)} \\\\"
        )
    body = "\n    ".join(rows)
    n = R["n_test"]
    return rf"""\begin{{table*}}[t]
  \caption{{Main results on the {n}-query test split ({name} backend{caption_extra}). Ans.\ rate: fraction of queries answered. Acc.: accuracy on gold-answerable queries (abstention counts as an error). Sel.\ acc.: accuracy among answered answerable queries. Unans.\ abst.: fraction of unanswerable queries abstained on. Catch/Retain: share of wrong drafts on answerable queries that the policy abstained on, and share of correct drafts it delivered. ECE and AUROC are computed from the confidence score against draft correctness over all test queries (policy-independent). Brackets: 95\% paired bootstrap intervals over queries (2{{,}}000 resamples).}}
  \label{{tab:main-{name}}}
  \footnotesize
  \setlength{{\tabcolsep}}{{3.5pt}}
  \begin{{tabular}}{{lcccccccc}}
    \toprule
    System & Ans.\ rate & Acc.\ $\uparrow$ [95\% CI] & Sel.\ acc.\ $\uparrow$ & Unans.\ abst.\ $\uparrow$ & Catch/Retain & Abst.\ F1 $\uparrow$ & ECE $\downarrow$ [95\% CI] & AUROC $\uparrow$ \\
    \midrule
    {body}
    \bottomrule
  \end{{tabular}}
\end{{table*}}
"""


def table_external(R, name):
    order = ["TriSafe_default", "TriSafe_devtuned", "E1_verbalized", "E2_retrieval_gate", "E3_platt", "E4_isotonic",
             "E5_temperature", "E6_translate_retrieve", "E7_self_consistency"]
    boot = R["bootstrap_test"]
    rows = []
    for k in order:
        if k not in R["systems"]:
            continue
        S = R["systems"][k]
        o = S["test"]["overall"]
        b = boot.get(k, {})
        d = b.get("decision_accuracy", {}).get("delta_vs_ref", {}) if k != "B1_vanilla" else {}
        rows.append(
            f"{tex_escape(S['label'])} & {f(o['answer_rate'],2)} & {f(o['acc_answerable'],3)} & {f(o['selective_accuracy'],3)} & "
            f"{f(o['unanswerable_abstention'],2)} & {f(o['decision_accuracy'],3)} {ci(b.get('decision_accuracy'))} & "
            f"{f(o['ece_draft'],3)} & {f(o['aurc'],3)} \\\\"
        )
    body = "\n    ".join(rows)
    return rf"""\begin{{table*}}[t]
  \caption{{External baselines on the test split ({name} backend). All gates and thresholds were tuned on the development split with the same objective (decision accuracy, ties broken toward higher answer rate). Decision acc.: answerable queries answered correctly plus unanswerable queries abstained on, over all queries. AURC: area under the risk--coverage curve of the confidence score (lower is better). E1--E5 are two-way gates (answer/refuse); E6 uses the TriSafe policy on drafts produced after translating non-English queries with the same LLM.}}
  \label{{tab:external-{name}}}
  \footnotesize
  \setlength{{\tabcolsep}}{{4pt}}
  \begin{{tabular}}{{lccccccc}}
    \toprule
    System & Ans.\ rate & Acc.\ $\uparrow$ & Sel.\ acc.\ $\uparrow$ & Unans.\ abst.\ $\uparrow$ & Decision acc.\ $\uparrow$ [95\% CI] & ECE $\downarrow$ & AURC $\downarrow$ \\
    \midrule
    {body}
    \bottomrule
  \end{{tabular}}
\end{{table*}}
"""


def table_fairness(R, name):
    order = ["B1_vanilla", "B3_abstention_only", "TriSafe_default", "TriSafe_devtuned", "TriSafe_perlang"]
    boot = R["bootstrap_test"]
    rows = []
    for k in order:
        if k not in R["systems"]:
            continue
        S = R["systems"][k]
        pl = S["test"]["per_language"]
        g = S["test"]["gaps"]
        cells = []
        for lang in LANGS:
            v = pl.get(lang)
            cells.append(f"{f(v['acc_answerable'],2)}/{f(v['answer_rate'],2)}" if v else "--")
        b = boot.get(k, {})
        rows.append(
            f"{tex_escape(S['label'])} & " + " & ".join(cells)
            + f" & {f(g['gap_acc_answerable'],2)} {ci(b.get('gap_acc_answerable'))} & {f(g['gap_answer_rate'],2)} & {f(g['equalized_odds_gap'],2)} & {f(g['cross_lingual_consistency'],2)} \\\\"
        )
    body = "\n    ".join(rows)
    # counts
    any_sys = R["systems"]["TriSafe_default"]["test"]["per_language"]
    counts = ", ".join(f"{LANG_LABEL[l]}: {any_sys[l]['n']} ({any_sys[l]['n_unanswerable']} unans.)" for l in LANGS if l in any_sys)
    return rf"""\begin{{table*}}[t]
  \caption{{Per-language results on the test split ({name} backend). Each language cell is accuracy on answerable queries / answer rate, so a narrowing accuracy gap can be read against coverage. $\Delta A$: max--min accuracy gap over the three single-language groups and the code-switched group with at least five test queries; $\Delta R$: the same gap for answer rate; $\Delta EO$: max of the TPR and FPR gaps on the answer/abstain decision; CLC $= 1-\Delta A$. Test queries per language: {counts}.}}
  \label{{tab:fair-{name}}}
  \footnotesize
  \setlength{{\tabcolsep}}{{3.5pt}}
  \begin{{tabular}}{{lcccccccc}}
    \toprule
    System & en & si & ta & mixed & $\Delta A \downarrow$ [95\% CI] & $\Delta R \downarrow$ & $\Delta EO \downarrow$ & CLC $\uparrow$ \\
    \midrule
    {body}
    \bottomrule
  \end{{tabular}}
\end{{table*}}
"""


def table_sweep(R, name):
    rows = R["weight_sweep"]
    head = " & ".join(f"{r['w_gr']:.1f}" for r in rows)
    dev = " & ".join(f(r["dev_ece"], 3) for r in rows)
    test = " & ".join(f(r["test_ece"], 3) for r in rows)
    dauc = " & ".join(f(r["dev_auroc"], 3) for r in rows)
    tauc = " & ".join(f(r["test_auroc"], 3) for r in rows)
    cols = "c" * len(rows)
    return rf"""\begin{{table}}[t]
  \caption{{Sensitivity to the grounding weight $w_{{\mathrm{{gr}}}}$ ({name} backend), with the remaining weight redistributed in the default proportions. Reported separately on the development split (used to choose defaults) and the test split.}}
  \label{{tab:sweep-{name}}}
  \footnotesize
  \setlength{{\tabcolsep}}{{3pt}}
  \begin{{tabular}}{{l{cols}}}
    \toprule
    $w_{{\mathrm{{gr}}}}$ & {head} \\
    \midrule
    ECE (dev) & {dev} \\
    ECE (test) & {test} \\
    AUROC (dev) & {dauc} \\
    AUROC (test) & {tauc} \\
    \bottomrule
  \end{{tabular}}
\end{{table}}
"""


def table_cases(RN, RO, RL=None):
    """Three domain slices, both backends, B1 vs TriSafe (dev-tuned)."""
    dom_label = {"civil": "CS1 Civil documents", "health": "CS2 Public health", "disaster": "CS3 Disaster response"}
    rows = []
    for dom in ("civil", "health", "disaster"):
        for bname, R in (("offline", RO), ("neural", RN), ("neural7b", RL)):
            if R is None:
                continue
            for k, lab in (("B1_vanilla", "B1"), ("TriSafe_devtuned", "TriSafe")):
                S = R["systems"][k]["test_by_domain"].get(dom)
                if not S:
                    continue
                o = S["overall"]
                rows.append(
                    f"{dom_label[dom]} & {bname} & {lab} & {o['n']} ({o['n_unanswerable']}) & {f(o['answer_rate'],2)} & "
                    f"{f(o['acc_answerable'],3)} & {f(o['selective_accuracy'],3)} & {f(o['unanswerable_abstention'],2)} & {f(o['abstention_f1'],3)} \\\\"
                )
        rows.append("\\addlinespace")
    body = "\n    ".join(rows)
    return rf"""\begin{{table*}}[t]
  \caption{{Case studies as domain slices of the test split, for the offline reproduction backend (hash embedder, extractive EchoLLM, lexical grounding) and the neural backend (multilingual-e5-base, Qwen2.5-1.5B-Instruct, mDeBERTa NLI grounding). $n$ (unans.): test queries in the slice and how many are gold-unanswerable. TriSafe uses the thresholds tuned on the development split.}}
  \label{{tab:cases}}
  \footnotesize
  \setlength{{\tabcolsep}}{{4pt}}
  \begin{{tabular}}{{lllcccccc}}
    \toprule
    Case & Backend & System & $n$ (unans.) & Ans.\ rate & Acc.\ $\uparrow$ & Sel.\ acc.\ $\uparrow$ & Unans.\ abst.\ $\uparrow$ & Abst.\ F1 $\uparrow$ \\
    \midrule
    {body}
    \bottomrule
  \end{{tabular}}
\end{{table*}}
"""


def table_tests(R, name):
    pt = R["paired_tests_vs_B1"]
    rows = []
    for k, v in pt.items():
        if k not in R["systems"]:
            continue
        rows.append(f"{tex_escape(R['systems'][k]['label'])} & {pval(v.get('correct'))} & {pval(v.get('abstained'))} \\\\")
    body = "\n    ".join(rows)
    return rf"""\begin{{table}}[t]
  \caption{{Paired sign-flip permutation tests against B1 on the test split ({name} backend): two-sided $p$-values for the per-query difference in correctness (answerable queries) and in abstention (all queries).}}
  \label{{tab:tests-{name}}}
  \footnotesize
  \begin{{tabular}}{{lcc}}
    \toprule
    System & $p$ (correct) & $p$ (abstained) \\
    \midrule
    {body}
    \bottomrule
  \end{{tabular}}
\end{{table}}
"""


def macros(RN, RO, RL=None):
    """Numbers quoted in the running text, as LaTeX macros.

    Prefixes: N = neural (1.5B generator), L = neural7b (7B generator), O = offline.
    """
    out = []

    def m(name, val):
        out.append(f"\\newcommand{{\\{name}}}{{{val}}}")

    for tag, R in (("N", RN), ("O", RO), ("L", RL)):
        if R is None:
            continue
        for name, val in seed_macros(R).items():
            m(f"{tag}{name}", val)
        b1 = R["systems"]["B1_vanilla"]["test"]
        td = R["systems"]["TriSafe_default"]["test"]
        tt = R["systems"]["TriSafe_devtuned"]["test"]
        tp = R["systems"].get("TriSafe_perlang", {}).get("test")
        m(f"{tag}ntest", R["n_test"]); m(f"{tag}ndev", R["n_dev"]); m(f"{tag}nq", R["n_queries"])
        m(f"{tag}BoneAns", f(b1["overall"]["answer_rate"], 2)); m(f"{tag}BoneAcc", f(b1["overall"]["acc_answerable"], 3))
        m(f"{tag}BoneEce", f(b1["overall"]["ece_draft"], 3)); m(f"{tag}BoneGap", f(b1["gaps"]["gap_acc_answerable"], 2))
        m(f"{tag}BoneUnans", f(b1["overall"]["unanswerable_abstention"], 2))
        m(f"{tag}DefAns", f(td["overall"]["answer_rate"], 2)); m(f"{tag}DefAcc", f(td["overall"]["acc_answerable"], 3))
        m(f"{tag}DefEce", f(td["overall"]["ece_draft"], 3)); m(f"{tag}DefGap", f(td["gaps"]["gap_acc_answerable"], 2))
        m(f"{tag}DefUnans", f(td["overall"]["unanswerable_abstention"], 2))
        m(f"{tag}TunAns", f(tt["overall"]["answer_rate"], 2)); m(f"{tag}TunAcc", f(tt["overall"]["acc_answerable"], 3))
        m(f"{tag}TunSel", f(tt["overall"]["selective_accuracy"], 3)); m(f"{tag}TunEce", f(tt["overall"]["ece_draft"], 3))
        m(f"{tag}TunGap", f(tt["gaps"]["gap_acc_answerable"], 2)); m(f"{tag}TunUnans", f(tt["overall"]["unanswerable_abstention"], 2))
        m(f"{tag}TunF1", f(tt["overall"]["abstention_f1"], 3)); m(f"{tag}TunEO", f(tt["gaps"]["equalized_odds_gap"], 2))
        m(f"{tag}BoneF1", f(b1["overall"]["abstention_f1"], 3)); m(f"{tag}BoneEO", f(b1["gaps"]["equalized_odds_gap"], 2))
        m(f"{tag}BoneSel", f(b1["overall"]["selective_accuracy"], 3))
        if tp:
            m(f"{tag}PlAns", f(tp["overall"]["answer_rate"], 2)); m(f"{tag}PlGap", f(tp["gaps"]["gap_acc_answerable"], 2))
            m(f"{tag}PlGapR", f(tp["gaps"]["gap_answer_rate"], 2)); m(f"{tag}PlAcc", f(tp["overall"]["acc_answerable"], 3))
        th = R["tuned_thresholds"]
        m(f"{tag}TauAns", f(th["answer_threshold"], 2)); m(f"{tag}TauRef", f(th["refuse_threshold"], 2)); m(f"{tag}TauG", f(th["min_support"], 2))
        sc = R["retrieval_scale"]
        m(f"{tag}RetLo", f(sc["ret_lo"], 3)); m(f"{tag}RetHi", f(sc["ret_hi"], 3)); m(f"{tag}MarScale", f(sc["margin_scale"], 1))
        bi = R.get("backend_info", {})
        m(f"{tag}nchunks", bi.get("n_chunks", "--")); m(f"{tag}ndocs", bi.get("n_docs", "--"))
        ma = R.get("model_abstention_rate", {})
        for l in LANGS:
            if l in ma:
                m(f"{tag}ModelAbst{l}", f(ma[l], 2))
        da = R.get("draft_accuracy_test", {})
        for l in LANGS:
            if l in da:
                m(f"{tag}DraftAcc{l}", f(da[l], 2))
        rs = R.get("retrieval_stats_test", {})
        for l in LANGS:
            if l in rs and rs[l].get("gold_in_topk") is not None:
                m(f"{tag}RetHit{l}", f(rs[l]["gold_in_topk"], 2))
        trs = R.get("translate_retrieval_stats", {})
        for l in LANGS:
            if l in trs and trs[l].get("gold_in_topk") is not None:
                m(f"{tag}TransRetHit{l}", f(trs[l]["gold_in_topk"], 2))
        rrf = R["systems"].get("TriSafe_rrf", {}).get("test", {}).get("overall", {})
        if rrf:
            m(f"{tag}RrfAns", f(rrf.get("answer_rate"), 2))
        words = {"E1": "Eone", "E2": "Etwo", "E3": "Ethree", "E4": "Efour", "E5": "Efive", "E6": "Esix", "E7": "Eseven"}
        for k in ("E1_verbalized", "E2_retrieval_gate", "E3_platt", "E4_isotonic", "E5_temperature", "E6_translate_retrieve",
                  "E7_self_consistency", "TriSafe_platt", "TriSafe_lexical", "TriSafe_rrf", "B3_abstention_only", "B2_grounding_only"):
            if k in R["systems"]:
                o = R["systems"][k]["test"]["overall"]
                key = words.get(k.split("_")[0], {"TriSafe_platt": "TPlatt", "TriSafe_lexical": "Lex", "TriSafe_rrf": "Rrf",
                                                  "B3_abstention_only": "Bthree", "B2_grounding_only": "Btwo"}.get(k, "X"))
                m(f"{tag}{key}Dec", f(o["decision_accuracy"], 3)); m(f"{tag}{key}Ece", f(o["ece_draft"], 3))
                m(f"{tag}{key}Ans", f(o["answer_rate"], 2)); m(f"{tag}{key}Acc", f(o["acc_answerable"], 3))
                m(f"{tag}{key}Unans", f(o["unanswerable_abstention"], 2)); m(f"{tag}{key}Sel", f(o["selective_accuracy"], 3))
                m(f"{tag}{key}Auroc", f(o["auroc_draft"], 3)); m(f"{tag}{key}Fone", f(o["abstention_f1"], 3))
        for key, S in (("Tun", tt), ("Def", td), ("Bone", b1)):
            m(f"{tag}{key}Catch", f(S["overall"].get("catch_rate_wrong"), 2)); m(f"{tag}{key}Retain", f(S["overall"].get("retain_rate_correct"), 2))
            m(f"{tag}{key}Auroc", f(S["overall"].get("auroc_draft"), 3)); m(f"{tag}{key}Brier", f(S["overall"].get("brier_draft"), 3))
        m(f"{tag}DraftAccAll", f(tt["overall"].get("draft_acc_answerable"), 3))
        m(f"{tag}DefDec", f(td["overall"]["decision_accuracy"], 3)); m(f"{tag}TunDec", f(tt["overall"]["decision_accuracy"], 3))
        m(f"{tag}BoneDec", f(b1["overall"]["decision_accuracy"], 3))
        pt = R["paired_tests_vs_B1"].get("TriSafe_devtuned", {})
        m(f"{tag}TunPcorrect", pval(pt.get("correct"))); m(f"{tag}TunPabst", pval(pt.get("abstained")))
        bt = R["bootstrap_test"]["TriSafe_devtuned"]
        m(f"{tag}TunDeltaAccCI", ci(bt["acc_answerable"]["delta_vs_ref"])); m(f"{tag}TunDeltaEceCI", ci(bt["ece_draft"]["delta_vs_ref"], 3))
        m(f"{tag}TunDeltaGapCI", ci(bt["gap_acc_answerable"]["delta_vs_ref"]))
        m(f"{tag}TunDeltaAcc", f(bt["acc_answerable"]["delta_vs_ref"]["point"], 3)); m(f"{tag}TunDeltaEce", f(bt["ece_draft"]["delta_vs_ref"]["point"], 3))
        m(f"{tag}TunDeltaGap", f(bt["gap_acc_answerable"]["delta_vs_ref"]["point"], 2))
        m(f"{tag}TunDeltaUnansCI", ci(bt["unanswerable_abstention"]["delta_vs_ref"]))
        m(f"{tag}TunDeltaUnans", f(bt["unanswerable_abstention"]["delta_vs_ref"]["point"], 2))
        m(f"{tag}seeds", len(R["seeds"]))
        # development-split decision accuracy: what the tuning objective saw
        for key, k in (("Tun", "TriSafe_devtuned"), ("Def", "TriSafe_default"), ("Bone", "B1_vanilla"), ("Etwo", "E2_retrieval_gate")):
            dv = R["systems"].get(k, {}).get("dev", {}).get("overall", {})
            if dv:
                m(f"{tag}{key}DevDec", f(dv["decision_accuracy"], 3)); m(f"{tag}{key}DevAns", f(dv["answer_rate"], 2))
        # per-language tuned answer thresholds
        for l, th_l in (R.get("per_language_thresholds") or {}).items():
            if "answer_threshold" in th_l:
                m(f"{tag}PlTau{l}", f(th_l["answer_threshold"], 2))
        # case-study slice sizes (test split)
        dom_key = {"civil": "Civil", "health": "Health", "disaster": "Disaster"}
        for dom, dk in dom_key.items():
            S = tt.get("test_by_domain", R["systems"]["TriSafe_devtuned"].get("test_by_domain", {})).get(dom)
            if S:
                m(f"{tag}Cs{dk}N", S["overall"]["n"]); m(f"{tag}Cs{dk}U", S["overall"]["n_unanswerable"])
        # abstention reasons of the dev-tuned system on the test split
        reasons = tt["overall"].get("abstention_reasons") or {}
        n_abst = sum(reasons.values()) or 1
        m(f"{tag}TunAbstN", sum(reasons.values()))
        for rk in ("model_abstained", "weak_grounding", "low_confidence", "borderline_evidence", "no_evidence_retrieved"):
            m(f"{tag}TunReason{rk.replace('_', '')}", reasons.get(rk, 0))
            m(f"{tag}TunReasonPct{rk.replace('_', '')}", f(reasons.get(rk, 0) / n_abst, 2))
        for l in LANGS:
            if l in tt["per_language"]:
                m(f"{tag}TunAcc{l}", f(tt["per_language"][l]["acc_answerable"], 2)); m(f"{tag}TunAns{l}", f(tt["per_language"][l]["answer_rate"], 2))
                m(f"{tag}BoneAcc{l}", f(b1["per_language"][l]["acc_answerable"], 2)); m(f"{tag}BoneAns{l}", f(b1["per_language"][l]["answer_rate"], 2))
        # what the policy adds on top of the generator's own abstention (test split, answerable queries):
        # wrong drafts that B1 delivered and TriSafe withheld ("caught"), correct drafts B1 delivered and TriSafe withheld ("lost")
        backend = R["backend"]
        ra = _load_records(backend, "TriSafe_devtuned"); rb = _load_records(backend, "B1_vanilla")
        if ra and rb:
            test_ids = set(json.loads((ROOT / "data" / "benchmark" / "split.json").read_text(encoding="utf-8"))["test"])
            caught = {l: 0 for l in LANGS}; lost = {l: 0 for l in LANGS}
            for qid, r in ra.items():
                if qid not in test_ids or not r["gold_answerable"] or qid not in rb:
                    continue
                if rb[qid]["decision"] == "answer" and r["decision"] != "answer":
                    (lost if r["draft_correct"] else caught)[r["language"]] += 1
            for l in LANGS:
                m(f"{tag}Caught{l}", caught[l]); m(f"{tag}Lost{l}", lost[l])
            m(f"{tag}CaughtAll", sum(caught.values())); m(f"{tag}LostAll", sum(lost.values()))
            # per-language anatomy of the answerable test queries: how many the generator
            # answered at all, how many of those drafts were right, and how often the gold
            # answer was in the retrieved passages when the generator abstained anyway
            drafts = _load_drafts(backend)
            if drafts:
                for l in LANGS:
                    ids = [q for q, r in ra.items() if q in test_ids and r["gold_answerable"] and r["language"] == l and q in drafts]
                    n_draft = sum(1 for q in ids if not drafts[q]["model_abstained"])
                    n_ok = sum(1 for q in ids if not drafts[q]["model_abstained"] and ra[q]["draft_correct"])
                    abst = [q for q in ids if drafts[q]["model_abstained"]]
                    gold_in = sum(1 for q in abst if any(a and a in (h.get("text") or "") for h in drafts[q].get("hits", []) for a in (drafts[q].get("gold_answers") or [])))
                    m(f"{tag}AnsN{l}", len(ids)); m(f"{tag}DraftsN{l}", n_draft); m(f"{tag}DraftsOk{l}", n_ok)
                    m(f"{tag}GenAbstN{l}", len(abst)); m(f"{tag}GenAbstGoldIn{l}", gold_in)
    return "\n".join(out) + "\n"


def figures(R, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    # reliability diagrams
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.9), sharey=True)
    for ax, k, title in zip(axes, ("B1_vanilla", "TriSafe_devtuned", "TriSafe_platt"),
                            ("B1 vanilla (retrieval score)", "TriSafe-RAG (weighted sum)", "TriSafe-RAG + Platt")):
        if k not in R["systems"]:
            ax.set_visible(False)
            continue
        bins = R["systems"][k]["reliability_test"]
        ece = R["systems"][k]["test"]["overall"]["ece_draft"]
        xs = [(b["lo"] + b["hi"]) / 2 for b in bins]
        accs = [b["acc"] if b["acc"] is not None else 0.0 for b in bins]
        ns = [b["n"] for b in bins]
        ax.bar(xs, accs, width=0.1, edgecolor="black", color="#8fb3d9", label="empirical")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        for x, a, n in zip(xs, accs, ns):
            if n:
                ax.text(x, a + 0.02, str(n), ha="center", fontsize=6)
        ax.set_title(f"{title}\nECE = {ece:.3f}", fontsize=9)
        ax.set_xlabel("Predicted confidence", fontsize=8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("Empirical accuracy (draft)", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / f"reliability_{name}.pdf"); fig.savefig(FIG / f"reliability_{name}.png", dpi=200)
    plt.close(fig)

    # per-language accuracy vs answer rate
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    markers = {"B1_vanilla": "o", "TriSafe_default": "s", "TriSafe_devtuned": "^", "TriSafe_perlang": "D"}
    colors = {"en": "#1f77b4", "si": "#d62728", "ta": "#2ca02c", "mixed": "#9467bd"}
    for k, mk in markers.items():
        if k not in R["systems"]:
            continue
        pl = R["systems"][k]["test"]["per_language"]
        for lang in LANGS:
            if lang in pl:
                ax.scatter(pl[lang]["answer_rate"], pl[lang]["acc_answerable"], marker=mk, color=colors[lang], s=36,
                           edgecolor="black", linewidth=0.5)
    from matplotlib.lines import Line2D
    short = {"B1_vanilla": "B1 vanilla", "TriSafe_default": "TriSafe (default thresholds)",
             "TriSafe_devtuned": "TriSafe (dev-tuned)", "TriSafe_perlang": "TriSafe (per-language, dev)"}
    h1 = [Line2D([], [], marker=mk, color="grey", linestyle="", label=short[k]) for k, mk in markers.items() if k in R["systems"]]
    h2 = [Line2D([], [], marker="o", color=c, linestyle="", label=l) for l, c in colors.items()]
    ax.legend(handles=h1 + h2, fontsize=6, loc="upper left", ncol=2, frameon=False)
    ax.set_xlabel("Answer rate", fontsize=8); ax.set_ylabel("Accuracy on answerable queries", fontsize=8)
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(FIG / f"lang_tradeoff_{name}.pdf"); fig.savefig(FIG / f"lang_tradeoff_{name}.png", dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RN, RO, RL = load("neural"), load("offline"), load("neural7b")
    for name, R in (("neural", RN), ("neural7b", RL), ("offline", RO)):
        if R is None:
            continue
        (OUT / f"table_main_{name}.tex").write_text(table_main(R, name), encoding="utf-8")
        (OUT / f"table_external_{name}.tex").write_text(table_external(R, name), encoding="utf-8")
        (OUT / f"table_fairness_{name}.tex").write_text(table_fairness(R, name), encoding="utf-8")
        (OUT / f"table_sweep_{name}.tex").write_text(table_sweep(R, name), encoding="utf-8")
        (OUT / f"table_tests_{name}.tex").write_text(table_tests(R, name), encoding="utf-8")
        (OUT / f"table_seeds_{name}.tex").write_text(table_seeds(R, name), encoding="utf-8")
        figures(R, name)
    (OUT / "table_cases.tex").write_text(table_cases(RN, RO, RL), encoding="utf-8")
    (OUT / "table_scaling.tex").write_text(table_scaling(RN, RL), encoding="utf-8")
    (OUT / "macros.tex").write_text(macros(RN, RO, RL), encoding="utf-8")
    print("wrote", sorted(p.name for p in OUT.iterdir()))




# --------------------------------------------------------------------------- #
# Qualitative material generated from the per-query records
# --------------------------------------------------------------------------- #

def _load_records(backend: str, system: str):
    p = ROOT / "results" / backend / "records" / f"{system}.jsonl"
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out.setdefault(r["id"], r)  # first seed
    return out


def _load_drafts(backend: str, seed: int = 0):
    p = ROOT / "results" / backend / f"drafts_seed{seed}.jsonl"
    if not p.exists():
        return {}
    return {json.loads(l)["id"]: json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}


_SI_RUN = re.compile("[\u0d80-\u0dff][\u0d80-\u0dff\\s\u200d.,;:0-9]*")
_TA_RUN = re.compile("[\u0b80-\u0bff][\u0b80-\u0bff\\s\u200d.,;:0-9]*")


def _tex_text(t: str) -> str:
    return tex_escape(t).replace("#", "\\#").replace("$", "\\$").replace("{", "\\{").replace("}", "\\}")


def script_wrap(text: str) -> str:
    """Wrap Sinhala and Tamil runs in \\textsi{...} / \\textta{...} (defined in
    preamble.tex: Noto Sans fonts under XeLaTeX, a bracketed marker under
    pdfLaTeX).  Other non-ASCII characters are dropped so that the file is safe
    for both engines."""
    out, i = [], 0
    for m in re.finditer("%s|%s" % (_SI_RUN.pattern, _TA_RUN.pattern), text):
        out.append(_tex_text(text[i:m.start()].encode("ascii", "ignore").decode("ascii")))
        run = _tex_text(m.group(0).strip())
        cmd = "textsi" if _SI_RUN.match(m.group(0)) else "textta"
        out.append(f"\\{cmd}{{{run}}}" + (" " if m.group(0).endswith(" ") else ""))
        i = m.end()
    out.append(_tex_text(text[i:].encode("ascii", "ignore").decode("ascii")))
    return " ".join("".join(out).split())


def _short(text: str, n: int = 110) -> str:
    """Excerpt of at most n characters with Sinhala/Tamil script kept and
    font-wrapped (see script_wrap)."""
    t = " ".join((text or "").split())
    truncated = len(t) > n
    t = t[: n - 1] if truncated else t
    return script_wrap(t) + ("\\ldots{}" if truncated else "")


def example_items() -> str:
    """One answerable and one unanswerable benchmark item in all four query forms."""
    items = {it["group"]: it for it in json.loads((ROOT / "data" / "benchmark" / "items.json").read_text(encoding="utf-8"))["items"]}
    rows = []
    for gid, note in (("cd001", "answerable (civil documents)"), ("ph027", "false premise, answerable with a refusal of the premise (public health)"), ("dr018", "unanswerable: only support is the stale bulletin (disaster response)")):
        it = items.get(gid)
        if not it:
            continue
        gold = it.get("gold", {})
        gold_all = gold.get("all") or []
        rows.append(rf"\multicolumn{{2}}{{l}}{{\textbf{{{gid}}}: {note}}} \\")
        for lang in ("en", "si", "ta", "mixed"):
            q = it["queries"].get(lang)
            if not q:
                continue
            g = gold.get(lang) or gold_all
            gtxt = "; ".join(script_wrap(a) for a in g[:4]) if g else "(no gold answer: any answer is a failure)"
            rows.append(rf"\quad {LANG_LABEL.get(lang, lang)} & {script_wrap(q)} \\")
            if lang == "en" or gold.get(lang):
                rows.append(rf"\quad \emph{{gold}} & {gtxt} \\")
        rows.append(r"\addlinespace")
    body = "\n    ".join(rows)
    return rf"""\begin{{table}}[t]
  \caption{{Three SL-PubServ-QA items in all query forms. Gold aliases are shared across languages unless a language-specific alias list is given; numeric aliases are matched as whole numbers. Sinhala and Tamil are typeset from the benchmark file; the code-switched form is romanised Sinhala with English terms.}}
  \label{{tab:examples}}
  \footnotesize
  \setlength{{\tabcolsep}}{{3pt}}
  \begin{{tabular}}{{lp{{0.78\columnwidth}}}}
    \toprule
    {body}
    \bottomrule
  \end{{tabular}}
\end{{table}}
"""


def case_probes(backend: str):
    """Decisions of B1 and dev-tuned TriSafe on the adversarial probes, per language."""
    b1 = _load_records(backend, "B1_vanilla")
    ts = _load_records(backend, "TriSafe_devtuned")
    if not b1 or not ts:
        return "% no records yet\n"
    probes = [
        ("dr017", "current DMC hotline (answerable; stale bulletin says 110)"),
        ("dr018", "rely on past-season helpline 110? (unanswerable)"),
        ("dr019", "past-season shelter still open? (unanswerable)"),
        ("dr020", "past-season shelter hours (unanswerable)"),
        ("ph027", "does lime juice cure dengue? (false premise)"),
        ("ph028", "treat child's flu with antibiotics? (false premise)"),
        ("ph006", "papaya leaf juice a proven cure? (misinformation)"),
        ("ph003", "aspirin for dengue? (safety)"),
    ]
    sym = {"answer": "A", "defer": "D", "refuse": "R"}
    rows = []
    for group, label in probes:
        cells = []
        for lang in ("en", "si", "ta", "mx"):
            qid = f"{group}-{lang}"
            if qid not in b1:
                cells.append("--")
                continue
            r1, r2 = b1[qid], ts[qid]
            def mark(r):
                s = sym[r["decision"]]
                if r["gold_answerable"]:
                    s += "\checkmark" if r["correct"] else ("$\times$" if r["decision"] == "answer" else "")
                return s
            cells.append(f"{mark(r1)}/{mark(r2)}")
        rows.append(f"{tex_escape(label)} & " + " & ".join(cells) + " \\\\")
    body = "\n    ".join(rows)
    split_note = "probes in the development split are marked with an asterisk in the repository split file" 
    return rf"""\begin{{table}}[t]
  \caption{{Adversarial probes ({backend} backend): decision of B1 / dev-tuned TriSafe-RAG per language (A answer, D defer, R refuse; \checkmark correct answer, $\times$ wrong answer). For unanswerable probes any A is a failure. Both development and test probes are shown.}}
  \label{{tab:probes-{backend}}}
  \footnotesize
  \setlength{{\tabcolsep}}{{3pt}}
  \begin{{tabular}}{{lcccc}}
    \toprule
    Probe & en & si & ta & mixed \\
    \midrule
    {body}
    \bottomrule
  \end{{tabular}}
\end{{table}}
"""


def qualitative(backend: str):
    """Four real records: kept correct, caught wrong, caught unanswerable, lost correct."""
    b1 = _load_records(backend, "B1_vanilla")
    ts = _load_records(backend, "TriSafe_devtuned")
    drafts = _load_drafts(backend)
    if not b1 or not ts or not drafts:
        return "% no records yet\n"

    test_ids = set(json.loads((ROOT / "data" / "benchmark" / "split.json").read_text(encoding="utf-8"))["test"])

    def pick(cond, prefer=("si", "ta", "mixed", "en")):
        """First matching test record, trying the preferred languages in order."""
        for lang in prefer:
            for qid in sorted(ts):
                r, d = ts[qid], drafts.get(qid)
                if qid in test_ids and d and r["language"] == lang and cond(r, b1[qid], d):
                    return qid, r, d
        return None

    def has_gold_in_hits(d):
        gold = d.get("gold_answers") or []
        return any(a and a in (h.get("text") or "") for h in d.get("hits", []) for a in gold)

    # (title, condition, language preference).  Sinhala first wherever Sinhala records exist,
    # so that the examples cover the script the generator serves worst.
    cases = [
        ("Generator abstention although the evidence was retrieved",
         lambda r, b, d: r["gold_answerable"] and d["model_abstained"] and has_gold_in_hits(d), ("si", "ta", "mixed")),
        ("Correct draft delivered",
         lambda r, b, d: r["gold_answerable"] and r["decision"] == "answer" and r["correct"], ("si", "ta", "mixed", "en")),
        ("Wrong draft caught",
         lambda r, b, d: r["gold_answerable"] and r["decision"] != "answer" and not d["model_abstained"] and not r["draft_correct"] and b["decision"] == "answer", ("si", "ta", "mixed", "en")),
        ("Wrong draft caught (draft written in the query language)",
         lambda r, b, d: r["gold_answerable"] and r["decision"] != "answer" and not d["model_abstained"] and not r["draft_correct"] and b["decision"] == "answer" and _SI_RUN.search(d["draft_text"] or "") is not None, ("si",)),
        ("Unanswerable query refused by the policy",
         lambda r, b, d: not r["gold_answerable"] and r["decision"] != "answer" and b["decision"] == "answer", ("si", "ta", "mixed", "en")),
        ("Correct draft lost",
         lambda r, b, d: r["gold_answerable"] and r["decision"] != "answer" and r["draft_correct"] and b["correct"], ("si", "ta", "mixed", "en")),
    ]
    items, used = [], set()
    for title, cond, prefer in cases:
        got = pick(lambda r, b, d, c=cond: c(r, b, d) and d["id"] not in used, prefer)
        if not got:
            continue
        qid, r, d = got
        used.add(qid)
        sig = r["signals"]
        evid = " Gold answer present in the retrieved passages: " + ("yes" if has_gold_in_hits(d) else "no") + "."
        items.append(
            rf"\item \textbf{{{title}}} ({qid}, {LANG_LABEL.get(d['language'], d['language'])}). Query: \emph{{{_short(d['query'], 120)}}} "
            f"Draft: ``{_short(d['draft_text'])}''{evid} "
            f"Signals: retrieval {sig[0]:.2f}, margin {sig[1]:.2f}, grounding {sig[2]:.2f}, citation {sig[3]:.2f}, language match {int(sig[4])}; "
            rf"confidence {r['confidence']:.2f}; decision \textsc{{{r['decision']}}} ({tex_escape(r['reason'])}); B1 decision \textsc{{{b1[qid]['decision']}}}."
        )
    if not items:
        return "% no qualitative items\n"
    return ("The following records are taken verbatim from the test-split audit log of the dev-tuned system (" + backend +
            " backend); queries and drafts are shown in their original script, Sinhala records first wherever one exists.\n\\begin{itemize}\n"
            + "\n".join(items) + "\n\\end{itemize}\n")


def write_qualitative():
    backend = "neural" if (ROOT / "results" / "neural" / "records").exists() else "offline"
    (OUT / "qualitative.tex").write_text(qualitative(backend), encoding="utf-8")
    (OUT / "case_probes.tex").write_text(case_probes(backend), encoding="utf-8")
    (OUT / "table_examples.tex").write_text(example_items(), encoding="utf-8")



# --------------------------------------------------------------------------- #
# Seeds: per-seed test metrics from the per-query records, and the 1.5B vs 7B
# scaling comparison
# --------------------------------------------------------------------------- #

SEED_KEYS = ("B1_vanilla", "TriSafe_default", "TriSafe_devtuned", "E1_verbalized", "E3_platt", "E4_isotonic", "E7_self_consistency")
SEED_METRICS = (("answer_rate", "Ans", 2), ("acc_answerable", "Acc", 3), ("selective_accuracy", "Sel", 3),
                ("unanswerable_abstention", "Unans", 2), ("decision_accuracy", "Dec", 3), ("ece_draft", "Ece", 3),
                ("auroc_draft", "Auroc", 3), ("auroc_answered", "AurocAns", 3))
SHORT_KEY = {"B1_vanilla": "Bone", "TriSafe_default": "Def", "TriSafe_devtuned": "Tun", "E1_verbalized": "Eone",
             "E3_platt": "Ethree", "E4_isotonic": "Efour", "E7_self_consistency": "Eseven"}


def _per_seed(R, system):
    """{seed: summarise(test records of that seed)} for one system of one backend."""
    import sys
    sys.path.insert(0, str(ROOT))
    from src.eval.metrics import summarise
    p = ROOT / "results" / R["backend"] / "records" / f"{system}.jsonl"
    if not p.exists():
        return {}
    test_ids = set(json.loads((ROOT / "data" / "benchmark" / "split.json").read_text(encoding="utf-8"))["test"])
    by_seed = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["id"] in test_ids:
            by_seed.setdefault(r.get("seed", 0), []).append(r)
    return {s: summarise(recs) for s, recs in sorted(by_seed.items())}


def _mean_sd(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    import statistics
    return (float(statistics.mean(vals)), float(statistics.stdev(vals)) if len(vals) > 1 else 0.0)


def seed_macros(R):
    """Mean and standard deviation over seeds of the headline test metrics (macros ...Mean / ...Sd)."""
    out = {"nseeds": len(R["seeds"])}
    for sysname in SEED_KEYS:
        if sysname not in R["systems"]:
            continue
        ps = _per_seed(R, sysname)
        if not ps:
            continue
        for key, short, nd in SEED_METRICS:
            mean, sd = _mean_sd([v.get(key) for v in ps.values()])
            if mean is not None:
                out[f"{SHORT_KEY[sysname]}{short}Mean"] = f(mean, nd)
                out[f"{SHORT_KEY[sysname]}{short}Sd"] = f(sd, nd)
    for sysname in ("B1_vanilla", "TriSafe_devtuned"):
        o = R["systems"].get(sysname, {}).get("test", {}).get("overall", {})
        if o.get("auroc_answered") is not None:
            out[f"{SHORT_KEY[sysname]}AurocAns"] = f(o["auroc_answered"], 3)
    return out


def table_seeds(R, name):
    """Per-seed test metrics for the headline systems: how much of the picture is seed noise."""
    rows = []
    for sysname in ("B1_vanilla", "TriSafe_devtuned", "E1_verbalized", "E7_self_consistency"):
        if sysname not in R["systems"]:
            continue
        ps = _per_seed(R, sysname)
        if not ps:
            continue
        label = tex_escape(R["systems"][sysname]["label"])
        for s, v in ps.items():
            rows.append(f"{label if s == min(ps) else ''} & {s} & {f(v['answer_rate'],2)} & {f(v['acc_answerable'],3)} & "
                        f"{f(v['selective_accuracy'],3)} & {f(v['unanswerable_abstention'],2)} & {f(v['decision_accuracy'],3)} & "
                        f"{f(v['ece_draft'],3)} & {f(v['auroc_draft'],3)} \\\\")
        if len(ps) > 1:
            cells = []
            for key, nd in (("answer_rate", 2), ("acc_answerable", 3), ("selective_accuracy", 3),
                            ("unanswerable_abstention", 2), ("decision_accuracy", 3), ("ece_draft", 3), ("auroc_draft", 3)):
                mean, sd = _mean_sd([v.get(key) for v in ps.values()])
                cells.append(f"{f(mean, nd)}$\\pm${f(sd, nd)}")
            rows.append(r"\quad \emph{mean $\pm$ sd} & & " + " & ".join(cells) + r" \\")
        rows.append(r"\addlinespace")
    body = "\n    ".join(rows)
    return rf"""\begin{{table*}}[t]
  \caption{{Per-seed test results ({name} backend). Each seed is an independent generation pass (sampling seed of the generator); retrieval and the grounding checker are deterministic. The main tables pool all seeds and resample queries, so this table shows how much of the variation is seed noise.}}
  \label{{tab:seeds-{name}}}
  \footnotesize
  \setlength{{\tabcolsep}}{{4pt}}
  \begin{{tabular}}{{llccccccc}}
    \toprule
    System & Seed & Ans.\ rate & Acc.\ $\uparrow$ & Sel.\ acc.\ $\uparrow$ & Unans.\ abst.\ $\uparrow$ & Decision acc.\ $\uparrow$ & ECE $\downarrow$ & AUROC $\uparrow$ \\
    \midrule
    {body}
    \bottomrule
  \end{{tabular}}
\end{{table*}}
"""


def table_scaling(RN, RL):
    """1.5B vs 7B generator, same pipeline, same queries: mean over seeds (sd in small type)."""
    if RN is None and RL is None:
        return "% no neural results\n"
    systems = [("B1_vanilla", "B1 vanilla"), ("TriSafe_devtuned", "TriSafe-RAG (dev-tuned)"),
               ("E1_verbalized", "E1 verbalized confidence"), ("E3_platt", "E3 Platt gate"),
               ("E7_self_consistency", "E7 self-consistency gate")]
    metrics = [("answer_rate", "Ans.\\ rate", 2), ("acc_answerable", "Acc.", 3), ("selective_accuracy", "Sel.\\ acc.", 3),
               ("unanswerable_abstention", "Unans.\\ abst.", 2), ("decision_accuracy", "Decision acc.", 3),
               ("ece_draft", "ECE", 3), ("auroc_draft", "AUROC", 3)]
    rows = []
    for sysname, label in systems:
        first = True
        for gen, R in (("1.5B", RN), ("7B", RL)):
            if R is None or sysname not in R["systems"]:
                continue
            ps = _per_seed(R, sysname)
            cells = []
            for key, _, nd in metrics:
                if ps:
                    mean, sd = _mean_sd([v.get(key) for v in ps.values()])
                    cells.append(f"{f(mean, nd)}{{\\scriptsize$\\pm${f(sd, nd)}}}" if len(ps) > 1 else f(mean, nd))
                else:
                    cells.append(f(R["systems"][sysname]["test"]["overall"].get(key), nd))
            rows.append(f"{label if first else ''} & {gen} & " + " & ".join(cells) + r" \\")
            first = False
        rows.append(r"\addlinespace")
    lang_rows = []
    for gen, R in (("1.5B", RN), ("7B", RL)):
        if R is None:
            continue
        ma, da = R.get("model_abstention_rate", {}), R.get("draft_accuracy_test", {})
        lang_rows.append(f"Generator abstention rate & {gen} & " + " & ".join(f(ma.get(l), 2) for l in LANGS) + r" & & & \\")
        lang_rows.append(f"Draft accuracy (answerable) & {gen} & " + " & ".join(f(da.get(l), 2) for l in LANGS) + r" & & & \\")
    head = " & ".join(m[1] for m in metrics)
    body = "\n    ".join(rows)
    lbody = "\n    ".join(lang_rows)
    return rf"""\begin{{table*}}[t]
  \caption{{Generator scale: Qwen2.5-1.5B-Instruct (fp16) against Qwen2.5-7B-Instruct (Q4\_K\_M) with the same retriever, grounding checker, prompts, tuning protocol, and queries. Test split, mean over seeds with the standard deviation across seeds in small type. The lower block gives, per language, how often the generator itself abstained before any policy and how often its draft was correct on answerable queries.}}
  \label{{tab:scaling}}
  \footnotesize
  \setlength{{\tabcolsep}}{{3.5pt}}
  \begin{{tabular}}{{llccccccc}}
    \toprule
    System & Gen. & {head} \\
    \midrule
    {body}
    \midrule
     & & en & si & ta & mixed & & & \\
    \midrule
    {lbody}
    \bottomrule
  \end{{tabular}}
\end{{table*}}
"""


if __name__ == "__main__":
    main()
    write_qualitative()
