"""Per-workload summary of a symtn-gemm-trace/1 corpus (Markdown table).

usage: python3 summarize.py CORPUS.jsonl.gz
Counts weight every record by its multiplicity.
"""
import gzip, json, sys
from collections import defaultdict
from statistics import median


def wmedian(values, weights):
    pairs = sorted(zip(values, weights))
    half, acc = sum(weights) / 2, 0
    for v, w in pairs:
        acc += w
        if acc >= half:
            return v


groups = defaultdict(list)
with gzip.open(sys.argv[1], "rt") as f:
    for line in f:
        r = json.loads(line)
        if r["kind"] != "call":
            continue
        family = r["step"].split(".")[0]
        key = (r["model"], r["chi_states"], r["chi_multiplets"], r["scalar"], family)
        groups[key].append(r)

print("| model | chi states (multiplets) | scalar | steps | calls | distinct | jobs/call min/med/max | m | n | k | mkn min/med/max | work-wtd med mkn | largest-job share med | distinct geometries/call med |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for (model, states, mult, scalar, family), recs in sorted(groups.items()):
    calls = sum(r["multiplicity"] for r in recs)
    jobs_per_call, shares, geos, mkns, works = [], [], [], [], []
    ms, ns, ks = [], [], []
    for r in recs:
        mult_r = r["multiplicity"]
        w = [j[3] * j[4] * j[5] for j in r["jobs"]]
        jobs_per_call += [len(r["jobs"])] * mult_r
        shares += [max(w) / sum(w) if sum(w) else 0.0] * mult_r
        geos += [len({tuple(j[3:6]) for j in r["jobs"]})] * mult_r
        for j, x in zip(r["jobs"], w):
            ms.append(j[3]); ns.append(j[4]); ks.append(j[5])
            mkns += [x] * mult_r
    chi = f"{states}" if model.startswith("u1") else f"{states} ({mult})"
    print(
        f"| {model} | {chi} | {scalar} | {family} | {calls} | {len(recs)} "
        f"| {min(jobs_per_call)}/{median(jobs_per_call):g}/{max(jobs_per_call)} "
        f"| {min(ms)}-{max(ms)} | {min(ns)}-{max(ns)} | {min(ks)}-{max(ks)} "
        f"| {min(mkns)}/{median(mkns):g}/{max(mkns)} | {wmedian(mkns, mkns)} "
        f"| {median(shares):.2f} | {median(geos):g} |"
    )
