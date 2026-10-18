#!/usr/bin/env python3
"""Offline analysis of the complete hom-space structure census (#1365).

    complete_structure_census.py STEPS.csv[.gz] TRACE.txt[.gz]

STEPS.csv holds the `STEP` rows of `symtn-network/examples/complete_structure_census.rs`;
TRACE.txt the `M`/`L`/`A` lines of `complete_structure_census_lldb.py` from the
same run. The script

1. checks that every step's traced lookups equal its observed hits + misses;
2. replays the trace through an exact model of the current cache (FIFO, cap 5,
   byte budget, max entry bytes) and checks every step's predicted hits,
   misses, evictions and bypasses against the observed counters;
3. replays it against FIFO and LRU at larger caps, and prints the tables;
4. replays it against the #2014 bounds (the degeneracy-structure cache: no
   entry cap, 64 MiB over 2 shards, so the largest entry is 32 MiB; the
   cache now uses one shard, `symtn-core/src/cache.rs`, which this model
   does not reflect) and
   exits non-zero if any warm miss is not
   compulsory or any E1/conj row evicts in a warm call. The FIFO model is
   exact here because no warm call evicts.
"""

import collections
import csv
import gzip
import statistics
import sys

# The bounds the trace was recorded under; the exact-model check uses these.
CAP, BUDGET, MAX_ENTRY = 5, 1_764_237, 1_650_641
# The #2014 bounds: byte budget only, 64 MiB over two shards.
NEW_CAP, NEW_BUDGET = 1 << 62, 64 * 1024 * 1024
NEW_MAX_ENTRY = NEW_BUDGET // 2
CAPS = (5, 8, 16, 32, 64)
UNBOUNDED = 1 << 62


def warm_from(workload):
    """First steady-state iteration: E1 rows are warm from their second call;
    sweeps and iTEBD first let the truncated bond spaces settle."""
    group = workload.split("/")[0]
    return {"e1": 2, "conj": 2, "mps": 3, "itebd": 11}[group]


class Cache:
    def __init__(self, policy, cap, budget=UNBOUNDED, max_entry=UNBOUNDED):
        self.policy, self.cap, self.budget, self.max_entry = policy, cap, budget, max_entry
        self.entries = collections.OrderedDict()  # key -> bytes, oldest first
        self.charged = 0

    def lookup(self, key, size):
        """Returns (hit, evictions, bypassed)."""
        if key in self.entries:
            if self.policy == "lru":
                self.entries.move_to_end(key)
            return True, 0, False
        if size > self.max_entry or size > self.budget:
            return False, 0, True
        evicted = 0
        while self.entries and (
            len(self.entries) >= self.cap or self.charged + size > self.budget
        ):
            _, gone = self.entries.popitem(last=False)
            self.charged -= gone
            evicted += 1
        self.entries[key] = size
        self.charged += size
        return False, evicted, False


def text(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def load(steps_path, trace_path):
    steps = [
        row
        for row in csv.reader(text(steps_path))
        if row and row[0] == "STEP" and row[1] != "workload"
    ]
    # Keys are named by HomSpace content only, so sizes are scoped per
    # workload: each workload resets the cache and uses one rule.
    segments, admitted = [[]], [[]]
    for line in text(trace_path):
        kind, *rest = line.split()
        if kind == "M":
            segments.append([])
            admitted.append([])
        elif kind == "L":
            segments[-1].append(rest[0])
        elif kind == "A":
            admitted[-1].append((rest[0], int(rest[1])))
    if len(segments) - 1 != len(steps):
        raise SystemExit(f"{len(segments) - 1} marks for {len(steps)} steps")
    workloads, sizes, varying = collections.OrderedDict(), {}, 0
    for row, lookups, admits in zip(steps, segments, admitted):
        observed = tuple(int(v) for v in row[4:9])  # hits misses admissions evictions bypasses
        if len(lookups) != observed[0] + observed[1]:
            raise SystemExit(f"step {row}: {len(lookups)} traced lookups")
        name = row[1]
        for key, size in admits:
            previous = sizes.setdefault((name, key), size)
            varying += previous != size
        lookups = [(name, key) for key in lookups]
        workloads.setdefault(name, []).append((int(row[2]), row[3], lookups, observed))
    print(f"re-admissions charged differently from the first: {varying}")
    return workloads, sizes


def validate(workloads, sizes):
    """Exact current-cache model against the observed counters, per step."""
    mismatches = 0
    for name, steps in workloads.items():
        cache = Cache("fifo", CAP, BUDGET, MAX_ENTRY)
        for _, step, lookups, (hits, misses, _, evictions, bypasses) in steps:
            got = [0, 0, 0, 0]
            for key in lookups:
                hit, evicted, bypassed = cache.lookup(key, sizes[key])
                got[0 if hit else 1] += 1
                got[2] += evicted
                got[3] += bypassed
            if got != [hits, misses, evictions, bypasses]:
                mismatches += 1
                if mismatches <= 5:
                    print(f"model mismatch {name} {step}: {got} vs {[hits, misses, evictions, bypasses]}")
    return mismatches


def warm_misses(steps, sizes, policy, cap, budget=UNBOUNDED, warm=2):
    cache, misses, lookups = Cache(policy, cap, budget), 0, 0
    for iteration, _, keys, _ in steps:
        for key in keys:
            hit, _, _ = cache.lookup(key, sizes[key])
            if iteration >= warm:
                lookups += 1
                misses += not hit
    return misses, lookups


def min_cap(steps, sizes, policy, warm):
    """Smallest cap whose warm misses are only compulsory ones (keys first
    seen in the warm phase, which no cache can hit)."""
    floor = compulsory(steps, warm)
    for cap in range(1, 4097):
        if warm_misses(steps, sizes, policy, cap, warm=warm)[0] == floor:
            return cap
    return None


def compulsory(steps, warm):
    seen, count = set(), 0
    for iteration, _, keys, _ in steps:
        for key in keys:
            count += iteration >= warm and key not in seen
            seen.add(key)
    return count


def census(name, steps, sizes):
    warm = warm_from(name)
    per_iter = collections.defaultdict(set)
    per_call, lookups_per_call = [], []
    for iteration, _, keys, _ in steps:
        if iteration >= warm:
            per_iter[iteration].update(keys)
            per_call.append(len(set(keys)))
            lookups_per_call.append(len(keys))
    live = max(per_iter.values(), key=lambda s: sum(sizes[k] for k in s))
    distinct = {k for _, _, keys, _ in steps for k in keys}
    return {
        "warm": warm,
        "calls": len(per_call),
        "lookups_max": max(lookups_per_call),
        "distinct_call_max": max(per_call),
        "live_set": max(len(s) for s in per_iter.values()),
        "live_bytes": sum(sizes[k] for k in live),
        "distinct": len(distinct),
        "entry_bytes": sorted(sizes[k] for k in distinct),
        "observed_warm_misses": sum(o[1] for i, _, _, o in steps if i >= warm),
        "observed_evictions": sum(o[3] for _, _, _, o in steps),
        "warm_lookups": sum(lookups_per_call),
        "compulsory": compulsory(steps, warm),
        "min_cap_fifo": min_cap(steps, sizes, "fifo", warm),
        "min_cap_lru": min_cap(steps, sizes, "lru", warm),
        "rates": {
            (policy, cap): warm_misses(steps, sizes, policy, cap, warm=warm)
            for policy in ("fifo", "lru")
            for cap in CAPS
        },
        "rates_budget": {
            (policy, cap): warm_misses(steps, sizes, policy, cap, BUDGET, warm)
            for policy in ("fifo", "lru")
            for cap in CAPS
        },
    }


def new_bounds_gate(workloads, sizes):
    """Replays every workload under the new bounds. Returns the violations:
    warm misses beyond the compulsory ones, and warm evictions of E1/conj rows."""
    print(f"\n## Replay under the new bounds (FIFO, cap {NEW_CAP}, budget {NEW_BUDGET}, max entry {NEW_MAX_ENTRY})")
    print("| workload | warm lookups | warm misses | compulsory | warm evictions | evictions | bypasses | peak entries | peak bytes |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    violations = 0
    rows = collections.Counter()
    for name, steps in workloads.items():
        warm = warm_from(name)
        cache = Cache("fifo", NEW_CAP, NEW_BUDGET, NEW_MAX_ENTRY)
        lookups = misses = warm_evictions = evictions = bypasses = peak = peak_bytes = 0
        for iteration, _, keys, _ in steps:
            for key in keys:
                hit, evicted, bypassed = cache.lookup(key, sizes[key])
                evictions += evicted
                bypasses += bypassed
                peak = max(peak, len(cache.entries))
                peak_bytes = max(peak_bytes, cache.charged)
                if iteration >= warm:
                    lookups += 1
                    misses += not hit
                    warm_evictions += evicted
        floor = compulsory(steps, warm)
        bad = misses - floor + (warm_evictions if name.split("/")[0] in ("e1", "conj") else 0)
        violations += bad
        group = name.split("/")[0]
        if group in ("mps", "itebd"):
            print(
                f"| {name} | {lookups} | {misses} | {floor} | {warm_evictions} | {evictions} "
                f"| {bypasses} | {peak} | {peak_bytes} |"
            )
        else:
            rows.update(
                {(group, "rows"): 1, (group, "warm misses"): misses, (group, "compulsory"): floor,
                 (group, "warm evictions"): warm_evictions, (group, "evictions"): evictions,
                 (group, "bypasses"): bypasses}
            )
            rows[(group, "peak entries")] = max(rows[(group, "peak entries")], peak)
            rows[(group, "peak bytes")] = max(rows[(group, "peak bytes")], peak_bytes)
    for group in ("e1", "conj"):
        print(
            f"| {group} ({rows[(group, 'rows')]} rows, summed) | — | {rows[(group, 'warm misses')]} "
            f"| {rows[(group, 'compulsory')]} | {rows[(group, 'warm evictions')]} | {rows[(group, 'evictions')]} "
            f"| {rows[(group, 'bypasses')]} | {rows[(group, 'peak entries')]} (max) | {rows[(group, 'peak bytes')]} (max) |"
        )
    print(f"\nnew-bounds gate violations: {violations}")
    return violations


def quantiles(values):
    values = sorted(values)
    p90 = values[min(len(values) - 1, int(0.9 * len(values)))]
    return values[0], statistics.median(values), p90, values[-1]


def main():
    workloads, sizes = load(sys.argv[1], sys.argv[2])
    total = sum(len(s) for s in workloads.values())
    print(f"steps {total}, workloads {len(workloads)}, distinct keys {len(sizes)}")
    print(f"exact current-cache model mismatches: {validate(workloads, sizes)} of {total} steps\n")
    results = {name: census(name, steps, sizes) for name, steps in workloads.items()}

    print("## E1 rows: max over 5 cases x {f64, c64}, per op and symmetry")
    print("| op | symmetry | lookups/call | distinct/call | live set | observed warm misses @5 | min cap FIFO / LRU |")
    print("|---|---|---:|---:|---:|---:|---:|")
    table = collections.OrderedDict()
    for name, r in results.items():
        parts = name.split("/")
        if parts[0] in ("e1", "conj"):
            group, symmetry, _, _, op = parts
            table.setdefault((group, op, symmetry), []).append(r)
    for (group, op, symmetry), rs in table.items():
        if group != "e1":
            continue
        print(
            f"| {op} | {symmetry} | {max(r['lookups_max'] for r in rs)} | {max(r['distinct_call_max'] for r in rs)} "
            f"| {max(r['live_set'] for r in rs)} | {sum(r['observed_warm_misses'] for r in rs)} "
            f"| {max(r['min_cap_fifo'] for r in rs)} / {max(r['min_cap_lru'] for r in rs)} |"
        )

    print("\n## #1368 conjugated sources (f64, max over 5 cases)")
    print("| variant | symmetry | lookups/call | distinct/call | live set | observed warm misses @5 | min cap FIFO / LRU |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for (group, op, symmetry), rs in table.items():
        if group == "conj":
            print(
                f"| {op} | {symmetry} | {max(r['lookups_max'] for r in rs)} | {max(r['distinct_call_max'] for r in rs)} "
                f"| {max(r['live_set'] for r in rs)} | {sum(r['observed_warm_misses'] for r in rs)} "
                f"| {max(r['min_cap_fifo'] for r in rs)} / {max(r['min_cap_lru'] for r in rs)} |"
            )

    print("\n## Sweeps and networks: warm miss rate by cap (entry cap only, bytes unbounded)")
    header = " | ".join(f"{p.upper()} {c}" for p in ("fifo", "lru") for c in CAPS)
    print(f"| workload | warm calls | warm lookups | compulsory | distinct/call | live set | min cap FIFO / LRU | {header} |")
    print("|---|---:|---:|---:|---:|---:|---:|" + "---:|" * (2 * len(CAPS)))
    for name, r in results.items():
        if name.split("/")[0] in ("mps", "itebd"):
            rates = " | ".join(
                f"{m / n:.1%}" for m, n in (r["rates"][(p, c)] for p in ("fifo", "lru") for c in CAPS)
            )
            print(
                f"| {name} | {r['calls']} | {r['warm_lookups']} | {r['compulsory']} | {r['distinct_call_max']} | {r['live_set']} "
                f"| {r['min_cap_fifo']} / {r['min_cap_lru']} | {rates} |"
            )
    print("\nSame, with the current byte budget (1_764_237) also applied:")
    for name, r in results.items():
        if name.split("/")[0] in ("mps", "itebd"):
            rates = " | ".join(
                f"{m / n:.1%}" for m, n in (r["rates_budget"][(p, c)] for p in ("fifo", "lru") for c in CAPS)
            )
            print(f"| {name} | {rates} |")
    print("\nObserved at the current cap (actual FIFO cache):")
    for name, r in results.items():
        if name.split("/")[0] in ("mps", "itebd"):
            print(
                f"  {name}: warm misses {r['observed_warm_misses']} / {r['warm_lookups']} lookups, "
                f"evictions {r['observed_evictions']}"
            )

    print("\n## Entry bytes per workload group (distinct keys) and live-set bytes")
    print("| group | distinct keys | min | median | p90 | max | max live set | max live-set bytes |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    groups = collections.OrderedDict()
    for name, r in results.items():
        parts = name.split("/")
        group = "/".join(parts[:3]) if parts[0] == "mps" else f"{parts[0]}/{parts[1]}"
        groups.setdefault(group, []).append(r)
    for group, rs in groups.items():
        entry = [b for r in rs for b in r["entry_bytes"]]
        low, mid, p90, high = quantiles(entry)
        print(
            f"| {group} | {len(entry)} | {low} | {mid:.0f} | {p90} | {high} "
            f"| {max(r['live_set'] for r in rs)} | {max(r['live_bytes'] for r in rs)} |"
        )

    if new_bounds_gate(workloads, sizes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
