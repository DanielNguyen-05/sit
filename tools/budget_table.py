"""Print every budget FID result of one or more runs as a table.
    python tools/budget_table.py                                   # every run in <repo>/results
    python tools/budget_table.py results/nested_uniform results/nested_time
Reads results/<exp>/fid_budget/*.json written by eval_fid.py --num-queries / --weak-queries."""
import glob, json, os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # tools/ -> repository root

rows = []
for run in sys.argv[1:] or [os.path.join(REPO, "results", "*")]:
    for f in sorted(glob.glob(os.path.join(run, "fid_budget", "*.json"))):
        r = json.load(open(f))
        rows.append((os.path.basename(os.path.dirname(os.path.dirname(f))), r.get("budget_mode") or "-",
                     r.get("train_steps") or 0, r.get("num_samples"), str(r.get("num_queries") or "max"),
                     str(r.get("weak_queries") or "-"), str(r.get("guidance_scale") or "-"),
                     r["fid"], r.get("minutes", 0)))
if not rows:
    sys.exit("no fid_budget/*.json found")
rows.sort(key=lambda r: (r[0], r[2], r[5], r[4]))
head = ("run", "trained", "steps", "N", "num_queries", "weak", "scale", "FID", "min")
w = [max(len(str(x)) for x in col) for col in zip(head, *[(a, b, c, d, e, f, g, f"{h:.2f}", i) for a, b, c, d, e, f, g, h, i in rows])]
fmt = "  ".join("{:<%d}" % n for n in w)
print(fmt.format(*head))
for a, b, c, d, e, f, g, h, i in rows:
    print(fmt.format(a, b, c, d, e, f, g, f"{h:.2f}", i))
