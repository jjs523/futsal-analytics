"""Copy audit sheets of several variants to anonymous names so raters cannot tell which tracker made them.

python experiments/blind_audit.py baseline final_a final_b final_d
Writes experiments/audit/_blind/<code>.jpg and experiments/audit/_blind/key.json (code -> variant, kind, original path, seconds),
and prints the sheet list as JSON for the visual-audit workflow.
"""
import hashlib
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import ROOT

if __name__ == "__main__":
    out = os.path.join(ROOT, "experiments", "audit", "_blind")
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)
    items = []
    for v in sys.argv[1:]:
        base = os.path.join(ROOT, "experiments", "audit", v)
        idx = json.load(open(os.path.join(base, "index.json")))
        for t in idx:
            items.append({"variant": v, "kind": "track", "src": t["sheet"], "seconds": t["seconds"]})
        ev = json.load(open(os.path.join(base, "events", "events.json")))
        for e in ev["sampled"]:
            items.append({"variant": v, "kind": "event", "src": e["sheet"], "seconds": None})
    key, sheets = {}, []
    for it in sorted(items, key=lambda it: hashlib.sha1(it["src"].encode()).hexdigest()):
        code = hashlib.sha1(("futsal-blind:" + it["src"]).encode()).hexdigest()[:10]
        dst = os.path.join(out, f"{it['kind']}_{code}.jpg")
        shutil.copyfile(it["src"], dst)
        key[code] = it
        sheets.append({"variant": code, "kind": it["kind"], "path": dst, "seconds": it["seconds"]})
    json.dump(key, open(os.path.join(out, "key.json"), "w"), indent=1)
    json.dump({"sheets": sheets}, open(os.path.join(out, "sheets.json"), "w"), indent=1)
    print(f"{len(sheets)} sheets -> {out}")
