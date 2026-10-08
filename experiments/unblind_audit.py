"""Aggregate a blinded visual-audit workflow result per variant: python experiments/unblind_audit.py <workflow output file>

Track sheets -> length-weighted identity purity (1 - foreign / judged crops) and the share of sheets with >1 identity;
event sheets -> swap rate among decided crossings (SWAP / (OK + SWAP)). Writes experiments/audit/_blind/summary.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import ROOT

if __name__ == "__main__":
    s = open(sys.argv[1], encoding="utf-8").read()
    d, _ = json.JSONDecoder().raw_decode(s[s.find("{"):])
    r = d.get("result", d)
    r = r if isinstance(r, dict) else json.loads(r)
    key = json.load(open(os.path.join(ROOT, "experiments", "audit", "_blind", "key.json")))
    agg = {}
    for x in r["details"]:
        it = key[x["variant"]]
        v = agg.setdefault(it["variant"], {"track_sheets": 0, "multi_id": 0, "w": 0.0, "wp": 0.0, "crops": 0, "foreign": 0,
                                           "unclear": 0, "events": 0, "ok": 0, "swap": 0, "unsure": 0, "swap_sheets": []})
        if x["kind"] == "track":
            judged = max(1, x["n_crops"] - x["unclear_crops"])
            pur = 1 - len(x["foreign_crops"]) / judged
            v["track_sheets"] += 1; v["multi_id"] += int(x["n_identities"] > 1)
            v["w"] += it["seconds"]; v["wp"] += pur * it["seconds"]
            v["crops"] += x["n_crops"]; v["foreign"] += len(x["foreign_crops"]); v["unclear"] += x["unclear_crops"]
        else:
            v["events"] += 1
            v[{"OK": "ok", "SWAP": "swap", "UNSURE": "unsure"}[x["verdict"]]] += 1
            if x["verdict"] == "SWAP":
                v["swap_sheets"].append(os.path.basename(it["src"]))
    out = {}
    for name, v in sorted(agg.items()):
        out[name] = {"purity_weighted": round(v["wp"] / max(v["w"], 1e-9), 3),
                     "sheets_with_2plus_ids": f"{v['multi_id']}/{v['track_sheets']}",
                     "foreign_crops": f"{v['foreign']}/{v['crops'] - v['unclear']}",
                     "swap_rate": round(v["swap"] / max(v["ok"] + v["swap"], 1), 3),
                     "crossings": f"OK {v['ok']}, SWAP {v['swap']}, UNSURE {v['unsure']}",
                     "swap_sheets": v["swap_sheets"]}
    json.dump(out, open(os.path.join(ROOT, "experiments", "audit", "_blind", "summary.json"), "w"), indent=1)
    print(json.dumps(out, indent=1, ensure_ascii=False))
