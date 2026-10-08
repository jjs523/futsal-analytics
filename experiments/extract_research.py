import json
import sys

src = sys.argv[1]
s = open(src, encoding="utf-8").read()
d, _ = json.JSONDecoder().raw_decode(s[s.find("{"):])
print(list(d.keys()), len(s))
if "result" in d:
    d = d["result"] if isinstance(d["result"], dict) else json.loads(d["result"])
plan = d.get("plan", "")
if not isinstance(plan, str):
    plan = json.dumps(plan, ensure_ascii=False, indent=1)
open("experiments/research_plan.md", "w", encoding="utf-8").write(plan)
json.dump(d.get("findings"), open("experiments/research_findings.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("plan chars", len(plan), "findings", len(d.get("findings") or []))
