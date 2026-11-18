python3 -c "
import json
s=json.load(open(\"/tmp/krisd/cl/flows/F41/summary.json\"))
for k,v in s[\"irs\"].items(): print(k, {x:v.get(x) for x in (\"first_us\",\"final_us\",\"passed\",\"model_e2e_us\",\"model_basis\",\"seed_e2e_us\",\"model_passed\",\"cache_split\",\"model_error\")})
print(s.get(\"accepted\"), s.get(\"rejected\"), s.get(\"llm_calls\"))
print(json.dumps({k:v for k,v in (s.get(\"cuda_learnings\") or {}).items() if k!=\"reuse\"})[:800])
"
