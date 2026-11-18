import sys; sys.path.insert(0,'/tmp/krisd/curator')
import llm,os,subprocess
print(llm.spent(), round(llm.usd(),3))
print(subprocess.run('wc -l /tmp/krisd/curator/*.jsonl; ps aux | grep b_judge | head -3',shell=True,capture_output=True,text=True).stdout)
