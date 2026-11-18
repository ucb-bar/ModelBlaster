import sys; sys.path.insert(0,'/tmp/krisd/curator')
import llm
t,r=llm.call(llm.H,'You are terse.','Reply with JSON {"ok":true} only.',50,'ping'); print(t,r)
t,r=llm.call(llm.O,'You are terse.','Reply with JSON {"ok":true} only.',50,'ping'); print(t,r)
print(llm.spent())
