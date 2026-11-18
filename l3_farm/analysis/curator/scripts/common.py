import sqlite3, json, glob, os, re, collections
BASE='/Users/krisd/Documents/ModelBlaster-clone/l3_farm/analysis/curator'
DB=glob.glob(BASE+'/work/v*.db')[0]
def con():
    return sqlite3.connect(f'file:{DB}?immutable=1&mode=ro',uri=True)
def claims(c=None):
    c=c or con(); out=[]
    for part,id_,st,bck,d,pv in c.execute('select * from gk_claims'):
        d=json.loads(d); pv=json.loads(pv); d['_prov']=pv; d['_status']=st; out.append(d)
    return out
def pbase(pid):
    return (pid or '').split('::')[0]
