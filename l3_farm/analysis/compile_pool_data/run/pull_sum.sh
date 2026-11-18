for f in /tmp/krisd/cpool/lt/*/summary.json; do python3 -c "import json,sys;print(json.dumps(json.load(open(sys.argv[1]))))" $f; done
