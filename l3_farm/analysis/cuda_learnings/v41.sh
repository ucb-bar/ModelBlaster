cd /tmp/krisd/cl/flows/F41/r00/variants; find . -maxdepth 3 | head -30; L=$(ls -d */run.log 2>/dev/null | head -1); ls */ ; tail -c 3000 $(ls -t $(find . -name "*.log" -maxdepth 2) | head -1)
