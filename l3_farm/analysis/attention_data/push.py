import sys
sys.path.insert(0, "/Users/krisd/Documents/ModelBlaster-clone/l3_farm")
import farm
job = sys.argv[1]
for loc, rem in zip(sys.argv[2::2], sys.argv[3::2]):
    rc, out, err = farm.put(job, loc, rem); print(loc, "->", rem, rc, err.decode()[-300:])
