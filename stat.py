#!/usr/bin/python
from numpy import *
from sys import argv
from os.path import basename
from json import dumps
a={}
for x in argv[1:]:
	lines=[array(map(float,line.split()))for line in open(x)]
	x=basename(x)
	deltas=[line[1]-line[0]for line in lines if len(line)==2]
	a[x]={
		"n":list(bincount(map(len,lines))/float(len(lines))),
		"mu"if deltas else Exception:mean(deltas),
		"sigma"if deltas else Exception:std(deltas),
		"p":1./len(argv[1:]),#assume uniform?
	}
	assert a[x].get("sigma",1)!=0,"zero sigma"
print"""\
#!/usr/bin/python
# This file is autogenerated
prior=%s"""%dumps(a,separators=",:",skipkeys=True)
