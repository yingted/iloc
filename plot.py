#!/usr/bin/python -Ou
from iloc import iterratios
from itertools import ifilter,islice
from matplotlib.pyplot import *
data=[]
for ratios in iterratios():
	print"\t".join(map("% 5.04f".__mod__,ratios))
	if len(ratios)==2:
		data.append(ratios)
		if len(data)==100:
			break
xlim((0,1))
ylim((0,1))
scatter(*zip(*data))
show()
