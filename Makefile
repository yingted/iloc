data?=data
prior.py:stat.py $(data)
	./stat.py $(data)/* > prior.py
print_offset:print_offset.c
	gcc -o print_offset{,.c} -std=c99
