#include<X11/Xlib.h>
#include<stdio.h>
int main(){
	printf("%d\n",(size_t)&DefaultScreen(NULL));
}
