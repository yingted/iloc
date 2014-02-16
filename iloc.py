#!/usr/bin/python -O
# vim: set ts=4 sw=4 noet:
pyany=any
pymin=min
pymax=max
from cv2 import *
from numpy import *
from itertools import cycle
from prior import prior
from collections import deque
from threading import Thread
from Queue import Queue,Empty
from time import time
from ctypes import *
from cPickle import dump,load,PickleError
from scipy.optimize import fmin
from numpy.lib import mgrid
libX11=CDLL("libX11.so")
libX11.XOpenDisplay.argtypes=c_char_p,
libX11.XOpenDisplay.restype=c_void_p
libX11.XCloseDisplay.argtypes=c_void_p,
libXxf86vm=CDLL("libXxf86vm.so")
libXxf86vm.XF86VidModeGetGammaRampSize.argtypes=c_void_p,c_int,POINTER(c_int)
libXxf86vm.XF86VidModeGetGammaRamp.argtypes=c_void_p,c_int,c_int,POINTER(c_ushort),POINTER(c_ushort),POINTER(c_ushort)
libXxf86vm.XF86VidModeSetGammaRamp.argtypes=c_void_p,c_int,c_int,POINTER(c_ushort),POINTER(c_ushort),POINTER(c_ushort)
def _P(s,ratios):
	if s in prior:
		o=prior[s]
		try:
			p=o["p"]*o["n"][len(ratios)]
		except IndexError:
			return 0
		if len(ratios)==2:
			sigma=o["sigma"]
			p*=exp(-.5*((ratios[1]-ratios[0]-o["mu"])/sigma)**2)/(sigma*(2*pi)**.5)
		return p
	return 0
def P(s,ratios,certainty=1.):
	return certainty*_P(s,ratios)+(1-certainty)*prior[s]["p"]
cascade="/usr/share/OpenCV/haarcascades/haarcascade_eye_tree_eyeglasses.xml"
cascade_face="/usr/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"
def iterratios():
	cap=VideoCapture(0)
	cc=CascadeClassifier(cascade)
	bg_cc=CascadeClassifier(cascade)
	if __debug__:
		face=CascadeClassifier(cascade_face)
	bg_face=CascadeClassifier(cascade_face)
	rects=[]
	padding=20
	zr=1.15
	t=-1
	rescan_in=Queue(1)
	rescan_out=Queue()
	def rescan():
		while True:
			job=rescan_in.get()
			if not job:
				break
			t,frame=job
			faces=bg_face.detectMultiScale(frame,1.3,1,0,(150,150),(350,350))
			ofs=array([0,0,0,0])
			if len(faces):
				x,y,w,h=faces[argmax(faces[:,2]*faces[:,3])]
				ofs[:2]=x,y+h/4
				frame=frame[ofs[1]:y+3*h/5,ofs[0]:x+w]
			rects=bg_cc.detectMultiScale(frame,1.1,1,0,(40,40),(70,70))
			for rect in rects:
				rescan_out.put((t,tuple(ofs+rect)))
	rescan_thd=Thread(target=rescan)
	rescan_thd.daemon=True
	rescan_thd.start()
	def rescan_in_replace(o):
		try:
			rescan_in.get_nowait()
		except Empty:
			pass
		rescan_in.put(o)
	try:
		while cap.grab():
			t+=1
			frame=cap.retrieve()[1]
			nrects=[]
			try:
				while True:
					rects.append(rescan_out.get_nowait())
			except Empty:
				pass
			for rect in rects:
				ti,(X,Y,W,H)=rect
				nearby=[(X-padding+x,Y-padding+y,w,h)for x,y,w,h in cc.detectMultiScale(frame[pymax(0,Y-padding):Y+H+padding,pymax(0,X-padding):X+W+padding],1.1,3,0,(int(.5+W/zr),int(.5+H/zr)),(int(.5+W*zr),int(.5+H*zr)))]
				if nearby:
					nrects.extend((t,rect)for rect in nearby)
				elif ti>t-10:
					nrects.append(rect)
			rects=[(ti,(X,Y,W,H))for I,(ti,(X,Y,W,H))in enumerate(nrects)if not pyany(I!=i and(
					(X<=x)+(x+w<=X+W)+(Y<=y)+(y+h<=Y+H)>2 or
					(I<i and pymax(0,pymin(X+W,x+w)-pymax(X,x))*pymax(0,pymin(Y+W,y+w)-pymax(Y,y))>.8*pymin(W,w)*pymin(H,h))
				)for i,(_,(x,y,w,h))in enumerate(nrects))]
			rescan_in_replace((t,frame))
			yield iloc(frame,[rect for ti,rect in rects if ti==t])

			if __debug__:
				for rect in face.detectMultiScale(frame,1.3,1,0,(150,150),(350,350)):
					x,y,w,h=rect
					rectangle(frame,(x,y),(x+w,y+h),(255,0,0))
				imshow("",frame)
				while True:
					k=waitKey(1)
					if k in(-1,27):
						break
				if k==27:
					break
	finally:
		rescan_in_replace(None)
def iloc(frame,rects):
	rects=[(x,y,w,h)for x,y,w,h in rects if(diff(clip((x,x+w,y,y+h),(0,0),frame.shape).reshape(2,2)).reshape(2)>0).all()]
	rects.sort(key=lambda x,y,w,h:h)
	rects[2:]=[]
	rects.sort()
	ratios=[]
	for i,(x,y,w,h)in enumerate(rects):
		x,x_w=clip((x-w/8,x+w+w/8),0,frame.shape[1])
		w=x_w-x
		y+=h/5
		h=h*3/5
		eye=frame[y:y+h,x:x+w]
		assert eye.size

		GaussianBlur(eye,(0,0),.8,eye)
		yrb=cvtColor(eye,COLOR_BGR2YCR_CB)
		eye=yrb[:,:,0]

		if __debug__:
			rectangle(frame,(x,y),(x+w,y+h),(0,255,0))

		# high pass
		eye2=eye.astype("int32")-GaussianBlur(eye,(0,0),(w*h)**.5*.2)
		eye2-=eye2.min()
		eye=(eye2*255/eye2.max()).astype("uint8")

		# remove AGC
		lo=eye.min()
		hi=eye.max()
		k=1./(hi-lo)
		lut=(array([3*((v-lo)*k)**2-2*((v-lo)*k)**3 if lo<v<hi else cmp(v,lo)for v in linspace(0,255,num=256)])*256).clip(0,255).astype("uint8")
		eye=LUT(eye,lut)
		#imshow("eye%d"%i,eye)

		# a b
		# c d
		eye=eye.astype("int")
		a=eye[:-1,:-1]
		b=eye[:-1,1:]
		c=eye[1:,:-1]
		d=eye[1:,1:]
		wij=255*4-(a+b+c+d) # times 4
		gi=c+d-a-b # times 4
		gj=b+d-a-c
		squared=1.7*gi**2+gj**2
		good=squared>percentile(squared,80)

		xi,xj=mgrid[map(slice,eye.shape)]
		gi_good=gi[good]
		gj_good=gj[good]
		xi_good=xi[good]
		xj_good=xj[good]

		def objective(p):
			ci,cj=clip(p,(0,0),(wij.shape[0]-1,wij.shape[1]-1))
			di=xi_good-ci
			dj=xj_good-cj
			abs_d=hypot(di,dj)
			i=pymin(wij.shape[0]-2,int(ci))
			j=pymin(wij.shape[1]-2,int(cj))
			with errstate(divide="ignore",invalid="ignore"):
				return-(
					+wij[i][j]*(1-(ci-i))*(1-(cj-j))
					+wij[i][j+1]*(1-(ci-i))*(cj-j)
					+wij[i+1][j]*(ci-i)*(1-(cj-j))
					+wij[i+1][j+1]*(ci-i)*(cj-j)
				)*(dot(di/abs_d,gi_good)+dot(dj/abs_d,gj_good))
		ci,cj=fmin(objective,(eye.shape[0]*.5,eye.shape[1]*.5),xtol=.01,disp=False)
		if __debug__:
			circle(frame,(int(round(x+.5+cj)),int(round(y+.5+ci))),1,(255,0,0),3)
			for _i,_row in enumerate(good):
				for _j,_v in enumerate(_row):
					if _v:
						frame[y+_i][x+_j]=(0,0,255)#actually (.5,.5) too low

		contours,_=findContours(good.astype("uint8"),RETR_EXTERNAL,CHAIN_APPROX_NONE)
		eps=int(round(hypot(w,h)*.1))
		perimeter=.25*hypot(w,h)
		lo=cj
		hi=lo+1
		for i,contour in enumerate(contours):
			if((eps<=contour[:,0,0])&(contour[:,0,0]<w-eps)&(eps<=contour[:,0,1])&(contour[:,0,1]<h-eps)).sum()<perimeter:
				continue
			lo=pymin(lo,pymax(eps,contour[:,0,0].min()))
			hi=pymax(hi,pymin(w-eps-1,contour[:,0,0].max()))
			if __debug__:
				contour[:,0]+=[x,y]
				drawContours(frame,contours,i,(0,255,255))

		ratios.append((.5+cj-lo)/(hi-lo))
	#print"\t".join(map("% 5.04f".__mod__,ratios)),
	#if len(ratios)>1:# and pymin(sorted(map(float.__sub__,ratios[1:],ratios[:-1])))<.11:
	#	print"*"*30,"% 5.04f"%pymin(sorted(map(float.__sub__,ratios[1:],ratios[:-1]))),
	#print"\t".join(map(str,ratios))
	return ratios
class X11Error(BaseException):
	pass
def x11_call(func,*args):
	ret=func(*args)
	if not ret:
		raise X11Error()
	return ret
if __name__=="__main__":
	from os import *
	from stat import *
	from fcntl import flock,LOCK_EX,LOCK_UN
	from signal import SIGTERM,signal
	from time import sleep
	disp=environ["DISPLAY"]
	if disp[0]!=":":#require local
		raise ValueError("invalid display",disp)
	dpy=x11_call(libX11.XOpenDisplay,disp)
	screen=cast(dpy+224,POINTER(c_int))[0]
	pidpath="/tmp/iloc.%d.pid"%int(float(disp[1:]))
	class SigTerm(BaseException):
		pass
	def handler(signum,frame):
		raise SigTerm()
	try:
		signal(SIGTERM,handler)
		while True:
			with fdopen(open(pidpath,O_RDWR|O_CREAT,S_IRUSR|S_IWUSR),"w+")as fd:
				try:
					flock(fd,LOCK_EX)
					try:
						data=load(fd)
					except(EOFError,PickleError):
						ramp_size=c_int()
						libXxf86vm.XF86VidModeGetGammaRampSize(dpy,screen,byref(ramp_size))
						ramp_t=c_ushort*ramp_size.value
						ramps=[ramp_t()for _ in"rgb"]
						libXxf86vm.XF86VidModeGetGammaRamp(*([dpy,screen,ramp_size]+ramps))
						data={
							'ramps':[map(int,ramp)for ramp in ramps],
						}
					else:
						ramp_t=c_ushort*len(data['ramps'][0])
						if data is None:
							continue
						try:
							kill(data['pid'],SIGTERM)
							break
						except OSError:
							pass
					fd.seek(0)
					fd.truncate()
					data['pid']=getpid()
					dump(data,fd)
					fd.flush()
					flock(fd,LOCK_UN)

					decay=.95#don't use time
					n=len(prior)
					posterior=[o['p']for o in prior.itervalues()]
					transition=decay*identity(n)+(1-decay)*repeat(posterior,n).reshape((n,n))
					def set_brightness(brightness):
						libXxf86vm.XF86VidModeSetGammaRamp(*([dpy,screen,ramp_size]+[ramp_t(*map(int,map(round,map((brightness**2*(3-2*brightness)).__mul__,ramp))))for ramp in ramps]))
					if __debug__:
						set_brightness=float
					cur_brightness=target_brightness=1
					def brightness_loop():
						global cur_brightness
						while True:
							tgt=target_brightness
							if tgt is None:
								break
							cur_brightness+=(tgt-cur_brightness)*.3
							set_brightness(cur_brightness)
							sleep(1/60.)
					brightness_thd=Thread(target=brightness_loop)
					brightness_thd.daemon=True
					#_len_0=_len_1=_len_2=0
					try:
						brightness_thd.start()
						for ratios in iterratios():
							#_len_0+=1
							#_len_1+=len(ratios)
							#_len_2+=len(ratios)**2
							posterior=dot(transition,posterior)
							posterior*=array([P(s,ratios)/o["p"]for s,o in prior.iteritems()])
							if not posterior.sum():
								posterior=ones(len(prior))
							posterior/=posterior.sum()
							#print dict(zip(prior,posterior))
							#print prior.items()[argmax(posterior)]
							#guess,prob=zip(*sorted(zip(posterior,prior),reverse=True))[::-1]
							#print" ".join(guess)," ".join(map("% 5.02f".__mod__,prob))
							#print{s:"%.02f"%(P(s,ratios)/o["p"])for s,o in prior.iteritems()},len(ratios)," ".join(map("%.04f".__mod__,ratios))
							target_brightness=pymin(1,posterior[prior.keys().index("c")])
					finally:
						#_mu=1.*_len_1/_len_0
						#_sigma=(1.*_len_2/_len_0-_mu**2)**.5
						#print "mu",_mu,"sigma",_sigma,"mu_sigma",_sigma/(_len_0-1)**.5
						target_brightness=None
						try:
							brightness_thd.join()
						except RuntimeError:
							pass
						set_brightness(1)
				finally:
					try:
						flock(fd,LOCK_EX)
						fd.seek(0)
						try:
							if load(fd)['pid']==getpid():
								unlink(pidpath)
								fd.seek(0)
								fd.truncate()
								dump(None,fd)
								fd.flush()
						except(EOFError,PickleError):
							pass
						except TypeError:
							pass
					except OSError:
						pass
			break
	except SigTerm:
		pass
	finally:
		libX11.XCloseDisplay(dpy)
