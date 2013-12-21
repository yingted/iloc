#!/usr/bin/python
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
def P(s,ratios,certainty=.8):
	return certainty*_P(s,ratios)+(1-certainty)*prior[s]["p"]
cascade="/usr/share/OpenCV/haarcascades/haarcascade_eye_tree_eyeglasses.xml"
def iterratios():
	cap=VideoCapture(0)
	cc=CascadeClassifier(cascade)
	bg_cc=CascadeClassifier(cascade)
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
			for rect in bg_cc.detectMultiScale(frame,1.1,1,0,(40,40),(70,70)):
				rescan_out.put((t,tuple(rect)))
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
				nearby=[(X-padding+x,Y-padding+y,w,h)for x,y,w,h in cc.detectMultiScale(frame[pymax(0,Y-padding):Y+H+padding,pymax(0,X-padding):X+W+padding],1.1,1,0,(int(.5+W/zr),int(.5+H/zr)),(int(.5+W*zr),int(.5+H*zr)))]
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

			#imshow("",frame)
			#while True:
			#	k=waitKey(1)
			#	if k in(-1,27):
			#		break
			#if k==27:
			#	break
	finally:
		rescan_in_replace(None)
def iloc(frame,rects):
	axis_max=1.3
	axis_factor=axis_max+1./axis_max
	area_fudge=5
	mid_hi_ratio=.34#.5

	rects.sort()
	ratios=[]
	for i,(x,y,w,h)in enumerate(rects):
		y+=h/5
		h=h*3/5
		eye=frame[y:y+h,x:x+w]
		if not eye.size:
			continue
		GaussianBlur(eye,(0,0),1,eye)
		yrb=cvtColor(eye,COLOR_BGR2YCR_CB)
		eye=yrb[:,:,0]

		eye2=eye.astype("int32")-GaussianBlur(eye,(0,0),(w*h)**.5*.5)
		eye2-=eye2.min()
		eye2=(eye2*255/eye2.max()).astype("uint8")
		eye=eye2

		rectangle(frame,(x,y),(x+w,y+h),(0,255,0))
		lo=eye.min()/255.
		hi=eye.max()/255.
		k=1./(hi-lo)
		lut=(array([3*((x-lo)*k)**2-2*((x-lo)*k)**3 if lo<x<hi else cmp(x,lo)for x in linspace(0,1,num=256)])*256).clip(0,255).astype("uint8")
		eye=LUT(eye,lut)

		cum=concatenate((zeros((h,1)),eye.cumsum(1)),1)
		def subrows(bounds):
			xl,yl,xh,yh=bounds
			for yi in xrange(yl,yh):
				dx=(((xh-xl)*(yh-yl))**2-((2*yi+1-yl-yh)*(xh-xl))**2)**.5/(yh-yl)
				pos=yi,pymax(0,int((xh+xl+1-dx)*.5)),pymin(w,int((xh+xl+1+dx)*.5)+1)
				if pos[1]<pos[2]:
					yield pos
		def integrate(bounds):
			xl,yl,xh,yh=bounds
			if xl<0 or yl<0 or xh>w or yh>h or(xh-xl)**2+(yh-yl)**2+area_fudge>((xh-xl)*(yh-yl)+area_fudge)*axis_factor:
				return float("nan")
			s=0.
			for yi,xs,xe in subrows(bounds):
				s+=cum[yi][xe]-cum[yi][xs]
			area=(xh-xl)*(yh-yl)
			return s/area if area else float("nan")

		ratio_hi=5.5
		ratio_mid=3
		bounds=array((w/2,h/2,w/2+1,h/2+1))
		count_hi=count_mid=0
		dirs=diag((-1,-1,1,1))
		for k in xrange(w*h):
			a=[integrate(bounds+delta)for delta in dirs]
			j=nanargmin(a)
			if isnan(j)or(k>3 and a[j]>ratio_hi):
				break
			bounds+=dirs[j]
			count_mid+=a[j]<=ratio_mid
			count_hi+=1

		if count_mid<=mid_hi_ratio*count_hi:
			continue

		eye=eye2

		xl,yl,xh,yh=bounds
		k=.8
		r=.5*.4
		#pos=lambda*p:tuple(int(round(v))for v in p)
		#ellipse(eye,pos((xh+xl)*.5-(xh-xl)*k,(yh+yl)*.5),pos((xh-xl)*r,(yh-yl)*r),0,0,360,(255,))
		#ellipse(eye,pos((xh+xl)*.5+(xh-xl)*k,(yh+yl)*.5),pos((xh-xl)*r,(yh-yl)*r),0,0,360,(255,))
		#def shift(bounds,k,r=2*r):
		#	bounds=bounds.reshape((2,2))
		#	ctr=bounds.mean(0)
		#	return(((bounds-ctr)*r+ctr+[bounds[:,0].ptp()*k,0]).flatten()+.5).astype("int32")
		#for k in-k,k:
		#	xl,yl,xh,yh=nbounds=shift(bounds,k)
		#	ellipse(eye,((xh+xl)/2,(yh+yl)/2),((xh-xl)/2,(yh-yl)/2),0,0,360,(255,))
		#	#print"% 5.02f"%integrate(nbounds)
		#	print list(eye[yi,xs:xe].max()for yi,xs,xe in subrows(nbounds))
		#for sgn in-1,1:
		#	nbounds=map(tuple,shift(bounds,1.2*sgn,1.8).reshape((2,2)))
		#	rectangle(eye,nbounds[-1],nbounds[1],(255,))
		at=lambda(x,y):int(eye[int(y),int(x)])
		#_pts=[]
		dists=[]
		for xi,sgn in(xl,-1),(xh,1):
			opos=array([xi+sgn*1.5,bounds[1::2].mean()])
			pos=opos.copy()
			vel=array([sgn,0.])
			best=at(pos)
			seen=set()
			#__pts=[]
			for j in xrange(20):#should take 20
				#__pts.append(pos)
				npos=pos+vel
				if sum((npos-opos)**2)**.5>(w*h)**.5*.25:# or(j==1 and at(npos)<100):
					pos=opos
					break
				try:
					best=pymax(best,at(npos))
				except IndexError:
					pos=opos
					break
				if best-at(npos)>10:
					break
				pos=npos
				grad=zeros(2)
				for delta in concatenate((identity(2),-identity(2))):
					try:
						grad+=delta*at(pos+delta)
					except IndexError:
						continue
				vel+=grad*.01
			#if all(pos!=opos):
			#	_pts.extend(__pts)
			dists.append(sum((pos-opos)**2)**.5)
		#ellipse(eye,((xh+xl)/2,(yh+yl)/2),((xh-xl)/2,(yh-yl)/2),0,0,360,(255,))
		#for x,y in _pts:
		#	try:
		#		eye[y,x]=0
		#	except IndexError:
		#		pass
		#imshow("eye%d"%i,eye)
		if sum(dists):
			ratios.append((dists[0]+(xh-xl)*.5)/(sum(dists)+xh-xl))#accurate enough
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
					transition=decay*identity(3)+(1-decay)*ones((3,3))/3
					posterior=ones(3)/3
					def set_brightness(brightness):
						libXxf86vm.XF86VidModeSetGammaRamp(*([dpy,screen,ramp_size]+[ramp_t(*map(int,map(round,map(brightness.__mul__,ramp))))for ramp in ramps]))
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
					try:
						brightness_thd.start()
						for ratios in iterratios():
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
							target_brightness=pymin(1,posterior[prior.keys().index("c")]/.8)
					finally:
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
