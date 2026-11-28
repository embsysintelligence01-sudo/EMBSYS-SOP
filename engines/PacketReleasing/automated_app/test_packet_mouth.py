"""Standing packet mouth test: existing YOLO segmentation + OpenCV heuristics.
Run from project root: python test_packet_mouth.py
Hold mouth toward fixed overhead camera. No new model needed.
"""
import argparse
from pathlib import Path
import time
from collections import deque
import cv2
import numpy as np
from ultralytics import YOLO


def measure(frame, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    w, h = rect[1]
    ratio = min(w,h)/max(w,h,1)
    # Search for a dark enclosed region inside the packet silhouette.
    # This is a cavity candidate, not proof of a physical opening.
    interior = cv2.erode(mask, np.ones((9,9),np.uint8))
    gray = cv2.GaussianBlur(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(5,5),0)
    pixels = gray[interior > 0]
    cavity = np.zeros_like(mask)
    if pixels.size:
        threshold = max(0, float(np.median(pixels))-30)
        dark = ((gray < threshold) & (interior > 0)).astype(np.uint8)*255
        dark = cv2.morphologyEx(dark,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(dark)
        if n > 1:
            idx = 1 + np.argmax(stats[1:,cv2.CC_STAT_AREA])
            cavity[labels == idx] = 255
    fraction = cv2.countNonZero(cavity)/max(cv2.countNonZero(mask),1)
    return ratio, fraction, cavity, rect


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=str(Path(__file__).resolve().parent/'models'/'packet-seg.onnx'))
    p.add_argument('--camera',type=int,default=0)
    p.add_argument('--conf',type=float,default=.5)
    p.add_argument('--imgsz',type=int,default=320)
    p.add_argument('--seconds',type=float,default=2)
    a=p.parse_args()
    if not Path(a.model).is_file(): p.error(f'Model not found: {a.model}. Use --model PATH')
    print(f'Loading model: {a.model}', flush=True)
    model=YOLO(a.model,task='segment')
    print(f'Opening camera {a.camera}...', flush=True)
    cap=cv2.VideoCapture(a.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap=cv2.VideoCapture(a.camera)
    if not cap.isOpened(): raise RuntimeError('Cannot open camera; check --camera or close other camera apps.')
    print('Camera opened. Opening window (press V to verify, Q/ESC to quit).', flush=True)
    win='Packet mouth test'; cv2.namedWindow(win,cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Open ratio x100',win,50,100,lambda x:None)
    cv2.createTrackbar('Closed ratio x100',win,20,100,lambda x:None)
    cv2.createTrackbar('Cavity x1000',win,30,300,lambda x:None)
    history=deque(); enabled=False; last_settings=None; last_sample=None
    try:
        while True:
            ok,frame=cap.read()
            if not ok: print('Camera read failed'); break
            view=frame.copy(); H,W=frame.shape[:2]
            result=model.predict(frame,imgsz=a.imgsz,conf=a.conf,device='cpu',verbose=False,retina_masks=True)[0]
            op=cv2.getTrackbarPos('Open ratio x100',win)/100
            cl=cv2.getTrackbarPos('Closed ratio x100',win)/100
            cav=max(1,cv2.getTrackbarPos('Cavity x1000',win))/1000
            settings=(op,cl,cav)
            if settings!=last_settings: history.clear(); last_settings=settings
            raw='NO PACKET'; info=''; valid=False; last_sample=None
            if result.masks is not None and len(result.masks.data):
                candidates=[]
                for m in result.masks.data.cpu().numpy():
                    mask=cv2.resize(m,(W,H),interpolation=cv2.INTER_NEAREST)
                    mask=(mask>.5).astype(np.uint8)*255
                    if cv2.countNonZero(mask)>max(500,.003*H*W): candidates.append(mask)
                if len(candidates)>1: raw='MULTIPLE PIECES - isolate mouth'
                elif len(candidates)==1:
                    mask=candidates[0]
                    ratio,fraction,cavity,rect=measure(frame,mask)
                    last_sample=(ratio,fraction)
                    blue=np.array([255,150,0]); green=np.array([0,255,0])
                    view[mask>0]=(view[mask>0]*.7+blue*.3).astype(np.uint8)
                    view[cavity>0]=(view[cavity>0]*.4+green*.6).astype(np.uint8)
                    box=cv2.boxPoints(rect).astype(np.int32); cv2.polylines(view,[box],True,(255,150,0),2)
                    info=f'Aspect: {ratio:.3f}  Dark region: {fraction:.3f}'
                    ys,xs=np.where(mask>0)
                    if xs.min()<5 or ys.min()<5 or xs.max()>=W-5 or ys.max()>=H-5: raw='REPOSITION - clipped packet'
                    elif not enabled: raw='ALIGN MOUTH - press V to verify'
                    elif cl>=op: raw='INVALID THRESHOLDS'
                    else:
                        valid=True
                        raw='OPEN' if ratio>=op and fraction>=cav else 'CLOSED' if ratio<=cl and fraction<cav else 'UNCERTAIN'
            now=time.monotonic()
            if not valid or raw=='UNCERTAIN': history.clear()
            else:
                if history and (history[-1][1]!=raw or now-history[-1][0]>.75): history.clear()
                history.append((now,raw))
                while len(history)>1 and now-history[1][0]>=a.seconds: history.popleft()
            stable=valid and len(history)>=5 and now-history[0][0]>=a.seconds
            status=raw+' (heuristic)' if stable else ('CHECKING '+raw if valid and raw!='UNCERTAIN' else raw)
            color=(0,220,0) if stable and raw=='OPEN' else (0,0,255) if stable and raw=='CLOSED' else (0,220,255)
            cv2.rectangle(view,(0,0),(W,92),(25,25,25),-1)
            for i,(s,col) in enumerate([(status,color),(info,(255,255,255)),('V: verify/pause | O/C: log sample | Q: quit',(255,255,255))]):
                cv2.putText(view,s,(8,23+i*27),cv2.FONT_HERSHEY_SIMPLEX,.55,col,1,cv2.LINE_AA)
            cv2.imshow(win,view)
            key=cv2.waitKey(1)&255
            if key in (ord('q'),27): break
            if key==ord('v'): enabled=not enabled; history.clear()
            if key in (ord('o'),ord('c')) and last_sample is not None:
                import csv
                out=Path('mouth_samples.csv'); new=not out.exists()
                with out.open('a',newline='') as f:
                    writer=csv.writer(f)
                    if new: writer.writerow(['actual','aspect_ratio','dark_region_ratio'])
                    writer.writerow(['OPEN' if key==ord('o') else 'CLOSED',*last_sample])
                print('Sample saved:',last_sample)
            if cv2.getWindowProperty(win,cv2.WND_PROP_VISIBLE)<1: break
    finally:
        cap.release(); cv2.destroyAllWindows()

if __name__=='__main__': main()
