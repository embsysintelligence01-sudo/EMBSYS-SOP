"""EMBSYS seven-stage inspection prototype. Run beside models/; see --help.
Fixed overhead camera; start with a genuinely empty bed. No manual stage controls.
All visual rules are experimental; UNKNOWN never counts as a successful inspection.
"""
import argparse, json, time, re, traceback
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QPlainTextEdit

ROOT = Path(__file__).resolve().parent
FILES = {'packet':'packet-seg.onnx','number':'packet-no-obb.onnx','strip':'strip-seg.onnx','hand':'hand-seg.onnx','scissor':'scissor-seg.pt','cut':'cut-seg.pt','gold':'jeweltrack.onnx'}
STAGES = ['EMPTY TESTBED','FRONT / OCR','BACK / SEAL','CUTTING ACTIVITY','CUT SEPARATION','MOUTH INSPECTION','JEWELLERY COUNT','COMPLETE']

def stamp(): return datetime.now().astimezone().isoformat(timespec='seconds')
def union(ms, shape):
    out=np.zeros(shape,np.uint8)
    for m in ms: out |= m
    return out

def area(m): return int(cv2.countNonZero(m))
def overlap(a,b): return area(a & b)/max(1,area(a))
def near(a,b,px=15): return bool(area(cv2.dilate(a,np.ones((2*px+1,2*px+1),np.uint8)) & b))
def shape_ratio(m):
    pts=cv2.findNonZero(m)
    if pts is None: return 0
    w,h=cv2.minAreaRect(pts)[1]; return min(w,h)/max(w,h,1)

def mouth(frame,m):
    inner=cv2.erode(m,np.ones((9,9),np.uint8))
    g=cv2.GaussianBlur(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(5,5),0)
    vals=g[inner>0]; dark=np.zeros_like(m)
    if vals.size:
        dark=((g < max(0,float(np.median(vals))-30)) & (inner>0)).astype(np.uint8)*255
        dark=cv2.morphologyEx(dark,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
    n,_,stats,_=cv2.connectedComponentsWithStats(dark)
    cavity=max(stats[1:,cv2.CC_STAT_AREA],default=0)/max(1,area(m))
    return shape_ratio(m),float(cavity)

class Worker(QThread):
    image=pyqtSignal(QImage); status=pyqtSignal(str); event=pyqtSignal(str)
    def __init__(self,a):
        super().__init__(); self.a=a; self.stage=0; self.models={}; self.events=[]; self.violations=set(); self.packet_id=None; self.count=None; self.triggers={}; self.cut_ok=False; self.mouth_status='UNKNOWN'; self.started=stamp(); self.bg=None; self.ocr=None; self.last_ocr=0; self.front=None; self.before=None; self.seen_hand=False; self.tracks={}; self.next_id=1; self.ever_gold=False; self.empty_packet_seen=False; self.last_frame=None; self.evidence={}
        self.out=Path(a.output)/datetime.now().strftime('%Y%m%d_%H%M%S_%f'); self.out.mkdir(parents=True,exist_ok=True)
    def log(self,msg,kind='EVENT'):
        row={'time':stamp(),'stage':STAGES[self.stage],'kind':kind,'message':msg}; self.events.append(row); self.event.emit(f"{row['time'][11:19]} {kind}: {msg}")
        with (self.out/'events.jsonl').open('a',encoding='utf8') as f: f.write(json.dumps(row)+'\n')
    def warn(self,msg):
        if msg not in self.violations: self.violations.add(msg); self.log(msg,'WARNING')
    def stable(self,key,condition,seconds=1.0,frames=3):
        now=time.monotonic()
        if not condition: self.triggers.pop(key,None); return False
        start,n,last=self.triggers.get(key,(now,0,now))
        if now-last>self.a.max_frame_gap: start,n=now,0
        self.triggers[key]=(start,n+1,now)
        return now-start>=seconds and n+1>=frames
    def advance(self,msg):
        self.log(msg); self.stage+=1; self.triggers.clear()
        if self.last_frame is not None: cv2.imwrite(str(self.out/f'stage_{self.stage:02d}.jpg'),self.last_frame)
        self.log('Entered '+STAGES[self.stage])
    def model_path(self,k):
        p=Path(self.a.models)/FILES[k]
        if k=='gold' and self.a.jewel_model: p=Path(self.a.jewel_model)
        elif k=='gold' and not p.is_file() and (ROOT/FILES[k]).is_file(): p=ROOT/FILES[k]
        return p
    def predict(self,k,frame,conf=None):
        if k not in self.models: self.models[k]=YOLO(str(self.model_path(k)),task='obb' if k=='number' else 'segment')
        return self.models[k].predict(frame,imgsz=640 if k=='number' else 320,conf=conf or self.a.conf,device=self.a.device,verbose=False,retina_masks=True)[0]
    def masks(self,k,frame,conf=None):
        r=self.predict(k,frame,conf); out=[]
        if r.masks is not None:
            for m in r.masks.data.cpu().numpy():
                b=(cv2.resize(m,(frame.shape[1],frame.shape[0]),interpolation=cv2.INTER_NEAREST)>.5).astype(np.uint8)*255
                if area(b)>= (self.a.min_piece if k=='packet' else 30): out.append(b)
        return out
    def number(self,frame,p):
        r=self.predict('number',frame,.15); ob=r.obb
        if ob is None or not len(ob): return None
        order=np.argsort(-ob.conf.cpu().numpy()); corners=ob.xyxyxyxy.cpu().numpy()
        for i in order:
            pts=corners[i].astype(np.float32); region=np.zeros(p.shape,np.uint8); cv2.fillConvexPoly(region,pts.astype(np.int32),255)
            if overlap(region,p)<.65: continue
            w=int(np.linalg.norm(pts[1]-pts[0])); h=int(np.linalg.norm(pts[2]-pts[1]))
            if min(w,h)<5: continue
            dst=np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
            crop=cv2.warpPerspective(frame,cv2.getPerspectiveTransform(pts,dst),(w,h))
            if h>w: crop=cv2.rotate(crop,cv2.ROTATE_90_CLOCKWISE)
            return crop
        return None
    def read_id(self,crop):
        if self.ocr is None:
            from paddleocr import PaddleOCR
            self.ocr=PaddleOCR(lang='en',use_doc_orientation_classify=False,use_doc_unwarping=False,use_textline_orientation=False)
        for im in (crop,cv2.rotate(crop,cv2.ROTATE_180)):
            im=cv2.resize(im,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)
            for r in self.ocr.predict(im):
                for t,s in zip(r.get('rec_texts',[]),r.get('rec_scores',[])):
                    t=re.sub(r'\s+','',str(t))
                    if float(s)>=self.a.ocr_conf and re.fullmatch(self.a.id_regex,t): return t
        return None
    def foreground(self,frame):
        if self.bg is None: return np.zeros(frame.shape[:2],np.uint8)
        d=cv2.absdiff(cv2.GaussianBlur(frame,(7,7),0),self.bg).max(axis=2)
        return cv2.morphologyEx((d>self.a.bg_delta).astype(np.uint8)*255,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
    def paint(self,view,ms,color):
        for m in ms:
            view[m>0]=(view[m>0]*.7+np.array(color)*.3).astype(np.uint8)
            cs,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(view,cs,-1,color,2)
    def track(self,ms,view):
        # One-to-one mask matching; count current stable instances, not cumulative IDs.
        candidates=[]
        for i,m in enumerate(ms):
            for tid,(old,miss) in self.tracks.items():
                iou=area(m&old)/max(1,area(m|old))
                if iou>.25: candidates.append((iou,i,tid))
        assigned={}; used=set()
        for _,i,tid in sorted(candidates,reverse=True):
            if i not in assigned and tid not in used: assigned[i]=tid; used.add(tid)
        updated={tid:(m,miss+1) for tid,(m,miss) in self.tracks.items() if tid not in used and miss<5}
        for i,m in enumerate(ms):
            tid=assigned.get(i)
            if tid is None: tid=self.next_id; self.next_id+=1
            updated[tid]=(m,0); x,y,w,h=cv2.boundingRect(m); cv2.putText(view,f'J{tid}',(x,y+15),0,.6,(0,255,0),2)
        self.tracks=updated
        return tuple(sorted(t for t,(_,miss) in updated.items() if miss==0))
    def step(self,frame,view):
        a=self.a; shape=frame.shape[:2]; pm=self.masks('packet',frame); p=union(pm,shape); hands=self.masks('hand',frame); h=union(hands,shape); tools=self.masks('scissor',frame); sc=union(tools,shape); busy=bool(area(h)|area(sc)); self.paint(view,pm,(255,140,0)); self.paint(view,hands,(0,150,255)); self.paint(view,tools,(255,0,255)); fg=self.foreground(frame)
        if self.stage==0:
            gold=self.masks('gold',frame); clear=not(pm or hands or tools or gold)
            if self.bg is not None: clear=clear and area(fg)<a.empty_fraction*frame.shape[0]*frame.shape[1]
            if self.stable('empty',clear,2):
                if self.bg is None: self.bg=cv2.GaussianBlur(frame,(7,7),0); self.warn('Empty reference captured from initial scene; undetected objects cannot be excluded')
                self.advance('Initial testbed clear by available detectors')
            return 'Clear the testbed; waiting for empty scene'
        if 1<self.stage<6 and not pm and not busy:
            if self.stable('missing',True,2): self.warn('Registered packet missing; workflow held')
            return 'Packet missing - return same packet'
        self.stable('missing',False)
        if self.stage==1:
            if len(pm)!=1 or busy: self.triggers.pop('ocr',None); return 'Place packet FRONT side up; remove hands and tools'
            if time.monotonic()-self.last_ocr>1:
                self.last_ocr=time.monotonic(); crop=self.number(frame,p); text=self.read_id(crop) if crop is not None else None
                if text and text==getattr(self,'candidate',None): self.ocr_hits=getattr(self,'ocr_hits',0)+1
                else: self.candidate=text; self.ocr_hits=1 if text else 0
                if self.ocr_hits>=2:
                    self.packet_id=text; self.front=frame.copy(); self.front_mask=p.copy(); self.advance('OCR repeated: '+text)
            return f'Place FRONT side up | OCR candidate: {getattr(self,"candidate",None)}'
        if self.stage==2:
            if hands and near(h,p): self.seen_hand=True
            crop=self.number(frame,p) if len(pm)==1 and not busy else None
            strips=self.masks('strip',frame) if len(pm)==1 and not busy else []; self.paint(view,strips,(0,0,255))
            # Front/back is not a trained class. Require handling + disappearance of number + visible appearance change.
            common=p&self.front_mask; delta=cv2.absdiff(frame,self.front).max(axis=2); changed=area(((delta>35)&(common>0)).astype(np.uint8)*255)/max(1,area(common))
            back=self.seen_hand and len(pm)==1 and not busy and crop is None and changed>.15 and area(common)>.25*area(p)
            if self.stable('strip',back and bool(strips),1): self.warn('TopStrip visible after probable turn: NOT READY')
            if self.stable('back',back and not strips,2): self.before=p.copy(); self.advance('Probable backside, TopStrip absent; pre-cut mask saved'); cv2.imwrite(str(self.out/'before_mask.png'),p)
            return 'Turn packet to BACK side | '+('TopStrip visible - NOT READY' if strips else 'Waiting for turn evidence / clear seal view')
        if self.stage==3:
            interaction=bool(hands and tools and near(h,sc) and near(sc,p))
            if self.stable('interaction',interaction,.3,2): self.advance('Hands and scissors interacting with packet; cutting activity observed')
            return 'READY TO CUT - waiting for hand/scissor interaction'
        if self.stage==4:
            if not self.stable('clear_tools',not busy,5,3): self.triggers.pop('cut',None); return 'Cutting activity / waiting 5 seconds after hands and scissors leave'
            cm=self.masks('cut',frame); self.paint(view,cm,(0,220,220)); gap=0.; bg_ok=False; pair=False
            min_area=max(a.min_piece,int(area(self.before)*.015)); pieces=[m for m in pm if area(m)>=min_area]
            if len(pieces)==2:
                m1,m2=pieces; gap=float(cv2.distanceTransform((m1==0).astype(np.uint8),cv2.DIST_L2,5)[m2>0].min()); pair=gap>=a.gap and .40<=area(m1|m2)/max(1,area(self.before))<=1.6
                # Closest contour points; visible background at intermediate points.
                c1=cv2.findNonZero(m1); c2=cv2.findNonZero(m2); dist=cv2.distanceTransform((m1==0).astype(np.uint8),cv2.DIST_L2,5); x2,y2=min(c2[:,0,:],key=lambda pt:dist[pt[1],pt[0]]); x1,y1=min(c1[:,0,:],key=lambda pt:(float(pt[0])-x2)**2+(float(pt[1])-y2)**2); xs=np.rint(np.linspace(x1,x2,9)[2:-2]).astype(int); ys=np.rint(np.linspace(y1,y2,9)[2:-2]).astype(int); bg_ok=bool(np.all(fg[ys,xs]==0)); cv2.line(view,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
            cut_evidence=bool(cm) and any(near(m,p) for m in cm)
            self.evidence.update(gap_px=gap,cut_model_detected=cut_evidence,background_gap=bg_ok)
            if self.stable('cut',pair and bg_ok and cut_evidence, a.cut_seconds): self.cut_ok=True; self.cut_main=max(pieces,key=area); self.advance('Cut confirmed by model + two masks + visible gap (2D evidence)')
            return f'Cut verification | pieces={len(pieces)} gap={gap:.1f}px model={cut_evidence}'
        if self.stage==5:
            if not pm: return 'Hold main packet vertically, mouth towards camera'
            main=max(pm,key=area); ratio,cavity=mouth(frame,main); gold=self.masks('gold',frame); inside=any(overlap(m,main)>.8 for m in gold); pose=bool(hands and near(h,main) and area(main&h)/max(1,area(main))<.15 and area(main)<.85*area(self.before) and not tools)
            raw='OPEN' if pose and ratio>=a.open_ratio and (cavity>=a.cavity or inside) else 'CLOSED' if pose and ratio<=a.closed_ratio and cavity<a.cavity and not inside else 'UNKNOWN'
            self.evidence.update(mouth_ratio=ratio,cavity_ratio=cavity,gold_inside=inside)
            if self.stable('closed',raw=='CLOSED',2): self.mouth_status='CLOSED'; self.warn('Mouth appears closed; waiting for opening')
            if self.stable('open',raw=='OPEN',2): self.mouth_status='OPEN'; self.advance('Mouth appears open using heuristic geometry/interior evidence')
            return f'Hold mouth towards camera | {raw} | ratio={ratio:.2f} cavity={cavity:.3f}'
        if self.stage==6:
            gold=self.masks('gold',frame); outside=[m for m in gold if overlap(m,p)<.05 and overlap(m,h)<.05]; self.paint(view,outside,(0,255,0)); ids=self.track(outside,view); n=len(ids)
            if n: self.ever_gold=True
            # Do not infer empty interior from absence of gold alone. Require a visible open-mouth recheck after unloading, then packet removal.
            if self.ever_gold and pm and hands and not tools:
                main=max(pm,key=area); ratio,cavity=mouth(frame,main); inside=any(overlap(m,main)>.15 for m in gold)
                if self.stable('empty_view', ratio>=a.open_ratio and cavity>=a.cavity and not inside and near(h,main),2):
                    if not self.empty_packet_seen: self.log('Post-unload open mouth observed; no gold detected inside (not proof of empty packet)')
                    self.empty_packet_seen=True
            else: self.stable('empty_view',False)
            if any(overlap(m,p)>.15 for m in gold): self.empty_packet_seen=False
            if ids!=getattr(self,'last_ids',None): self.triggers.pop('count',None); self.last_ids=ids
            if self.stable('count',self.empty_packet_seen and not pm and not busy and n>0,a.count_seconds): self.count=n; self.advance(f'Final visible jewellery count: {n}; packet removed after interior recheck')
            return f'Unload jewellery, show empty mouth, remove packet | visible count={n} | interior recheck={self.empty_packet_seen}'
    def report(self,result):
        data={'result':result,'started':self.started,'ended':stamp(),'packet_id':self.packet_id,'cut_confirmed_2d':self.cut_ok,'mouth_status':self.mouth_status,'visible_jewellery_count':self.count,'interior_recheck':self.empty_packet_seen,'warnings':sorted(self.violations),'metrics':self.evidence,'events':self.events,'limitations':'Experimental 2D heuristics; no proof of actual depth, physical identity, seal integrity or complete unloading.'}
        (self.out/'report.json').write_text(json.dumps(data,indent=2),encoding='utf8'); self.event.emit('Report: '+str(self.out/'report.json'))
    def run(self):
        cap=None; result='STOPPED / INCOMPLETE'
        try:
            missing=[str(self.model_path(k)) for k in FILES if not self.model_path(k).is_file()]
            if missing: raise FileNotFoundError('Missing models: '+', '.join(missing))
            if self.a.empty_reference:
                ref=cv2.imread(self.a.empty_reference)
                if ref is None: raise ValueError('Cannot read empty reference')
                self.bg=cv2.GaussianBlur(ref,(7,7),0)
            cap=cv2.VideoCapture(self.a.camera); cap.set(cv2.CAP_PROP_FRAME_WIDTH,self.a.width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,self.a.height)
            if not cap.isOpened(): raise RuntimeError('Camera unavailable')
            self.log('Inspection started'); failures=0
            while not self.isInterruptionRequested() and self.stage<7:
                ok,frame=cap.read()
                if not ok:
                    failures+=1; self.triggers.clear(); self.status.emit('Camera read failed'); time.sleep(.1)
                    if failures>=20: raise RuntimeError('Camera disconnected')
                    continue
                failures=0
                if self.bg is not None and self.bg.shape!=frame.shape: raise ValueError('Empty reference resolution differs from camera')
                self.last_frame=frame.copy(); view=frame.copy(); detail=self.step(frame,view); self.status.emit(f'{self.stage+1 if self.stage<7 else 7}/7 | {STAGES[self.stage]} | ID: {self.packet_id or "--"}\n{detail}')
                rgb=cv2.cvtColor(view,cv2.COLOR_BGR2RGB); hh,ww=rgb.shape[:2]; self.image.emit(QImage(rgb.data,ww,hh,rgb.strides[0],QImage.Format.Format_RGB888).copy())
            if self.stage==7: result='COMPLETED WITH WARNINGS' if self.violations else 'COMPLETED (HEURISTIC)'; self.status.emit(f'{result}\nPacket {self.packet_id} | Cut: {self.cut_ok} | Mouth: {self.mouth_status} | Visible jewellery: {self.count}')
        except Exception as e:
            result='ERROR / INCOMPLETE'; self.log(str(e),'ERROR'); self.status.emit(str(e)); traceback.print_exc()
        finally:
            if cap is not None: cap.release()
            self.report(result)

class Window(QWidget):
    def __init__(self,a):
        super().__init__(); self.a=a; self.worker=None; self.setWindowTitle('EMBSYS | Packet-Cut Automatic Inspection'); self.resize(1100,800)
        self.preview=QLabel('Camera preview'); self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter); self.preview.setMinimumSize(320,240); self.status=QLabel('START with a clear testbed'); self.status.setWordWrap(True); self.logs=QPlainTextEdit(); self.logs.setReadOnly(True); self.logs.setMaximumHeight(180)
        self.start=QPushButton('START'); self.stop=QPushButton('STOP'); self.stop.setEnabled(False); self.start.clicked.connect(self.begin); self.stop.clicked.connect(self.halt); row=QHBoxLayout(); row.addWidget(self.start); row.addWidget(self.stop); layout=QVBoxLayout(self); layout.addLayout(row); layout.addWidget(self.status); layout.addWidget(self.preview,1); layout.addWidget(self.logs)
    def begin(self):
        self.logs.clear(); self.worker=Worker(self.a); self.worker.image.connect(self.show_frame); self.worker.status.connect(self.status.setText); self.worker.event.connect(self.logs.appendPlainText); self.worker.finished.connect(self.done); self.start.setEnabled(False); self.stop.setEnabled(True); self.worker.start()
    def show_frame(self,im): self.preview.setPixmap(QPixmap.fromImage(im).scaled(self.preview.size(),Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation))
    def halt(self):
        if self.worker: self.worker.requestInterruption(); self.stop.setEnabled(False)
    def done(self): self.start.setEnabled(True); self.stop.setEnabled(False)
    def closeEvent(self,e):
        if self.worker and self.worker.isRunning(): self.halt(); e.ignore(); self.worker.finished.connect(self.close)
        else: e.accept()

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--models',default=str(ROOT/'models')); p.add_argument('--jewel-model'); p.add_argument('--camera',type=int,default=0); p.add_argument('--device',default='cpu'); p.add_argument('--width',type=int,default=640); p.add_argument('--height',type=int,default=480); p.add_argument('--output',default=str(ROOT/'packet_cut_outputs')); p.add_argument('--empty-reference'); p.add_argument('--id-regex',default=r'\d{6}'); p.add_argument('--ocr-conf',type=float,default=.75); p.add_argument('--conf',type=float,default=.25); p.add_argument('--min-piece',type=int,default=300); p.add_argument('--gap',type=float,default=8); p.add_argument('--bg-delta',type=int,default=35); p.add_argument('--empty-fraction',type=float,default=.015); p.add_argument('--open-ratio',type=float,default=.50); p.add_argument('--closed-ratio',type=float,default=.20); p.add_argument('--cavity',type=float,default=.03); p.add_argument('--cut-seconds',type=float,default=10); p.add_argument('--count-seconds',type=float,default=3); p.add_argument('--max-frame-gap',type=float,default=3); a=p.parse_args(); app=QApplication([]); w=Window(a); w.show(); app.exec()
if __name__=='__main__': main()
