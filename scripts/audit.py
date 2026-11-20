#!/usr/bin/env python3
"""Inspect candidate/tracked Git content without displaying secret values."""
from pathlib import Path
import ast
import hashlib
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def main():
    candidates = subprocess.check_output(['git','-C',str(ROOT),'ls-files','--cached','--others','--exclude-standard','-z']).decode().split('\0')
    import sys
    sys.path.insert(0,str(ROOT))
    from scripts.cleanup_layout import asset_location
    secrets = set()
    for p in ROOT.rglob('*.local.json'):
        def visit(v):
            if isinstance(v,dict):
                for k,x in v.items():
                    if k in ('password','passwd') and isinstance(x,str) and x:
                        secrets.add(x)
                    else:
                        visit(x)
            elif isinstance(v,str) and v.startswith('rtsp://'):
                from urllib.parse import urlsplit, unquote
                password=urlsplit(v).password
                if password:
                    secrets.add(unquote(password))
            elif isinstance(v,list):
                for x in v:visit(x)
        visit(json.loads(p.read_text()))
    findings=[]
    syntax=[]
    binary=[]
    for name in filter(None,candidates):
        p=ROOT/name
        if p.suffix.lower() in {'.onnx','.pt','.pth','.bin','.xml','.task','.npz','.npy','.hef','.joblib','.pkl','.pickle','.onnx_data'}:
            binary.append(name)
        if not p.is_file():continue
        data=p.read_bytes()
        if any(secret.encode() in data for secret in secrets):findings.append(name+' (migrated password literal)')
        if p.suffix in {'.py','.md','.sh','.json','.txt','.ps1','.diff'}:
            text=data.decode(errors='replace')
            if re.search(r'rtsp://[^\s/:]+:[^\s/@]+@',text):findings.append(name+' (literal RTSP credentials)')
            if p.suffix=='.py':
                try:ast.parse(text)
                except SyntaxError:syntax.append(name)
    snapshot=json.loads((ROOT/'docs/SOURCE_SNAPSHOT.json').read_text())
    originals=[]
    for name,entry in snapshot.items():
        path=entry['source']
        for cmd,key in [(['status','--porcelain=v1'],'status'),(['branch','--show-current'],'branch'),(['rev-parse','HEAD'],'head')]:
            actual=subprocess.check_output(['git','-C',path]+cmd,text=True)
            if (actual if key=='status' else actual.strip())!=entry[key]:originals.append(name+': '+key+' changed')
    source_file_changes=[]
    manifest=json.loads((ROOT/'docs/COPY_MANIFEST.json').read_text())
    for relative, expected in manifest.items():
        name, rel=relative.split('/',1)
        path=Path(snapshot[name]['source'])/rel
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            source_file_changes.append(relative)
    unexpected_function_changes=[]
    allowed={
        'Jeweltracker/jewel_tracker_5.py': {'MainWindow.review_acid_recordings', 'MainWindow.__init__'},
        'Jeweltracker/packet_stage.py': {'packet_module', 'packet_wait_reason', 'testing_snapshot',
            'PacketStage.__init__', 'PacketStage.begin', 'PacketStage.tick', 'PacketStage.finish_analysis', 'PacketStage.release_gold'},
        'Packetsealing_001/standalone/tracking.py': {'TrackingBridge.__init__', 'TrackingBridge.process', 'TrackingBridge.pickup_event', 'TrackingBridge.reset_assignment'},
        'Packetsealing_001/standalone/app.py': {'PacketWorker.__init__', 'PacketWorker._restart_assignment', 'PacketWorker._commands', 'PacketWorker.run', 'PacketWorker._observations', 'MainWindow.show_state'},
        'Packetsealing_001/standalone/evidence.py': {'export_session'},
        'Packetsealing_001/standalone/report.py': {'write_session_report'},
        'Jeweltracker/jewel_tracker_similarity.py': {'MainWindow.__init__'},
    }
    def functions(tree):
        result={}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                result[node.name]=ast.dump(node)
            elif isinstance(node, ast.ClassDef):
                for method in node.body:
                    if isinstance(method, ast.FunctionDef):
                        result[node.name+'.'+method.name]=ast.dump(method)
        return result
    for relative in manifest:
        if not relative.endswith('.py') or '/tests/' in relative:
            continue
        name, rel=relative.split('/',1)
        old=functions(ast.parse((Path(snapshot[name]['source'])/rel).read_text()))
        location=asset_location(relative)
        if location is None:
            unexpected_function_changes.append(relative+': missing cleanup resolution')
            continue
        new=functions(ast.parse(location.read_text()))
        for function in old:
            if old[function]!=new.get(function) and function not in allowed.get(relative,set()):
                unexpected_function_changes.append(relative+':'+function)
    model_names=[str(p.relative_to(ROOT)) for base in (ROOT/'engines',ROOT/'development') for p in base.rglob('*') if p.is_file() and p.suffix.lower() in {'.onnx','.pt','.pth','.bin','.xml','.task','.npz','.npy','.hef','.joblib','.pkl','.pickle','.onnx_data'}]
    ignored=subprocess.check_output(['git','-C',str(ROOT),'check-ignore','--stdin'],input='\n'.join(model_names)+'\n',text=True).splitlines()
    report={'candidate_files':len(list(filter(None,candidates))),'secret_findings':findings,'syntax_errors':syntax,'model_candidates':binary,'models_ignored':len(ignored),'model_files':len(model_names),'original_repository_changes':originals,'original_file_changes':source_file_changes,'unexpected_function_changes':unexpected_function_changes}
    (ROOT/'docs/AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return int(bool(findings or syntax or binary or originals or source_file_changes or unexpected_function_changes or len(ignored)!=len(model_names)))

if __name__=='__main__':raise SystemExit(main())
