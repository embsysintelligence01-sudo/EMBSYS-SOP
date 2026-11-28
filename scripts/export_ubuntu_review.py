"""Create a separate Ubuntu-playable review copy without changing saved evidence."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from shared.compat import install
install()
from evidence_archive import RetentionPolicy, checked_local, encoder_available, validate_pdf
from compact_evidence import media_info
from simple_session import encode_budgeted
from audit_report import render_report
from session_evidence import prediction_text


def export_review(source, destination):
    source=Path(source).resolve();destination=Path(destination).resolve()
    data=copy.deepcopy(json.loads((source/'session.json').read_text()))
    policy=RetentionPolicy.load();exe=encoder_available()
    if not exe:raise RuntimeError('FFmpeg is required to export AVI clips')
    destination.mkdir(parents=True,exist_ok=False)
    photos=destination/'photos';photos.mkdir()
    for index,jewel in enumerate(data.get('jewels',[]),1):
        path=jewel.get('image')
        if path and checked_local(source,path).is_file():
            target=photos/f'jewel_{index}.jpg'
            shutil.copy2(checked_local(source,path),target)
            jewel['image']=str(target)
    jobs=[]
    for index,record in enumerate(data['tests'],1):
        compact_report=record.pop('compact_report',None)
        record.pop('acid_candidate_clip',None)
        folder=destination/f'test_{index:02d}'
        folder.mkdir()
        if compact_report and checked_local(source,compact_report).is_file():
            # The immutable report may be the only remaining copy of its photos.
            shutil.copy2(checked_local(source,compact_report),folder/'report.pdf')
            record['compact_report']=str(folder/'report.pdf')
        record['source_record_path']=record.get('record_path')
        record['record_path']=str(folder/'manifest.json')
        record['review_copy']=True
        for key,name in (('rubbing_clip','rubbing.avi'),('acid_clip','acid.avi')):
            clip=record.get(key)
            if not clip or not clip.get('path'):continue
            original=checked_local(source,clip['path'])
            info=media_info(original)
            output=folder/name;output.touch()
            record[key]=dict(clip,path=str(output),source_path=str(original))
            jobs.append((record,index,key,original,output,info['duration_seconds']))
    report=destination/'report.pdf'
    render_report(report,destination,data['jewels'],data['tests'],data.get('recording_problems',[]),
                  prediction_text,simple_output=True)
    validate_pdf(report)
    share=(report.stat().st_size+max(1,len(data['tests']))-1)//max(1,len(data['tests']))
    for record,index,key,original,output,duration in jobs:
        total=sum(job[5] for job in jobs if job[1]==index)
        budget=int((policy.target_test_bytes-share-4096)*duration/total)
        encode_budgeted(original,output,budget,policy,exe)
        if shutil.which('gst-launch-1.0'):
            check=subprocess.run(['gst-launch-1.0','-q','filesrc','location='+str(output),'!',
                'avidemux','!','decodebin','!','fakesink','sync=false'],capture_output=True,text=True,timeout=120)
            if check.returncode:raise RuntimeError('Ubuntu decoding failed: '+check.stderr)
        record[key]['archive_media']=dict(media_info(output),codec='mpeg4',container='avi',fourcc='XVID')
        print('Validated',output,flush=True)
    data['source_session']=str(source)
    data['review_copy']=True
    for record in data['tests']:
        Path(record['record_path']).write_text(json.dumps(record,indent=2))
    (destination/'session.json').write_text(json.dumps(data,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('destination',type=Path)
    args=parser.parse_args()
    print('Review PDF:',export_review(args.source,args.destination))
