"""Create a narrated-by-captions MP4 from the saved simulation; no graphics server."""
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path

from video_ui import ROOT, W, H, Scene, intro, summary_screen

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,default=ROOT/'deliverables/動画_PAI最終課題_Toshigakusei.mp4')
    ap.add_argument('--preview',action='store_true')
    args=ap.parse_args()
    scene=Scene()
    frames=ROOT/'results/video_frames'; frames.mkdir(parents=True,exist_ok=True)
    for old in frames.glob('*.png'):old.unlink()
    screens=[intro(scene),summary_screen(scene)]
    for name,im in zip(['intro','summary'],screens):im.save(frames/(name+'.png'))
    for t in [0,6.2,12,29.8,44.05,53,68.05]:scene.frame(t).save(frames/(f'run_{t:05.2f}.png'))
    if args.preview:return
    fps=20
    args.out.parent.mkdir(parents=True,exist_ok=True)
    command=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24',
             '-s',f'{W}x{H}','-r',str(fps),'-i','-','-an','-c:v','libx264','-threads','2',
             '-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(args.out)]
    pipe=subprocess.Popen(command,stdin=subprocess.PIPE)
    count=0
    def write(im):
        nonlocal count
        pipe.stdin.write(im.tobytes());count+=1
    try:
        for _ in range(6*fps):write(screens[0])
        for j in range(int(round(scene.duration*fps))+1):
            write(scene.frame(min(j/fps,scene.duration)))
            if j%400==0:print(f'rendered simulation {j/fps:.0f} / {scene.duration:.1f}s',flush=True)
        for _ in range(8*fps):write(screens[1])
    finally:
        pipe.stdin.close()
    if pipe.wait()!=0:raise RuntimeError('FFmpeg failed')
    metadata={'width':W,'height':H,'fps':fps,'frames':count,'duration_seconds':count/fps,
              'simulation_seconds':scene.duration,'simulation_playback_speed':1,
              'sections_seconds':{'intro':6,'simulation':(int(round(scene.duration*fps))+1)/fps,
                                  'summary':8},
              'rendering':'saved planar wheel-motion trajectory rendered with Pillow',
              'audio':'none; Japanese captions',
              'video_file':args.out.name}
    (ROOT/'results/video_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(metadata,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
