"""Rendering of recorded simulation states, with no influence on control."""
from __future__ import annotations
import csv
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
W, H = 1920, 1080
BG = '#0A1424'
PANEL = '#122238'
EDGE = '#29415C'
TEXT = '#F0F4F7'
MUTED = '#A2B5C9'
MINT = '#65E2C3'
GOLD = '#FFD177'
BLUE = '#7AB3F5'

@lru_cache(maxsize=60)
def font(size, bold=False):
    name = 'NotoSansJP-Bold.ttf' if bold else 'NotoSansJP-Regular.ttf'
    return ImageFont.truetype(str(ROOT/'assets/fonts'/name), int(size))

def text(d, pos, value, size=26, fill=TEXT, bold=False, anchor=None):
    d.text(pos, str(value), font=font(size,bold), fill=fill, anchor=anchor)

def card(d, box, fill=PANEL, outline=EDGE, radius=22):
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=2)

def header(title='矢印を読んで、進路を選ぶ'):
    im = Image.new('RGB',(W,H), BG)
    d = ImageDraw.Draw(im)
    text(d,(64,22),'PHYSICAL AI  /  FINAL PROJECT',19,MINT,True)
    text(d,(60,53),'SIGN NAV',65,TEXT,True)
    text(d,(465,87),title,32,TEXT,True)
    text(d,(1856,37),'Toshigakusei',20,MUTED,anchor='ra')
    text(d,(1856,74),'合成標識・平面車輪運動モデル',20,MUTED,anchor='ra')
    d.line((64,153,1856,153),fill=EDGE,width=2)
    return im

class Scene:
    def __init__(self, demo_dir=None):
        self.demo_dir = Path(demo_dir or ROOT/'results/demo')
        self.world=json.loads((self.demo_dir/'world.json').read_text())
        self.mission=json.loads((self.demo_dir/'mission.json').read_text())
        with (self.demo_dir/'telemetry.csv').open() as f:
            self.rows=list(csv.DictReader(f))
        self.times=np.array([float(r['time']) for r in self.rows])
        self.observations=self.mission['observations']
        self.images=[Image.open(self.demo_dir/o['image']).convert('RGB') for o in self.observations]
        self.duration=float(self.mission['result']['duration_seconds'])
        self.xy=[self.project(float(r['x']),float(r['y'])) for r in self.rows]
        self.base=self.draw_base()

    @staticmethod
    def project(x,y,z=0):
        return (int(164+94*(y+.6)+12*x),int(550+104*x-8*(y-5)-83*z))

    def poly(self,d,points,fill,outline=None):
        d.polygon([self.project(*p) for p in points],fill=fill,outline=outline)

    def box(self,d,x,y,dx,dy,z,color='#344D66'):
        p=[(x-dx/2,y-dy/2),(x+dx/2,y-dy/2),(x+dx/2,y+dy/2),(x-dx/2,y+dy/2)]
        self.poly(d,[(a,b,0) for a,b in p], '#17283B')
        for j in (0,1):
            a,b=p[j],p[(j+1)%4]
            self.poly(d,[(a[0],a[1],0),(b[0],b[1],0),(b[0],b[1],z),(a[0],a[1],z)],'#263A50',EDGE)
        self.poly(d,[(a,b,z) for a,b in p],color,'#55718C')

    def draw_base(self):
        im=header()
        d=ImageDraw.Draw(im)
        card(d,(64,184,1322,850))
        text(d,(90,207),'RECORDED SIMULATION',20,MINT,True)
        text(d,(1292,210),'走行 1倍速',22,MUTED,anchor='ra')
        xmin,xmax,ymin,ymax=self.world['bounds']
        self.poly(d,[(xmin,ymin,0),(xmax,ymin,0),(xmax,ymax,0),(xmin,ymax,0)],'#1B3046','#36516A')
        for y in np.arange(0,11.1,.5):
            d.line((self.project(xmin,y),self.project(xmax,y)),fill='#233B51',width=1)
        for x in np.arange(-1.5,1.6,.5):
            d.line((self.project(x,ymin),self.project(x,ymax)),fill='#233B51',width=1)
        # Routes indicate the known practice-course geometry.
        d.line((self.project(0,-.35),self.project(0,10.85)),fill='#62778A',width=6)
        for branch in self.world['branches']:
            for key in ('left','right'):
                points=[branch['junction']]+branch[key]
                d.line([self.project(*p) for p in points],fill='#62778A',width=6,joint='curve')
        for ob in sorted(self.world['obstacles'],key=lambda r:r['x']):
            if ob['kind']=='pillar': self.box(d,ob['x'],ob['y'],ob['width'],ob['height'],.32)
        for i,s in enumerate(self.world['signs']):
            a=self.project(s['x']-.36,s['y'],0)
            b=self.project(s['x']-.36,s['y'],1.12)
            d.line((a,b),fill='#7D95AB',width=5)
            q=self.project(s['x'],s['y'],1.05)
            d.rounded_rectangle((q[0]-33,q[1]-33,q[0]+33,q[1]+33),radius=7,fill='#D5E2EB')
            im.paste(self.images[i].resize((58,58)),(q[0]-29,q[1]-29))
            text(d,(q[0],q[1]-69),f'SIGN 0{i+1}',19,MUTED,True,anchor='ma')
        for value,p,color in [('START',(0,-.35),BLUE),('GOAL',(0,10.85),MINT)]:
            a=self.project(*p)
            d.ellipse((a[0]-20,a[1]-20,a[0]+20,a[1]+20),outline=color,width=3)
            text(d,(a[0],a[1]+31),value,18,color,True,anchor='ma')
        text(d,(90,807),'灰線：経路  緑線：走行軌跡  左右は進行方向を基準',21,MUTED)
        card(d,(1350,184,1856,1006))
        text(d,(1380,208),'AI INPUT',20,MINT,True)
        text(d,(1380,239),'合成標識画像',27,TEXT,True)
        text(d,(1380,576),'3クラスの出力スコア',21,MUTED)
        text(d,(1380,974),'表示値はモデルの出力',17,MUTED)
        return im

    def robot(self,d,row):
        x,y,th=float(row['x']),float(row['y']),float(row['theta'])
        u,v=self.project(x,y)
        d.ellipse((u-32,v-26,u+32,v+26),outline=MINT,width=2)
        f=np.array([math.cos(th),math.sin(th)])
        side=np.array([-math.sin(th),math.cos(th)])
        def loc(a,b,z):
            p=np.array([x,y])+f*a+side*b
            return self.project(*p,z)
        for s in (-.17,.17):
            pts=[loc(a,b,.08) for a,b in [(-.13,s-.04),(.13,s-.04),(.13,s+.04),(-.13,s+.04)]]
            d.polygon(pts,fill='#07111E',outline='#748CA2')
        p0=[loc(a,b,.03) for a,b in [(-.19,-.135),(.19,-.135),(.19,.135),(-.19,.135)]]
        p1=[loc(a,b,.22) for a,b in [(-.19,-.135),(.19,-.135),(.19,.135),(-.19,.135)]]
        for j in range(4):
            d.polygon([p0[j],p0[(j+1)%4],p1[(j+1)%4],p1[j]],fill='#C48A34')
        d.polygon(p1,fill=GOLD,outline='#FFE7B5')
        c=loc(.20,0,.22)
        d.ellipse((c[0]-4,c[1]-4,c[0]+4,c[1]+4),fill=BLUE)
        end=loc(.43,0,0)
        d.line(((u,v),end),fill=MINT,width=3)

    def frame(self,t):
        idx=min(int(np.searchsorted(self.times,t,side='right')-1),len(self.rows)-1)
        idx=max(0,idx)
        row=self.rows[idx]
        im=self.base.copy(); d=ImageDraw.Draw(im)
        if idx>0:d.line(self.xy[:idx+1],fill=MINT,width=7,joint='curve')
        self.robot(d,row)
        latest=-1
        for k,o in enumerate(self.observations):
            if o['time']<=t+1e-7:latest=k
        if latest>=0:
            im.paste(self.images[latest].resize((240,240),Image.Resampling.NEAREST),(1483,310))
            p=self.observations[latest]['prediction']
            values=p['probabilities']
            name=['左の枝へ','中央を直進','右の枝へ'][p['label']]
            text(d,(1603,854),name,35,MINT,True,anchor='ma')
        else:
            d.rounded_rectangle((1483,310,1723,550),radius=12,fill='#1C324A',outline=EDGE,width=2)
            text(d,(1603,395),'読取り位置へ',25,MUTED,True,anchor='ma')
            values=[0,0,0]
        names=['LEFT / 左','STRAIGHT / 直進','RIGHT / 右']
        for j,val in enumerate(values):
            yy=623+j*67
            text(d,(1380,yy),names[j],20,TEXT,True)
            text(d,(1826,yy),f'{val:.3f}',20,MUTED,anchor='ra')
            d.rounded_rectangle((1380,yy+37,1826,yy+49),radius=6,fill='#2B4158')
            if val>0:d.rounded_rectangle((1380,yy+37,1380+max(3,int(446*val)),yy+49),radius=6,fill=MINT)
        text(d,(1380,927),f"観測済み {latest+1} / 3",21,MUTED)
        text(d,(1826,927),f"{t:05.1f} s",22,TEXT,True,anchor='ra')
        phase=row['state']
        state={'APPROACH':'標識へ接近','ALIGN':'停止・向きを調整','READ_SIGN':'画像AIで認識','FOLLOW_BRANCH':'選んだ枝を走行','GOAL':'ゴールへ移動','FINISH':'ゴールで停止','COMPLETE':'完了'}.get(phase,phase)
        boxes=[(64,880,466,1006),(486,880,889,1006),(909,880,1322,1006)]
        for b in boxes:card(d,b)
        text(d,(88,896),'STATE',17,MUTED,True)
        text(d,(88,934),state,27,TEXT,True)
        text(d,(510,896),'WHEELS / 左・右',17,MUTED,True)
        text(d,(510,939),f"{float(row['wheel_left']):.2f}   /   {float(row['wheel_right']):.2f} rad/s",23,GOLD,True)
        text(d,(933,896),'BODY SPEED',17,MUTED,True)
        text(d,(933,932),f"{float(row['linear']):.3f} m/s",31,TEXT,True)
        text(d,(64,1030),'車輪状態を積分した走行ログを描画。標識画像は手続き生成した入力パッチ。',19,MUTED)
        return im

def intro(scene):
    im=header('画像AIと移動をつなぐ学習デモ')
    d=ImageDraw.Draw(im)
    text(d,(96,225),'矢印を読み、',60,TEXT,True)
    text(d,(96,312),'指定された枝を進む。',60,TEXT,True)
    text(d,(100,440),'合成した標識を、左・直進・右の3つに分類。',30,MUTED)
    text(d,(100,493),'分類した結果を、決めておいた経路の選択に使います。',30,MUTED)
    for i,img in enumerate(scene.images):
        xx=1150+i*220
        card(d,(xx,235,xx+190,490))
        im.paste(img.resize((150,150)),(xx+20,258))
        text(d,(xx+95,433),['LEFT','STRAIGHT','RIGHT'][i],22,TEXT,True,anchor='ma')
    for i,(title,sub) in enumerate([('画像を分類','小さなニューラルネット'),('進む枝を選ぶ','左・直進・右'),('車輪を制御','経由点へ向かって移動')]):
        x=96+i*585
        card(d,(x,650,x+535,848))
        text(d,(x+28,686),title,34,TEXT,True)
        text(d,(x+28,760),sub,25,MUTED)
        if i<2:text(d,(x+550,720),'→',28,MINT,True)
    text(d,(96,930),'合成標識 / 固定コース / 平面車輪運動モデル / 走行1倍速',25,MUTED)
    return im

def summary_screen(scene):
    im=header('今回の実装で確認したこと')
    d=ImageDraw.Draw(im)
    result=scene.mission['result']
    text(d,(98,218),'画像の分類を、進路の選択につなげる。',46,TEXT,True)
    lines=[('AIの担当','標識画像を、左・直進・右に分類する。'),
           ('制御プログラムの担当','選んだ経由点へ向かうように車輪を動かす。'),
           ('このデモの結果',f"3つの標識に従って完走。走行時間 {result['duration_seconds']:.2f} 秒。")]
    for i,(a,b) in enumerate(lines):
        yy=343+i*165
        card(d,(98,yy,1822,yy+138))
        text(d,(128,yy+21),a,23,MINT,True)
        text(d,(128,yy+64),b,31,TEXT,True)
    text(d,(98,886),'合成標識と固定コースでの学習デモ。強い画像変動への対応は今後の課題。',27,MUTED)
    text(d,(98,977),'制作支援：設計・コード・資料に生成AIを使用。数値は同梱コードの実行結果。',20,MUTED)
    return im
