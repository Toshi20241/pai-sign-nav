"""Create a compact two-page report from this learning demo's saved results."""
from pathlib import Path
from xml.sax.saxutils import escape
import json

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'deliverables/説明資料_PAI最終課題_Toshigakusei.pdf'
PW,PH=A4
M=43
CW=PW-2*M
NAVY='#152B3B'; TEAL='#197C70'; MUTED='#536979'; LINE='#D4DFE4'; PALE='#F1F6F7'

def main():
    for name,file in [('JP','NotoSansJP-Regular.ttf'),('JPB','NotoSansJP-Bold.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(ROOT/'assets/fonts'/file)))
    vision=json.loads((ROOT/'results/vision_metrics.json').read_text())
    bench=json.loads((ROOT/'results/benchmark/summary.json').read_text())
    demo=json.loads((ROOT/'results/demo/mission.json').read_text())
    video=json.loads((ROOT/'results/video_metadata.json').read_text())
    OUT.parent.mkdir(parents=True,exist_ok=True)
    c=canvas.Canvas(str(OUT),pagesize=A4,pageCompression=1)
    c.setTitle('SIGN NAV | 矢印を読んで進路を選ぶ学習デモ')
    c.setAuthor('Toshigakusei')
    c.setSubject('Synthetic arrow recognition and planar robot control learning demo')

    def label(value,x,y,size=10,color=NAVY,bold=False):
        c.setFillColor(colors.HexColor(color));c.setFont('JPB' if bold else 'JP',size)
        c.drawString(x,PH-y-size*.85,value)

    def para(value,y,size=10.3,leading=17,x=M,width=CW,color=NAVY,bold=False):
        s=ParagraphStyle('p',fontName='JPB' if bold else 'JP',fontSize=size,leading=leading,
                         textColor=colors.HexColor(color),wordWrap='CJK')
        p=Paragraph(value,s);_,h=p.wrap(width,PH);p.drawOn(c,x,PH-y-h)
        if y+h>790:raise ValueError(f'Paragraph outside content area: {y+h}')
        return y+h

    def page(n,title,subtitle):
        label('SIGN NAV / PHYSICAL AI',M,30,9,TEAL,True)
        label('Toshigakusei',PW-M-73,30,9,MUTED)
        c.setStrokeColor(colors.HexColor(LINE));c.line(M,PH-49,PW-M,PH-49)
        label(title,M,76,23,NAVY,True)
        para(subtitle,115,10,16,color=MUTED)
        label('合成標識を使った移動ロボットの学習デモ',M,807,8,MUTED)
        label(f'{n} / 2',PW-M-30,807,8,MUTED)

    def section(title,y):
        c.setFillColor(colors.HexColor(TEAL));c.rect(M,PH-y-17,3,17,stroke=0,fill=1)
        label(title,M+11,y,13,NAVY,True)

    def table(headers,rows,y,widths):
        s=ParagraphStyle('cell',fontName='JP',fontSize=9.4,leading=14.5,wordWrap='CJK',textColor=colors.HexColor(NAVY))
        hs=ParagraphStyle('head',parent=s,fontName='JPB',textColor=colors.white)
        cells=[[Paragraph(escape(str(v)),hs) for v in headers]]
        cells += [[Paragraph(escape(str(v)).replace('\n','<br/>'),s) for v in row] for row in rows]
        t=Table(cells,colWidths=widths)
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor(NAVY)),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor(PALE)]),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),9),
            ('RIGHTPADDING',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,-1),7),
            ('BOTTOMPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,-1),.4,colors.HexColor(LINE))]))
        _,h=t.wrap(CW,PH);t.drawOn(c,M,PH-y-h);return y+h

    page(1,'矢印を読んで、進路を選ぶ','画像AIと車輪の制御をつなぐ、小規模な学習課題')
    para('合成した矢印標識を「左・直進・右」の3種類に分類し、その結果に応じて移動ロボットが進む枝を選ぶ。AIの画像認識を、シミュレーション内の行動につなげることを目的とした。',154)
    fig=ROOT/'results/video_frames/run_53.00.png'
    c.drawImage(str(fig),M,PH-221-CW*9/16,width=CW,height=CW*9/16)
    label('図1 左は走行の表示、右はAIへ入力した合成標識画像。',M,518,8.5,MUTED)
    section('動作は、3つの手順',552)
    steps=[('1  標識の画像を分類','読取り地点で停止し、画像AIが左・直進・右のいずれかを出力する。'),
           ('2  進む枝を選ぶ','分類結果を使って、あらかじめ用意した経由点の組を選ぶ。'),
           ('3  車輪を動かす','経由点へ向かうよう左右の車輪速度を計算し、次の標識とゴールへ進む。')]
    for i,(title,body) in enumerate(steps):
        y=582+i*52
        label(title,M,y,11,NAVY,True);para(body,y+20,9.7,15)
    para('AIは標識の分類を担当する。移動の計算は、既知の経由点へ向かう制御プログラムが担当する。',748,9.5,15,color=MUTED)
    c.showPage()

    page(2,'仕組みと確認結果','合成画像・固定コース・平面運動モデルで確認した範囲')
    section('小さな画像分類モデル',153)
    para('MLP（多層パーセプトロン）という基本的なニューラルネットを使用した。画像を小さくして入力し、3種類の矢印を分類する。',182,10,16)
    ev=vision['evaluations']
    rows=[['構造','24×24画素 → 隠れ層1段（32ユニット） → 3分類'],
          ['学習データ','合成画像900枚。別に150枚で学習途中の様子を確認。'],
          ['独立した画像テスト',f"標準条件 {ev['test_standard']['mlp_correct']} / {ev['test_standard']['n']}枚。強い変動 {ev['test_stress']['mlp_correct']} / {ev['test_stress']['n']}枚（{ev['test_stress']['mlp_accuracy']*100:.2f}%）。"]]
    y=table(['項目','内容'],rows,225,[135,CW-135])
    section('6条件での動作確認',y+25)
    rr=[]
    for key,title in [('learned','画像AIで枝を選ぶ'),('straight','常に直進する')]:
        b=bench['policies'][key]
        rr.append([title,f"{b['all_signs_followed']} / {b['runs']}",str(b['contact_events'])])
    y=table(['方式','3標識すべてに従って完走','接触'],rr,y+54,[210,215,CW-425])
    para('標識と開始時の条件を6通りに変え、同じ条件で比較した。コースの枝は合流するため、ゴール到達だけでなく、標識に従ったかを確認した。',y+14,9.4,15)
    section('できる範囲と課題',y+66)
    para('画像は合成標識のパッチで、実カメラによる撮影ではない。コースと自己位置は既知で、車輪による平面移動を計算している。強いぼかし・回転・部分的な隠れでは分類精度が下がるため、実環境への適用は今後の検証課題である。',y+94,9.4,15)
    section('再実行と制作支援',y+157)
    para('python main.py でデモ、python experiments.py で比較を再実行できる。学習済みモデル、画像、走行ログ、再学習コードを同梱した。',y+184,9.1,14)
    para(f"動画は{video['duration_seconds']:.2f}秒、走行部分は1倍速。設計・コード・資料の作成に生成AIを使用した。数値は同梱コードの実行結果である。",y+222,8.7,13.5,color=MUTED)
    label('参照：scikit-learn公式MLP資料。出典URLと実行方法はREADME.mdに記載。',M,778,7.8,MUTED)
    c.save();print(str(OUT))

if __name__=='__main__':main()

