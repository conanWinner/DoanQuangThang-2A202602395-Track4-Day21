"""Create four Vietnamese CP6 slides from existing artifacts; never overwrite output.
Requires reportlab (presentation export only; not required by the benchmark).
Run: python -B src/build_presentation_pdf.py --out output/pdf/day21_topic_a_presentation.pdf
"""
import argparse
import csv
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor

ROOT = Path(__file__).resolve().parents[1]
W, H = 960, 540
INK, MUTED, BLUE, TEAL = '#14213D', '#536278', '#2559DB', '#087F8C'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'output/pdf/day21_topic_a_presentation.pdf')
    parser.add_argument('--font-dir', type=Path, default=Path('/usr/share/fonts/truetype/dejavu'))
    args = parser.parse_args()
    if args.out.exists(): parser.error('Output exists; choose a new filename.')
    pdfmetrics.registerFont(TTFont('VN', str(args.font_dir/'DejaVuSans.ttf')))
    pdfmetrics.registerFont(TTFont('VN-Bold', str(args.font_dir/'DejaVuSans-Bold.ttf')))
    pdfmetrics.registerFontFamily('VN', normal='VN', bold='VN-Bold')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stream = args.out.open('xb')
    c = canvas.Canvas(stream, pagesize=(W,H))
    c.setTitle('Topic A - Calibration LiDAR-camera | Doan Quang Thang')
    c.setAuthor('Đoàn Quang Thắng - 2A202602395; AI hỗ trợ: OpenAI Codex')
    figures = ROOT/'results/figures/topic_a_completed_20261008'
    with (ROOT/'results/topic_a_local_20261008/summary.csv').open() as f:
        rows = list(csv.DictReader(f))

    def text(s,x,y,width,size=18,color=INK,bold=False):
        style = ParagraphStyle('p',fontName='VN-Bold' if bold else 'VN',fontSize=size,leading=size*1.32,textColor=HexColor(color))
        p=Paragraph(escape(s),style); _,height=p.wrap(width,H)
        p.drawOn(c,x,H-y-height)
        return height
    def title(n,s):
        c.setFillColor(HexColor('#F7F9FC')); c.rect(0,0,W,H,fill=1,stroke=0)
        text('TOPIC A / TRACK 4 / DAY 21',40,24,750,11,BLUE,True)
        text(s,40,48,880,29,INK,True)
        text('Đoàn Quang Thắng · 2A202602395',40,510,650,10,MUTED)
        text(f'{n}/4',885,510,40,10,MUTED)
    def image(name,x,y,w,h):
        c.drawImage(str(figures/name),x,H-y-h,width=w,height=h,preserveAspectRatio=True,anchor='c')

    title(1,'Lệch 1°: điểm vẫn trong ảnh, nhưng sai vị trí')
    text('Yaw +1°: lệch trung bình 15,42 pixel trên KITTI và 25,88 pixel trên nuScenes.',40,105,865,22,BLUE,True)
    text('105 frame × 43 cấu hình = 4.515 dòng. Mỗi lần chỉ đổi một yếu tố; seed 2026.',40,172,865,17)
    text('Hai hàm: LiDAR → camera rectified → pixel; loại điểm lỗi, sau camera và ngoài ảnh.',40,207,865,16)
    for x,name,label in [(40,'overlay_kitti_near_000008.png','Gần: Car 4,88 m'),(340,'overlay_kitti_mid_000010.png','Trung: Pedestrian 24,99 m'),(640,'overlay_kitti_far_000019.png','Xa: Car 59,77 m')]:
        image(name,x,266,280,115); text(label,x,395,280,15,INK,True)
    text('Khoảng cách là chuẩn Euclid tâm đáy box 3D, không phải khoảng cách của mọi điểm trong ảnh.',40,450,870,13,MUTED)
    c.showPage()

    title(2,'Yaw tăng: độ lệch tăng, tỷ lệ đúng box giảm')
    headers=['Dataset','Yaw','Lệch (pixel)','Đúng box','Coverage']
    xs=[55,270,385,565,745]
    for x,s in zip(xs,headers): text(s,x,116,180,16,BLUE,True)
    for idx,(dataset,cfg) in enumerate([(d,g) for d in ['kitti_mini','nuscenes_mini_subset'] for g in ['zero','yaw_+1','yaw_+2','yaw_+3']]):
        r=next(r for r in rows if r['dataset']==dataset and r['config']==cfg)
        vals=['KITTI' if dataset=='kitti_mini' else 'nuScenes', '0°' if cfg=='zero' else cfg.replace('yaw_','')+'°',f"{float(r['displacement_px_micro']):.2f}",f"{100*float(r['association_micro']):.2f}%",f"{100*float(r['coverage_micro']):.2f}%"]
        y=153+idx*31
        c.setStrokeColor(HexColor('#DBE3EF'));c.line(50,H-y-27,910,H-y-27)
        for x,s in zip(xs,vals):text(s,x,y,185,16)
    text('Trong ảnh gần như không đổi: KITTI 15,740% → 15,760% khi yaw 0° → +3°.',40,425,875,16,TEAL,True)
    text('Độ lệch chỉ dùng điểm nhìn thấy ở cả hai lần; đúng box giữ mẫu số baseline, điểm ra ngoài ảnh là miss. nuScenes box 2D sinh từ 3D nên không độc lập.',40,462,875,12,MUTED)
    c.showPage()

    title(3,'Failure thật: yaw -3°, score không báo lỗi')
    text('nuScenes scene-1094_015 | Trái: baseline | Phải: yaw -3°',40,105,880,17,BLUE,True)
    image('fail_01_nuscenes_mini_subset_scene-1094_015_yaw_-3.png',40,146,880,250)
    text('78,65 pixel lệch · 72,44% đúng box · Coverage 93,38%',40,408,880,20,INK,True)
    text('Score 0,014708 ≤ ngưỡng 0,036728 → bỏ sót. Lớp Metric: gần biên ảnh bất kỳ chưa chắc gần đúng vật thể; không kết luận tối là nguyên nhân duy nhất.',40,451,880,15)
    c.showPage()

    title(4,'Kết luận: benchmark đã chạy, cảnh báo còn yếu')
    text('Độ nhạy test drift: KITTI 30,00%; nuScenes 13,21%. Báo nhầm baseline: 30% và 15%. Không dùng score này làm cảnh báo duy nhất.',40,110,875,20,BLUE,True)
    text('ADAS: ghi log coverage, chất lượng ảnh, timestamp và score. Thử ghép biên độ sâu với biên ảnh, kiểm tra chuỗi thời gian và holdout scene độc lập.',40,215,875,19)
    text('Đã xác minh: 13 test đạt; Kaggle v1 COMPLETE; bản clone GitHub tái tạo cùng metric. Latency chỉ tính projection, không gồm I/O hay vẽ.',40,307,875,18)
    text('Giới hạn: sai lệch giả lập; frame lân cận còn tương quan; box nuScenes xấp xỉ. Chưa chứng minh phát hiện drift thật trên xe hoặc scene mới.',40,391,875,16,MUTED)
    text('Nguồn: REPORT.md, summary.csv, score_evaluation.json, fail_01_facts.json. AI hỗ trợ: OpenAI Codex; khai báo và cách kiểm chứng trong REPORT mục 6.',40,467,875,11,MUTED)
    c.save(); stream.close(); print(args.out)


if __name__=='__main__': main()
