"""Bundle current topic A code into a Kaggle notebook without publishing to Git.

Run from the repository root. Output must be a new filename; existing notebooks
are never replaced. The dataset is read from a pinned, existing Git revision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/conanWinner/DoanQuangThang-2A202602395-Track4-Day21.git"
DATA_REVISION = "bce73adec3dbd09b2869ed2061513328ee228272"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "src/topic_a_kaggle.ipynb")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Notebook exists; select a new --out filename.")
    paths = sorted((ROOT / "starter").glob("*.py"))
    paths += [ROOT / "src/__init__.py", ROOT / "src/calibration_benchmark.py"]
    tests = sorted((ROOT / "src/tests").glob("test_*.py"))
    if not tests:
        parser.error("Required projection/benchmark tests are missing.")
    paths += tests
    dependencies = (ROOT / "src/requirements-kaggle.txt").read_text().splitlines()
    dependencies = [line.strip() for line in dependencies if line.strip() and not line.startswith('#')]
    payload = {str(p.relative_to(ROOT)): p.read_text(encoding="utf-8") for p in paths}
    hashes = {name: hashlib.sha256(code.encode()).hexdigest() for name, code in payload.items()}
    cells = []

    def markdown(source):
        cells.append(dict(cell_type="markdown", metadata={}, source=source.splitlines(True)))

    def code(source):
        cells.append(dict(cell_type="code", metadata={}, execution_count=None,
                          outputs=[], source=source.splitlines(True)))

    markdown("""# Topic A — Kiểm tra calibration LiDAR–camera

**Đoàn Quang Thắng · 2A202602395 · Track 4 Day21**

Chạy toàn bộ 20 frame KITTI, 80 frame nuScenes và 5 frame synthetic trên CPU.
Thay đổi từng góc/hướng dịch chuyển riêng; đo FOV, độ lệch pixel, điểm trong box
và thử score khoảng cách tới biên Canny. Mọi bảng và ảnh được tạo từ lần chạy này.

Code được đóng gói từ bản địa phương, kiểm tra SHA-256 trước chạy; dữ liệu lấy từ
Git revision cố định. Không cần push Git hoặc tạo Kaggle dataset.
Nguồn ảnh: KITTI Vision Benchmark Suite; nuScenes (Motional), dùng cho học tập.
AI hỗ trợ code/rà soát: Codex qua Herdr; kiểm chứng bằng test và chạy dữ liệu thật.
""")
    code("import subprocess, sys\n" + f"DEPENDENCIES = {dependencies!r}\n" +
         "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *DEPENDENCIES], check=True)\n")
    code("""from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, uuid

run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
data_checkout = Path('/tmp') / ('day21_data_' + run_id)
run_root = Path('/tmp') / ('day21_code_' + run_id)
run_root.mkdir(exist_ok=False)
""" + f"REPO = {REPO!r}\nDATA_REVISION = {DATA_REVISION!r}\n" + """
subprocess.run(['git', 'clone', '--quiet', '--no-checkout', REPO, str(data_checkout)], check=True)
subprocess.run(['git', '-C', str(data_checkout), 'checkout', '--quiet', '--detach', DATA_REVISION], check=True)
assert subprocess.check_output(['git', '-C', str(data_checkout), 'rev-parse', 'HEAD'], text=True).strip() == DATA_REVISION
(run_root / 'data').symlink_to(data_checkout / 'data', target_is_directory=True)
""" + f"PAYLOAD = {payload!r}\nSOURCE_HASHES = {hashes!r}\n" + """
for name, source in PAYLOAD.items():
    destination = run_root / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8') as stream:
        stream.write(source)
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == SOURCE_HASHES[name]
os.chdir(run_root)
sys.path.insert(0, str(run_root))
output = Path('/kaggle/working/results') / ('topic_a_' + run_id)
print('Data revision:', DATA_REVISION)
print('Source hashes verified:', len(SOURCE_HASHES))
print('Fresh output:', output)
""")
    code("""import unittest
suite = unittest.defaultTestLoader.discover('src/tests')
assert suite.countTestCases() > 0, 'Tests must not be silently skipped'
result = unittest.TextTestRunner(verbosity=2).run(suite)
assert result.wasSuccessful(), 'Required tests failed'
""")
    code("""subprocess.run([sys.executable, '-m', 'src.calibration_benchmark',
                '--out-dir', str(output), '--seed', '2026', '--latency-repeats', '30'], check=True)
completion = json.loads((output / 'completion.json').read_text())
assert completion['status'] == 'complete'
import pandas as pd
metrics = pd.read_csv(output / 'per_frame_config.csv')
expected_frames = {'kitti_mini': 20, 'nuscenes_mini_subset': 80, 'synthetic': 5}
assert metrics.groupby('dataset').frame.nunique().to_dict() == expected_frames
assert len(metrics) == 105 * 43
assert not metrics.duplicated(['dataset', 'frame', 'config']).any()
assert (metrics.groupby(['dataset', 'frame']).config.nunique() == 43).all()
for artifact in ['summary.csv', 'score_evaluation.json', 'fail_01_facts.json', 'latency_summary.csv']:
    assert (output / artifact).is_file(), artifact
with (output / 'kaggle_source_manifest.json').open('x') as stream:
    json.dump({'data_revision': DATA_REVISION, 'source_hashes': SOURCE_HASHES,
               'run_id': run_id, 'dependencies': DEPENDENCIES,
               'test_count': suite.countTestCases()}, stream, indent=2)
print(completion)
""")
    markdown("""## Kết quả và giới hạn

Tỷ lệ điểm đúng box cố định nhóm điểm theo box 3D của lần không gây lệch;
điểm trôi ra ngoài ảnh được tính sai. Độ lệch pixel chỉ đo trên điểm còn nhìn thấy
ở cả hai lần; đọc kèm coverage để tránh che mất điểm rơi khỏi ảnh.

nuScenes sinh box 2D từ nhãn 3D nên kết quả box không phải kiểm chứng độc lập.
Ngưỡng score chỉ lấy từ baseline nhóm calibration; nhóm test không dùng để chỉnh
ngưỡng. Các frame gần nhau tương quan: phép thử cho thấy độ nhạy trong tập này,
chưa xác nhận khả năng phát hiện drift trên xe thật. Không dùng latency máy này
thay cho latency máy địa phương.
""")
    code("""import pandas as pd
from IPython.display import display, Image
summary = pd.read_csv(output / 'summary.csv')
display(summary[(summary.config == 'zero') | ((summary.factor == 'yaw') & summary.value.isin([1,2,3]))])
display(pd.read_csv(output / 'latency_summary.csv'))
score_results = json.loads((output / 'score_evaluation.json').read_text())
display(pd.DataFrame(score_results['results']))
failure = json.loads((output / 'fail_01_facts.json').read_text())
print('Failure selected by:', failure['selection_rule'])
print(json.dumps(failure['metrics'], indent=2))
figure_root = output / 'figures' if (output / 'figures').is_dir() else output
for figure in sorted(figure_root.glob('*.png')):
    print(figure.name)
    display(Image(filename=str(figure), width=1100))
""")
    notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                        language_info=dict(name="python")), nbformat=4, nbformat_minor=5)
    for index, cell in enumerate(cells):
        cell["id"] = f"topic-a-{index:02d}"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(notebook, stream, indent=1, ensure_ascii=False)
    print(args.out)
    print(json.dumps(hashes, indent=2))


if __name__ == "__main__":
    main()
