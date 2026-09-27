from pathlib import Path
import json
import queue
import sys
import time

import pytest

from product_search.workbench import Experiment, JobRunner, report_rows, discover_resources


def test_report_rows_preserve_meaning_and_missing_values():
    report = {'ranking': {'dense': {'ndcg': {'mean': .8}, 'mrr_e': {'mean': None}}},
              'retrieval': {'dense': {'known_e_recall': {'mean': .9}}}}
    rows = report_rows(report)
    row = next(r for r in rows if r[0] == '语义召回')
    assert row[1:] == ('0.8000', '—', '90.00%', '—')


def test_experiment_order_and_no_overwrite(tmp_path):
    data = tmp_path / 'data/processed/dev-1'
    data.mkdir(parents=True)
    (data / 'manifest.json').write_text('{}')
    exp = Experiment.create(tmp_path, data)
    other = Experiment.create(tmp_path, data)
    assert exp.index_dir != other.index_dir
    assert exp.index_dir.is_relative_to(tmp_path / 'artifacts')
    assert exp.command('build')[0] == 'build-indexes'
    with pytest.raises(ValueError, match='索引'):
        exp.command('train')
    exp.index_dir.mkdir(parents=True)
    (exp.index_dir / 'release.json').write_text('{}')
    assert exp.command('train')[0] == 'train-ranker'
    with pytest.raises(FileExistsError):
        exp.command('build')
    with pytest.raises(ValueError):
        exp.command('test')


def test_runner_returns_output_and_nonzero_exit(tmp_path):
    events = queue.Queue()
    runner = JobRunner(tmp_path, events)
    runner.start([sys.executable, '-u', '-c', "print('hello');raise SystemExit(3)"], label='probe')
    seen = []
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        item = events.get(timeout=5)
        seen.append(item)
        if item[0] == 'done':
            break
    assert any(event[0] == 'log' and 'hello' in event[1] for event in seen)
    assert seen[-1] == ('done', 3)
    assert not runner.busy


def test_runner_blocks_parallel_and_stops_only_owned_process(tmp_path):
    runner = JobRunner(tmp_path, queue.Queue())
    runner.start([sys.executable, '-c', 'import time;time.sleep(30)'], label='probe')
    with pytest.raises(RuntimeError):
        runner.start([sys.executable, '-c', 'pass'], label='probe2')
    runner.stop()
    deadline = time.monotonic() + 5
    while runner.busy and time.monotonic() < deadline:
        time.sleep(.05)
    assert not runner.busy


def test_resource_scan_ignores_pending_and_broken_reports(tmp_path):
    artifact = tmp_path / 'artifacts/.broken'
    artifact.mkdir(parents=True)
    (artifact / 'release.json').write_text('{}')
    report = tmp_path / 'reports/valid'
    report.mkdir(parents=True)
    (report / 'ranking-results.json').write_text(json.dumps({'partition':'validation','queries':2,'ranking':{}}))
    resources = discover_resources(tmp_path)
    assert resources['releases'] == []
    assert len(resources['reports']) == 1


def test_desktop_pages_navigate_and_existing_outputs_disable_steps():
    tk = pytest.importorskip('tkinter')
    from product_search.desktop import WorkbenchApp

    try:
        window = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f'Tk graphical display unavailable: {exc}')
    window.withdraw()
    app = WorkbenchApp(window, Path(__file__).parents[1])
    try:
        for page in ('体验搜索', '结果解读', '重做实验', '学习路线'):
            app.show_page(page)
            assert app.title.cget('text') == page

        class ExistingOutput:
            def command(self, _action):
                raise FileExistsError('output already exists')

        app.experiment = ExistingOutput()
        app._update_buttons()
        assert all(button.instate(['disabled']) for button in app.step_buttons.values())
    finally:
        app.destroy()
