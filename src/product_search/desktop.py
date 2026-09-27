"""点击式项目面板：Tk 主线程负责界面，后台进程复用原有 CLI。"""
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from urllib.request import ProxyHandler, build_opener
import webbrowser

from product_search.settings import load_settings
from product_search.workbench import Experiment, JobRunner, discover_resources, read_object, report_rows


class WorkbenchApp:
    def __init__(self, window, project_root):
        self.window, self.root = window, Path(project_root).resolve()
        self.settings = load_settings('configs/dev.toml', project_root=self.root)
        self.python = self.root / '.venv' / 'Scripts' / 'python.exe'
        self.url = f'http://127.0.0.1:{self.settings.service.port}'
        self.job_events, self.server_events, self.ui_events = queue.Queue(), queue.Queue(), queue.Queue()
        self.job, self.server = JobRunner(self.root, self.job_events), JobRunner(self.root, self.server_events)
        self.experiment = None
        self.active_action = None
        self.launching = False
        self.closing = False
        self.closed = False
        self.after_id = None
        self.release_var, self.report_var, self.dataset_var = tk.StringVar(), tk.StringVar(), tk.StringVar()
        self.status = tk.StringVar(value='准备好了。先打开搜索体验，再到结果解读页看真实指标。')
        self.server_status = tk.StringVar(value='搜索服务尚未由面板启动')
        self.experiment_status = tk.StringVar(value='选择已有 dev 数据集，然后点击“开始新一轮”。')
        self._build_layout()
        self.refresh_resources()
        self.show_page('体验搜索')
        self.window.protocol('WM_DELETE_WINDOW', self.close)
        self.after_id = self.window.after(100, self._poll)

    def _build_layout(self):
        w = self.window
        w.title('商品搜索 · 项目学习面板')
        w.geometry('1160x840')
        w.minsize(980, 720)
        w.configure(bg='#f4f7fa')
        style = ttk.Style(w)
        style.theme_use('clam')
        style.configure('.', font=('Microsoft YaHei UI', 10))
        style.configure('TFrame', background='#f4f7fa')
        style.configure('TLabel', background='#f4f7fa', foreground='#233d4d')
        style.configure('TButton', padding=(12, 8))
        style.configure('Primary.TButton', background='#136c63', foreground='white')
        style.map('Primary.TButton', background=[('active', '#0e574f'), ('disabled', '#b6c8c6')])
        style.configure('Treeview', rowheight=34, background='white', fieldbackground='white')
        rail = tk.Frame(w, bg='#153d4d', width=196)
        rail.pack(side='left', fill='y')
        rail.pack_propagate(False)
        tk.Label(rail, text='商品搜索\n从体验到理解', bg='#153d4d', fg='white', justify='left',
                 font=('Microsoft YaHei UI', 19, 'bold')).pack(anchor='w', padx=18, pady=(30, 24))
        self.pages = {}
        self.nav_buttons = {}
        names = ['体验搜索', '结果解读', '重做实验', '学习路线']
        for i, name in enumerate(names, 1):
            btn = tk.Button(rail, text=f'{i}   {name}', anchor='w', relief='flat', bd=0,
                            bg='#153d4d', fg='white', activebackground='#24566a', activeforeground='white',
                            padx=18, pady=14, font=('Microsoft YaHei UI', 11), command=lambda n=name: self.show_page(n))
            btn.pack(fill='x', pady=2)
            self.nav_buttons[name] = btn
        tk.Label(rail, text='本地 CPU · 公开 ESCI\n无需 Dify / 聊天 API\n真实报告，不编造提升', bg='#153d4d', fg='#bdd4dd',
                 justify='left', font=('Microsoft YaHei UI', 10)).pack(side='bottom', anchor='w', padx=18, pady=24)
        main = ttk.Frame(w, padding=22)
        main.pack(side='left', fill='both', expand=True)
        self.title = ttk.Label(main, font=('Microsoft YaHei UI', 22, 'bold'))
        self.title.pack(anchor='w', pady=(0, 14))
        self.body = ttk.Frame(main)
        self.body.pack(fill='both', expand=True)
        for name in names:
            self.pages[name] = ttk.Frame(self.body)
        self._build_search_page(self.pages['体验搜索'])
        self._build_report_page(self.pages['结果解读'])
        self._build_experiment_page(self.pages['重做实验'])
        self._build_learning_page(self.pages['学习路线'])
        ttk.Separator(main).pack(fill='x', pady=(12, 8))
        ttk.Label(main, textvariable=self.status, wraplength=820).pack(anchor='w')
        details = ttk.Frame(main)
        details.pack(fill='x', pady=(6, 4))
        self.progress = ttk.Progressbar(details, mode='indeterminate')
        self.progress.pack(side='left', fill='x', expand=True, padx=(0, 12))
        ttk.Button(details, text='打开日志目录', command=lambda: self.open_folder(self.root / 'logs')).pack(side='right')
        self.log = tk.Text(main, height=6, wrap='word', bg='#18313e', fg='#dfebf2',
                           font=('Consolas', 9), state='disabled', relief='flat')
        self.log.pack(fill='x')

    def _paragraph(self, frame, text, *, bold=False):
        label = ttk.Label(frame, text=text, wraplength=810, justify='left',
                          font=('Microsoft YaHei UI', 11, 'bold' if bold else 'normal'))
        label.pack(anchor='w', fill='x', pady=(0, 12))
        return label

    def _text_area(self, frame, height=12):
        container = ttk.Frame(frame)
        container.pack(fill='both', expand=True)
        text = tk.Text(container, height=height, wrap='word', bg='white', fg='#233d4d',
                       font=('Microsoft YaHei UI', 11), padx=16, pady=12, relief='flat', state='disabled')
        bar = ttk.Scrollbar(container, command=text.yview)
        text.configure(yscrollcommand=bar.set)
        bar.pack(side='right', fill='y')
        text.pack(side='left', fill='both', expand=True)
        return text

    @staticmethod
    def _set_text(widget, text):
        widget.configure(state='normal')
        widget.delete('1.0', 'end')
        widget.insert('end', text)
        widget.configure(state='disabled')

    def _build_search_page(self, frame):
        self._paragraph(frame, '你现在是一名顾客：想买跑鞋，系统能不能把合适的商品放在前面？', bold=True)
        self._paragraph(frame, '第一次直接用已有模型。点击下面按钮后，面板会检查并启动服务，再打开搜索网页。')
        ttk.Label(frame, text='本次使用的版本').pack(anchor='w')
        self.release_combo = ttk.Combobox(frame, textvariable=self.release_var, state='readonly', width=72)
        self.release_combo.pack(anchor='w', fill='x', pady=(6, 12))
        actions = ttk.Frame(frame)
        actions.pack(fill='x', pady=(0, 14))
        self.launch_button = ttk.Button(actions, text='打开搜索体验', style='Primary.TButton', command=self.launch_search)
        self.launch_button.pack(side='left')
        ttk.Button(actions, text='停止面板启动的服务', command=self.stop_server).pack(side='left', padx=8)
        ttk.Button(actions, text='刷新已有版本', command=self.refresh_resources).pack(side='left')
        ttk.Label(frame, textvariable=self.server_status, wraplength=800).pack(anchor='w', pady=(0, 16))
        box = self._text_area(frame)
        self._set_text(box,
            '第 1 次：搜索 running shoes\n看看前 5 个商品，是否真的是跑鞋？是否有童款、配件或其他鞋类混进来？\n\n'
            '第 2 次：搜索 red running shoes\n增加颜色要求，观察前几名和两路命中名次是否改变。没有改变也值得分析，不要预设系统一定理解了颜色。\n\n'
            '第 3 次：换成 wireless headphones\n留意商品是否只来自 BM25 或 dense。只显示一路说明另一条路没把它召回进 top-k，并不等于分数为零。\n\n'
            '看懂页面\n“bm25 #…”是关键词召回排名；“dense #…”是语义召回排名；最终分数用于排序，不是购买概率。\n\n'
            '比较两种流程\n选择“仅融合”版本可体验 RRF；选择“学习排序”版本可体验 LTR。切换前先停止本面板启动的服务。\n\n'
            '完成标准\n你能指出一个符合需求的结果、一个值得怀疑的结果，并说明它们来自哪路召回。下一步去“结果解读”。')

    def _build_report_page(self, frame):
        self._paragraph(frame, '凭一两次搜索不能判断整体质量。这里把已有验证报告翻译成表格。')
        chooser = ttk.Frame(frame)
        chooser.pack(fill='x', pady=(0, 10))
        self.report_combo = ttk.Combobox(chooser, textvariable=self.report_var, state='readonly')
        self.report_combo.pack(side='left', fill='x', expand=True)
        self.report_combo.bind('<<ComboboxSelected>>', lambda _: self.load_report())
        ttk.Button(chooser, text='打开报告目录', command=self.open_report_folder).pack(side='left', padx=(8, 0))
        self.report_meta = tk.StringVar()
        ttk.Label(frame, textvariable=self.report_meta, wraplength=820).pack(anchor='w', pady=(0, 8))
        cols = ('method', 'ndcg', 'mrr', 'recall', 'judged')
        self.report_table = ttk.Treeview(frame, columns=cols, show='headings', height=4, selectmode='browse')
        titles = ['方法', '排序质量', '首个精确匹配', '已知精确召回', '标注覆盖率']
        for col, title in zip(cols, titles):
            self.report_table.heading(col, text=title)
            self.report_table.column(col, width=140, anchor='center')
        self.report_table.pack(fill='x', pady=(0, 12))
        self.report_detail = self._text_area(frame)

    def _build_experiment_page(self, frame):
        self._paragraph(frame, '这次换你做工程师：把商品建成索引，训练排序器，再用相同验证集检查效果。', bold=True)
        self._paragraph(frame, '先停止搜索服务，减少资源争用。每轮使用新目录，原有模型与报告保留。这里只运行 dev / validation。')
        line = ttk.Frame(frame)
        line.pack(fill='x', pady=(0, 10))
        self.dataset_combo = ttk.Combobox(line, textvariable=self.dataset_var, state='readonly')
        self.dataset_combo.pack(side='left', fill='x', expand=True)
        self.new_button = ttk.Button(line, text='开始新一轮', command=self.new_experiment)
        self.new_button.pack(side='left', padx=(8, 0))
        ttk.Label(frame, textvariable=self.experiment_status, wraplength=820).pack(anchor='w', pady=(0, 12))
        self.step_buttons = {}
        stages = [('build', '1  建立索引', '让商品变成可搜索的关键词倒排和语义向量。'),
                  ('train', '2  训练排序器', '用人工标注学习候选的先后顺序；不是微调 MiniLM。'),
                  ('evaluate', '3  验证效果', '在相同 validation 查询上比较四种方法；不使用 test。')]
        for action, title, why in stages:
            row = ttk.Frame(frame)
            row.pack(fill='x', pady=6)
            btn = ttk.Button(row, text=title, command=lambda a=action: self.run_step(a))
            btn.pack(side='left', padx=(0, 16))
            self.step_buttons[action] = btn
            ttk.Label(row, text=why, wraplength=530).pack(side='left')
        actions = ttk.Frame(frame)
        actions.pack(fill='x', pady=12)
        ttk.Button(actions, text='停止当前实验', command=self.stop_job).pack(side='left')
        ttk.Button(actions, text='查看本轮结果', command=self.show_experiment_report).pack(side='left', padx=8)
        ttk.Button(actions, text='检查环境与测试', command=self.run_tests).pack(side='left')
        self._paragraph(frame,
            '每一步结束后再点下一步。失败时先看下方日志；不要通过修改 release.json 冒充成功。'
            '停止任务会保留已写文件；如果本轮输出目录已被占用，排查后再开始新一轮。')

    def _build_learning_page(self, frame):
        self._paragraph(frame, '把“我运行过代码”变成“我知道系统怎样工作”。', bold=True)
        text = self._text_area(frame)
        self._set_text(text,
            '第一关 · 顾客体验\n目的：知道输入、输出和系统边界。\n操作：在体验搜索页输入三个需求，指出好结果和坏结果。\n结论：单次搜索是案例，不能代表整体效果。\n\n'
            '第二关 · 拆开召回与排序\n目的：分清没找到与排错了。\n操作：看 BM25/dense 的命中名次，再看最终名次。\n结论：相关商品没进入候选时，只改排序器救不了召回。\n\n'
            '第三关 · 读一张真实成绩单\n目的：知道每个指标回答什么。\n操作：打开结果解读，看排序质量、已知精确召回和置信区间。\n结论：当前 dev 上 LTR 没有超过 dense；不能把“用了模型”写成“效果提升”。\n\n'
            '第四关 · 自己完成一轮实验\n目的：知道各阶段输入和产物。\n操作：在重做实验页依次建立索引、训练、验证。\n结论：商品/向量/模型必须来自兼容版本，不能随意混用。\n\n'
            '第五关 · 回到源码\n目的：能解释每个模块为何存在。\n阅读：schemas → 数据处理 → 召回 → 特征/排序 → SearchEngine → 评估/API。\n'
            '先说清输入、处理、输出，再研究语法。完整文件构建顺序见 docs/FILE-GUIDE.md。\n\n'
            '第六关 · 进阶实验\n目标：每次只改变一个因素，例如删除 dense_score 排序特征。\n'
            '此时才使用 docs/COMMANDS.md 的命令。test 冻结、完整 US 数据和 FAISS 对比不由此面板自动运行。')

    def show_page(self, name):
        for page in self.pages.values():
            page.pack_forget()
        self.pages[name].pack(fill='both', expand=True)
        self.title.configure(text=name)
        for title, button in self.nav_buttons.items():
            button.configure(bg='#276277' if title == name else '#153d4d')

    def refresh_resources(self):
        self.resources = discover_resources(self.root)
        self.release_paths = {}
        for path in self.resources['releases']:
            data = read_object(path)
            kind = '学习排序' if 'ranker.txt' in data['files'] else '仅融合'
            self.release_paths[f'{kind}  ·  {path.parent.name}'] = path.parent
        self.report_paths = {p.parent.name: p for p in self.resources['reports']}
        self.dataset_paths = {p.parent.name: p.parent for p in self.resources['datasets']}
        for combo, variable, values in [(self.release_combo, self.release_var, self.release_paths),
                                         (self.report_combo, self.report_var, self.report_paths),
                                         (self.dataset_combo, self.dataset_var, self.dataset_paths)]:
            combo.configure(values=list(values))
            if variable.get() not in values:
                variable.set(next(iter(values), ''))
        self.load_report()
        self._update_buttons()

    def load_report(self):
        path = self.report_paths.get(self.report_var.get())
        self.report_table.delete(*self.report_table.get_children())
        if path is None:
            self.report_meta.set('还没有验证报告。完成一轮实验后会在这里出现。')
            self._set_text(self.report_detail, '现在可以先体验已有搜索，再进行实验。')
            return
        try:
            report = read_object(path)
            rk, ck = report.get('ranking_k', 10), report.get('retrieval_k', 100)
            self.report_table.heading('ndcg', text=f'排序质量 NDCG@{rk}')
            self.report_table.heading('mrr', text=f'首个精确 MRR@{rk}')
            self.report_table.heading('recall', text=f'已知 E 召回@{ck}')
            self.report_table.heading('judged', text=f'标注覆盖@{ck}')
            for row in report_rows(report):
                self.report_table.insert('', 'end', values=row)
            self.report_meta.set(f"验证查询：{report.get('queries', '?')}  ·  数据：{report.get('dataset_id', '?')}  ·  {report.get('partition', '?')}")
            detail = ('怎么读这张表\n前两列只比较人工给定候选内的排序；后两列是从整个商品库召回。'
                      '“—”表示不适用或未定义，不是零分。未标注商品不是不相关。\n\n')
            scores = {k:v.get('ndcg', {}).get('mean') for k,v in report.get('ranking', {}).items()}
            valid = {k:v for k,v in scores.items() if isinstance(v, (int, float))}
            if valid:
                best = max(valid, key=valid.get)
                detail += f'本次结果\n人工候选 NDCG 均值最高的方法：{best}（{valid[best]:.4f}）。\n'
            comparison = report.get('comparisons', {}).get('ltr_minus_bm25', {})
            ci = comparison.get('ci95', [])
            if len(ci) == 2 and all(isinstance(v, (float, int)) for v in ci):
                detail += f'LTR 相对 BM25 的 95% 差值区间：[{ci[0]:.4f}, {ci[1]:.4f}]。'
                detail += '区间跨 0，没有显著提升证据。\n' if ci[0] <= 0 <= ci[1] else '区间未跨 0，但仍需结合独立测试确认。\n'
            detail += 'validation 同时用于早停；这是开发诊断，不能当作线上效果。\n\n'
            errors = path.parent / 'error-analysis.jsonl'
            if errors.exists():
                detail += '真实错误案例（人工候选评估，不等于网页上的全库结果）\n'
                with errors.open(encoding='utf-8') as stream:
                    for _ in range(3):
                        line = stream.readline()
                        if not line:
                            break
                        case = json.loads(line)
                        delta = case.get('deltas', {}).get('ltr_minus_bm25', {}).get('ndcg')
                        value = f'{delta:+.4f}' if isinstance(delta, (float, int)) else '未定义'
                        detail += f"• {case.get('text', '')}：LTR 相对 BM25 的 NDCG 差值 {value}\n"
            detail += '\n思考：提升发生在召回还是排序？如果答案说不清，先不要写“模型提升了效果”。'
            self._set_text(self.report_detail, detail)
        except (OSError, ValueError, TypeError) as exc:
            self._set_text(self.report_detail, f'报告读取失败：{exc}')

    def open_folder(self, path):
        path = Path(path)
        if not path.exists():
            messagebox.showinfo('尚未生成', '这个目录还没有生成，完成对应操作后再查看。', parent=self.window)
        else:
            os.startfile(str(path))

    def open_report_folder(self):
        path = self.report_paths.get(self.report_var.get())
        if path:
            self.open_folder(path.parent)

    def _cli(self, args):
        return [str(self.python), '-u', '-m', 'product_search', *args]

    def launch_search(self):
        if self.launching or self.closing:
            return
        release = self.release_paths.get(self.release_var.get())
        if release is None:
            messagebox.showinfo('没有发布包', '先在重做实验页建立索引和模型。', parent=self.window)
            return
        if self.settings.service.host != '127.0.0.1':
            messagebox.showerror('配置不匹配', '点击式面板只连接本机 127.0.0.1，请恢复 dev 服务地址。', parent=self.window)
            return
        self.launching = True
        self.launch_button.configure(state='disabled')
        self.server_status.set('正在核对版本并加载本地模型，请稍等…')
        threading.Thread(target=self._launch_worker, args=(release,), daemon=True).start()

    def _probe(self):
        with build_opener(ProxyHandler({})).open(self.url + '/version', timeout=.8) as response:
            return json.load(response)

    def _launch_worker(self, release):
        owned = False
        try:
            expected = read_object(release / 'release.json')['release_id']
            try:
                live = self._probe()
            except (OSError, ValueError):
                live = None
            if live is not None:
                if live.get('release_id') != expected:
                    raise RuntimeError('此端口正在运行另一个版本。请先停止原服务，再打开所选版本。')
                self.ui_events.put(('ready', '已连接正在运行的同版本服务；面板不会接管外部启动的进程。'))
                return
            if self.server.busy:
                raise RuntimeError('面板已有一个服务进程；请等待启动或先停止它。')
            if self.closing:
                self.ui_events.put(('launch_error', '面板正在关闭，已取消启动。'))
                return
            self.server.start(self._cli(['serve', '--project-root', str(self.root), '--release', str(release)]), label='加载搜索服务')
            owned = True
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline and not self.closing:
                if not self.server.busy:
                    raise RuntimeError('服务没有启动成功，请查看下方日志（常见原因：端口占用或模型缓存缺失）。')
                try:
                    live = self._probe()
                    if live.get('release_id') == expected:
                        self.ui_events.put(('ready', '搜索已就绪；浏览器已打开，面板保持运行即可。'))
                        return
                except (OSError, ValueError):
                    pass
                time.sleep(.4)
            raise RuntimeError('启动已取消或等待超时；查看日志后重试。')
        except Exception as exc:
            if owned:
                self.server.stop()
            self.ui_events.put(('launch_error', str(exc)))

    def stop_server(self):
        if self.server.busy:
            self.server.stop()
            self.server_status.set('正在停止本面板启动的搜索服务…')
        else:
            self.server_status.set('没有本面板拥有的服务进程。外部终端启动的服务，请在原终端停止。')

    def new_experiment(self):
        if self.job.busy:
            return
        dataset = self.dataset_paths.get(self.dataset_var.get())
        if dataset is None:
            messagebox.showinfo('没有 dev 数据', '请先按进阶说明准备 dev 数据集。', parent=self.window)
            return
        self.experiment = Experiment.create(self.root, dataset)
        self.experiment_status.set(f'本轮：{self.experiment.name}。已准备好，请先建立索引。')
        self._update_buttons()

    def run_step(self, action):
        if self.experiment is None:
            return
        if self.server.busy or self.launching:
            messagebox.showinfo('先停止搜索', '请到体验搜索页停止本面板的服务，再运行实验。', parent=self.window)
            return
        try:
            command = self.experiment.command(action)
            self.active_action = action
            self.job.start(self._cli(command), label={'build':'建立索引', 'train':'训练排序器', 'evaluate':'验证效果'}[action])
            self.status.set('任务正在后台运行。可以切换页面，下方显示进度；请勿重复点击。')
            self._update_buttons()
        except (OSError, ValueError, RuntimeError) as exc:
            messagebox.showerror('未启动任务', str(exc), parent=self.window)

    def run_tests(self):
        if self.job.busy:
            return
        try:
            self.active_action = 'tests'
            self.job.start([str(self.python), '-u', '-m', 'pytest', '-q'], label='检查环境与自动化测试')
            self.status.set('正在检查环境与自动化测试…')
            self._update_buttons()
        except OSError as exc:
            messagebox.showerror('启动失败', str(exc), parent=self.window)

    def stop_job(self):
        if self.job.busy and messagebox.askyesno('停止当前实验', '停止后保留日志和已写产物，未完成目录不能当作成功结果。现在停止？', parent=self.window):
            self.job.stop()

    def show_experiment_report(self):
        if self.experiment and (self.experiment.report_dir / 'ranking-results.json').exists():
            self.refresh_resources()
            self.report_var.set(self.experiment.report_dir.name)
            self.load_report()
            self.show_page('结果解读')
        else:
            messagebox.showinfo('尚无结果', '先完成本轮的第 3 步“验证效果”。', parent=self.window)

    def _update_buttons(self):
        busy = self.job.busy
        self.new_button.configure(state='disabled' if busy else 'normal')
        for action, button in self.step_buttons.items():
            enabled = self.experiment is not None and not busy
            if enabled:
                try:
                    self.experiment.command(action)
                except (ValueError, OSError, FileExistsError):
                    enabled = False
            button.configure(state='normal' if enabled else 'disabled')

    def _append_log(self, text):
        self.log.configure(state='normal')
        self.log.insert('end', text)
        if int(self.log.index('end-1c').split('.')[0]) > 400:
            self.log.delete('1.0', '100.0')
        self.log.see('end')
        self.log.configure(state='disabled')

    def _poll(self):
        if self.closed:
            return
        for events, is_job in ((self.job_events, True), (self.server_events, False)):
            for _ in range(150):
                try:
                    kind, value = events.get_nowait()
                except queue.Empty:
                    break
                if kind == 'log':
                    self._append_log(value)
                elif kind == 'done' and is_job:
                    self.status.set('任务完成。可以查看结果或进入下一步。' if value == 0 else f'任务未完成（退出码 {value}）。日志已保留，请先排查再开始新一轮。')
                    self.refresh_resources()
                    if value == 0 and self.active_action == 'evaluate':
                        self.show_experiment_report()
                elif kind == 'done':
                    self.server_status.set('本面板启动的搜索服务已结束。')
        while True:
            try:
                kind, value = self.ui_events.get_nowait()
            except queue.Empty:
                break
            self.launching = False
            self.launch_button.configure(state='normal')
            self.server_status.set(value)
            if kind == 'ready' and not self.closing:
                webbrowser.open(self.url)
            elif kind == 'launch_error' and not self.closing:
                messagebox.showerror('搜索尚未就绪', value, parent=self.window)
        if self.job.busy or self.launching:
            if not getattr(self, '_animating', False):
                self.progress.start(12)
                self._animating = True
        else:
            self.progress.stop()
            self._animating = False
        if self.closing and not self.job.busy and not self.server.busy and not self.launching:
            self.destroy()
            return
        self.after_id = self.window.after(100, self._poll)

    def close(self):
        if self.job.busy and not messagebox.askyesno('实验仍在运行', '关闭面板会停止本面板的实验和搜索服务，并保留日志。确定关闭？', parent=self.window):
            return
        self.closing = True
        self.job.stop()
        self.server.stop()
        self.status.set('正在结束本面板拥有的任务…')
        if not self.job.busy and not self.server.busy and not self.launching:
            self.destroy()

    def destroy(self):
        self.closed = True
        if self.after_id:
            self.window.after_cancel(self.after_id)
        self.window.destroy()


def main():
    root = Path(__file__).resolve().parents[2]
    window = tk.Tk()
    try:
        WorkbenchApp(window, root)
    except Exception as exc:
        window.withdraw()
        messagebox.showerror('项目面板无法打开', f'{exc}\n\n请确认项目 .venv、configs/dev.toml 和源码完整。')
        window.destroy()
        return
    window.mainloop()


if __name__ == '__main__':
    main()
