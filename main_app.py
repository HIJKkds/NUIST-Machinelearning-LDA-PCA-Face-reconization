import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import threading
import numpy as np
from PIL import Image, ImageTk

from data_engine import FacePreprocessor, DatasetLoader
from ml_core import ClassicFaceRecognizer


# GUI 应用：基于传统 ML 的人脸识别实验平台
class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("传统机器学习人脸识别实验平台")
        self.root.geometry("900x700")

        self.preprocessor = FacePreprocessor(size=(96, 96))
        self.loader       = DatasetLoader()
        self.model        = ClassicFaceRecognizer()
        
        self.X_test = None
        self.y_test = None
        self.X_train = None
        self.y_train = None

        self._build_ui()
        self._log("系统初始化完成。准备就绪。")

    # 构建主界面：顶部控制区、中间图像预览区、底部日志区
    def _build_ui(self):
        top_frame = tk.Frame(self.root)
        
        params_frame = tk.LabelFrame(top_frame, text="参数配置")        
        top_frame.pack(fill=tk.X, padx=10, pady=5)
        params_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y)
        
        tk.Label(params_frame, text="数据集路径:").grid(row=0, column=0, padx=5, pady=2, sticky='e')
        self.ent_db = tk.Entry(params_frame, width=15)
        self.ent_db.insert(0, "./gt_db/")
        self.ent_db.grid(row=0, column=1, padx=5, pady=2)
        
        tk.Label(params_frame, text="每类训练数:").grid(row=1, column=0, padx=5, pady=2, sticky='e')
        self.ent_train_n = tk.Entry(params_frame, width=15)
        self.ent_train_n.insert(0, "10")
        self.ent_train_n.grid(row=1, column=1, padx=5, pady=2)
        
        btn_frame = tk.Frame(top_frame)
        btn_frame.pack(side=tk.LEFT, padx=20, fill=tk.Y)
        
        self.btn_train = tk.Button(btn_frame, text="训练模型", command=self.action_train, width=16)
        self.btn_train.grid(row=0, column=0, padx=5, pady=2)
        self.btn_eval  = tk.Button(btn_frame, text="批量评估", command=self.action_eval, width=16)
        self.btn_eval.grid(row=0, column=1, padx=5, pady=2)
        self.btn_test  = tk.Button(btn_frame, text="单张验证", command=self.action_test, width=16)
        self.btn_test.grid(row=0, column=2, padx=5, pady=2)
        self.btn_save  = tk.Button(btn_frame, text="保存模型", command=self.action_save, width=16)
        self.btn_save.grid(row=1, column=0, padx=5, pady=2)
        self.btn_load  = tk.Button(btn_frame, text="加载模型", command=self.action_load, width=16)
        self.btn_load.grid(row=1, column=1, padx=5, pady=2)
        
        self.canvas = tk.Canvas(self.root, width=300, height=300, bg='black')
        self.canvas.pack(pady=5)

        self.lbl_status = tk.Label(self.root, text="就绪", font=("微软雅黑", 11, "bold"), fg="blue")
        self.lbl_status.pack(pady=2)

        log_frame = tk.Frame(self.root)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, font=("Consolas", 9), bg="#f4f4f4")
        self.log_text.pack(fill=tk.BOTH, expand=True)

    # 打印线程安全日志
    def _log(self, msg):
        self.root.after(0, lambda: [self.log_text.insert(tk.END, msg + "\n"), self.log_text.see(tk.END)])

    # 更新 UI 状态提示标语
    def _status(self, msg, color="black"):
        self.root.after(0, lambda: self.lbl_status.config(text=msg, fg=color))

    # 冻结/解冻所有按钮状态
    def _state(self, is_active):
        s = tk.NORMAL if is_active else tk.DISABLED
        for b in (self.btn_train, self.btn_test, self.btn_eval, self.btn_save, self.btn_load):
            self.root.after(0, lambda btn=b: btn.config(state=s))

    # 保存计算模型参数
    def action_save(self):
        path = filedialog.asksaveasfilename(defaultextension=".pkl", filetypes=[("Model Files", "*.pkl")])
        if path:
            extra_data = {
                'X_train': self.X_train, 'y_train': self.y_train,
                'X_test': self.X_test, 'y_test': self.y_test
            }
            self.model.save(path, extra=extra_data)
            self._log(f"[*] 模型及数据集缓存已保存至: {path}")
            self._status("模型保存成功", "green")

    # 载入计算模型参数
    def action_load(self):
        path = filedialog.askopenfilename(filetypes=[("Model Files", "*.pkl")])
        if path:
            d = self.model.load(path)
            self.X_train = d.get('X_train')
            self.y_train = d.get('y_train')
            self.X_test  = d.get('X_test')
            self.y_test  = d.get('y_test')
            
            self._log(f"[*] 加载外部模型: {path}")
            self._log(f"    - 阈值参数: Prob={self.model.prob_thr}, Dist={self.model.dist_thr:.4f}")
            if self.X_test is not None and len(self.X_test) > 0:
                self._log(f"    - 附带训练集: {len(self.X_train) if self.X_train is not None else 0} 张 | 测试集: {len(self.X_test)} 张")
            else:
                self._log("    - 该模型未查找到附加的数据集缓存")
            self._status("模型加载成功", "green")

    # 后台训练主事件
    def action_train(self):
        try:
            n = int(self.ent_train_n.get().strip())
        except ValueError:
            return messagebox.showerror("错误", "每类训练数应为整数")
        
        self.loader.root = self.ent_db.get().strip()
        self._state(False)
        self.X_test = self.y_test = None
        self._status("读取图像并提取特征中...", "blue")
        threading.Thread(target=self._task_train, args=(n,), daemon=True).start()

    # 实际训练线程任务
    def _task_train(self, n):
        Xt, Xv, yt, yv = self.loader.load(self.preprocessor, n_train=n)
        if len(Xt) == 0:
            self._status("未找到有效数据", "red")
            self._log("[!] 训练集空，请检查路径。")
            return self._state(True)

        self._log(f"[*] 训练启动。缓存测试集 {len(Xv)} 张")
        self._status("重构分类平面...", "blue")
        
        res = self.model.train(Xt, yt)
        self.X_train, self.y_train = Xt, yt
        self.X_test, self.y_test   = Xv, yv

        self._log(f"[*] 训练完成! 总类别={res['n_classes']}, 增广样本数={res['n_samples']}")
        self._log(f"    - 特征维度: {res['feat_dim']} -> PCA {res['n_pca']}维")
        self._status("模型已可用于测试", "green")
        self._state(True)

    # 批量评估事件
    def action_eval(self):
        self._state(False)
        self._status("批量评估中...", "blue")
        threading.Thread(target=self._task_eval, daemon=True).start()

    # 实际批量评估线程
    def _task_eval(self):
        total, ok, bad = len(self.X_test), 0, 0
        r_prob, r_dist = 0, 0

        for i in range(total):
            res = self.model.predict(self.X_test[i])
            st  = res['status']
            if st == 'accepted':
                if res['class'] == self.y_test[i]:
                    ok += 1
                else:
                    bad += 1
            elif st == 'rejected_prob': r_prob += 1
            elif st == 'rejected_dist': r_dist += 1

        acc = (ok / total) * 100 if total else 0
        self._log(f"[*] 批量评估完毕: 共 {total} 张")
        self._log(f"    - 命中:{ok} | 误判:{bad} | Prob拒:{r_prob} | Dist拒:{r_dist}")
        self._log(f"    - 首选准确率: {acc:.2f}%")
        self._status(f"评估完成 - 准确率 {acc:.1f}%", "green" if acc >= 90 else "orange")
        self._state(True)

    # 单图验证事件
    def action_test(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg")])
        if not path:
            return

        img = Image.open(path).convert("RGB")
        img.thumbnail((300, 300))
        self.tk_image = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(150, 150, image=self.tk_image, anchor="center")

        self._status("特征提取分类中...", "blue")
        self._state(False)
        self._log(f"\n[*] 独图验证 => {path}")
        
        bgr = np.array(img)[:, :, ::-1].copy()
        threading.Thread(target=self._task_test, args=(bgr,), daemon=True).start()

    # 实际单图验证线程
    def _task_test(self, bgr):
        vec = self.preprocessor.extract(bgr)
        if vec is None:
            self._status("无法定位人脸", "red")
            self._log("    - 错误: 预处理拒绝入场")
            return self._state(True)

        res = self.model.predict(vec)
        st  = res['status']
        self._log("    - TOP3: " + " | ".join([f"{c}({p:.2f})" for c, p in res['top3']]))
        self._log(f"    - DEBUG : Prob={res['prob']:.4f}({res['prob_thr']}), Dist={res['dist']:.1f}({res['dist_thr']:.1f})")

        if st == 'accepted':
            self._status(f"判断属于: {res['class']}", "green")
        elif st == 'rejected_prob':
            self._status("被拒 (概率偏低)", "red")
        else:
            self._status("被拒 (偏离质心)", "red")
            
        self._state(True)


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()