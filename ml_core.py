from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import joblib


# 传统机器学习人脸识别器：StandardScaler → PCA → LinearSVM，含质心距离辅助判断
class ClassicFaceRecognizer:
    def __init__(self):
        self.scaler    = StandardScaler()
        self.pca       = None
        # 线性核 SVM：小样本高维场景下泛化能力优于 RBF
        self.svm       = SVC(kernel='linear', C=1.0, probability=True,
                             decision_function_shape='ovo')
        self.centroids = {}
        self.is_trained    = False
        # 关闭概率/距离硬拒识（阈值为 0 表示不拒）
        self.prob_thr  = 0.0
        self.dist_thr  = 0.0

    # 训练：标准化 → PCA → SVM → 计算类质心与距离阈值，返回统计字典
    def train(self, X, y):
        y = np.array(y)
        X_s = self.scaler.fit_transform(X)

        n, d = X_s.shape
        k = max(1, min(n - 1, d, 350))
        self.pca = PCA(n_components=k, svd_solver='randomized', whiten=True)
        X_p = self.pca.fit_transform(X_s)

        self.svm.fit(X_p, y)

        # 计算各类质心及类内最大距离，用于距离阈值
        classes = np.unique(y)
        max_dists = []
        self.centroids = {}
        for cls in classes:
            samples = X_p[y == cls]
            c = np.mean(samples, axis=0)
            self.centroids[cls] = c
            if len(samples) > 1:
                max_dists.append(float(np.max(euclidean_distances(samples, [c]))))

        if max_dists:
            self.dist_thr = float(np.mean(max_dists)) * 3.5
        elif len(self.centroids) > 1:
            cs = list(self.centroids.values())
            dm = euclidean_distances(cs)
            tri = dm[np.triu_indices_from(dm, k=1)]
            self.dist_thr = float(np.median(tri)) * 0.5 if len(tri) > 0 else 0.0
        else:
            self.dist_thr = 0.0

        self.is_trained = True
        return {
            'n_samples':  n,
            'n_classes':  int(len(classes)),
            'n_pca':      self.pca.n_components_,
            'n_lda':      0,
            'feat_dim':   int(X.shape[1]),
            'prob_thr':   self.prob_thr,
            'dist_thr':   round(self.dist_thr, 4),
        }

    # 预测：返回含 status/class/prob/dist/top3 的结果字典
    def predict(self, vec):
        X_s = self.scaler.transform([vec])
        X_p = self.pca.transform(X_s)

        probs = self.svm.predict_proba(X_p)[0]
        idx   = int(np.argmax(probs))
        prob  = float(probs[idx])
        cls   = self.svm.classes_[idx]

        # 质心不存在时回退到最近质心
        if cls not in self.centroids:
            cls = min(self.centroids,
                      key=lambda c: float(euclidean_distances(X_p, [self.centroids[c]])[0][0]))
        dist = float(euclidean_distances(X_p, [self.centroids[cls]])[0][0])

        top3 = [(self.svm.classes_[i], round(float(probs[i]), 4))
                for i in np.argsort(probs)[::-1][:3]]

        info = {
            'prob': prob, 'dist': dist,
            'prob_thr': self.prob_thr, 'dist_thr': self.dist_thr,
            'class': cls, 'top3': top3,
        }

        if prob < self.prob_thr:
            return {**info, 'status': 'rejected_prob'}
        elif dist > self.dist_thr > 0:
            return {**info, 'status': 'rejected_dist'}
        else:
            return {**info, 'status': 'accepted'}

    # 保存模型到磁盘
    def save(self, path):
        joblib.dump({
            'scaler': self.scaler, 'pca': self.pca, 'svm': self.svm,
            'centroids': self.centroids,
            'prob_thr': self.prob_thr, 'dist_thr': self.dist_thr,
        }, path)

    # 从磁盘加载模型
    def load(self, path):
        d = joblib.load(path)
        self.scaler, self.pca, self.svm = d['scaler'], d['pca'], d['svm']
        self.centroids = d['centroids']
        self.prob_thr  = d['prob_thr']
        self.dist_thr  = d['dist_thr']
        self.is_trained = True