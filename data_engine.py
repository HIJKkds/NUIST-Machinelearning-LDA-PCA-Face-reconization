import cv2
import numpy as np
import os
import random


# 人脸预处理器：Haar 检测 → CLAHE 均衡 → HOG+多尺度LBP 特征提取
class FacePreprocessor:
    def __init__(self, size=(96, 96)):
        cascade = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.detector = cv2.CascadeClassifier(cascade)
        self.size = size  # (w, h)

    # Haar 人脸检测，返回最大脸框或 None
    def _detect(self, gray, scale=1.05, neighbors=4):
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=scale, minNeighbors=neighbors, minSize=(20, 20)
        )
        if len(faces) == 0:
            return None
        return max(faces, key=lambda r: r[2] * r[3])

    # 生成增广变体：仅水平翻转
    def _augment(self, face):
        flipped = cv2.flip(face, 1)
        return [flipped]

    # HOG + 双尺度 LBP 特征拼接，返回 1-D 向量
    def _features(self, face):
        w, h = self.size
        hog = cv2.HOGDescriptor(
            (w, h),
            (w // 6, h // 6),   # block: 16×16
            (w // 12, h // 12), # stride: 8×8
            (w // 12, h // 12), # cell: 8×8
            9
        )
        hog_feat = hog.compute(face).flatten()
        lbp1 = self._lbp(face, r=1)
        lbp2 = self._lbp(face, r=2)
        return np.concatenate([hog_feat, lbp1, lbp2])

    # 均匀 LBP 直方图（纯 numpy，支持指定半径）
    def _lbp(self, img, r=1):
        h, w = img.shape
        center = img[r:-r, r:-r].astype(np.int16)
        offsets = [(0,r), (-r,r), (-r,0), (-r,-r), (0,-r), (r,-r), (r,0), (r,r)]
        lbp = np.zeros((h - 2*r, w - 2*r), dtype=np.uint8)
        for bit, (dy, dx) in enumerate(offsets):
            nb = img[r+dy: r+dy+h-2*r, r+dx: r+dx+w-2*r].astype(np.int16)
            lbp |= ((nb >= center).astype(np.uint8) << bit)
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-7)
        return hist

    # 主入口：读取图片路径或 BGR 数组，返回特征向量（augment=True 返回列表）
    def extract(self, src, augment=False):
        if isinstance(src, str):
            img = cv2.imdecode(np.fromfile(src, dtype=np.uint8), cv2.IMREAD_COLOR)
        else:
            img = src
        if img is None or self.detector.empty():
            return [] if augment else None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        rect = self._detect(gray, scale=1.05, neighbors=4)
        if rect is None:
            rect = self._detect(gray, scale=1.03, neighbors=2)

        if rect is None:
            ih, iw = gray.shape
            x, y = int(iw * 0.2), int(ih * 0.2)
            w, h = int(iw * 0.6), int(ih * 0.6)
        else:
            x, y, w, h = rect

        roi = gray[y:y+h, x:x+w]
        roi = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(roi)
        face = cv2.resize(roi, self.size)

        primary = self._features(face)
        if not augment:
            return primary

        return [primary] + [self._features(a) for a in self._augment(face)]


# 数据集加载器：遍历 gt_db 目录，切分训练/测试集（训练集带增广）
class DatasetLoader:
    def __init__(self, root='./gt_db/'):
        self.root = root

    # 列出目录下所有 jpg 文件（已排序）
    def _list_imgs(self, cls_dir):
        return sorted(f for f in os.listdir(cls_dir) if f.lower().endswith('.jpg'))

    # 列出所有类别子目录
    def _list_classes(self):
        return sorted(
            d for d in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, d))
        )

    # 加载数据集，返回 (X_train, X_test, y_train, y_test)
    def load(self, preprocessor, n_classes=None, n_train=10):
        X_train, y_train = [], []
        X_test,  y_test  = [], []

        classes = self._list_classes()
        if n_classes is not None:
            classes = classes[:n_classes]

        random.seed(42)

        for cls in classes:
            cls_dir = os.path.join(self.root, cls)
            imgs    = self._list_imgs(cls_dir)
            random.shuffle(imgs)

            # 预扫：过滤出能成功提取特征的图片
            valid = []
            for name in imgs:
                fv = preprocessor.extract(os.path.join(cls_dir, name))
                if fv is not None:
                    valid.append(name)

            if not valid:
                continue

            k = min(n_train, len(valid))
            train_imgs = valid[:k]
            test_imgs  = valid[k:]

            # 训练集：带增广（每张生成2个向量: 原图+翻转）
            for name in train_imgs:
                for fv in preprocessor.extract(os.path.join(cls_dir, name), augment=True):
                    X_train.append(fv)
                    y_train.append(cls)

            # 测试集：原图
            for name in test_imgs:
                fv = preprocessor.extract(os.path.join(cls_dir, name))
                if fv is not None:
                    X_test.append(fv)
                    y_test.append(cls)

        return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)