import os

import cv2
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from features import extract_features


# 假设你有：图片路径 + 标签（1=清晰，0=不清晰）
def load_dataset(image_paths, labels):
    X = []
    y = []

    for img_path, label in zip(image_paths, labels):
        image = cv2.imread(img_path)
        features = extract_features(image)
        X.append(features)
        y.append(label)

    return np.array(X), np.array(y)


def train_model(image_paths, labels, model_path="GradientBoostingClassifier_best_new.pkl"):
    X, y = load_dataset(image_paths, labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3
    )

    model.fit(X_train, y_train)

    print("Train Acc:", model.score(X_train, y_train))
    print("Test  Acc:", model.score(X_test, y_test))

    joblib.dump(model, model_path)
    print("模型已保存：", model_path)


def load_all_samples(root_dir="dataset"):
    image_paths = []
    labels = []

    # 清晰图（1）
    clear_dir = os.path.join(root_dir, "clear")
    if os.path.exists(clear_dir):
        for f in os.listdir(clear_dir):
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                image_paths.append(os.path.join(clear_dir, f))
                labels.append(1)

    # 模糊图（0）
    blur_dir = os.path.join(root_dir, "blur")
    if os.path.exists(blur_dir):
        for f in os.listdir(blur_dir):
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                image_paths.append(os.path.join(blur_dir, f))
                labels.append(0)

    return image_paths, labels


if __name__ == "__main__":
    sample_images, sample_labels = load_all_samples("dataset")

    print("样本数量：", len(sample_images))
    print("清晰样本：", sample_labels.count(1))
    print("模糊样本：", sample_labels.count(0))

    train_model(sample_images, sample_labels)
