import argparse
import joblib
import numpy as np

from features import extract_features
from utils import load_image


def predict_single(model, image_url):
    image = load_image(image_url)
    feat = extract_features(image)
    proba = model.predict_proba([feat])[0][1]  # 清晰概率
    pred = model.predict([feat])[0]

    return {
        "image": image_url,
        "clarity_score": float(proba),
        "isClear": int(pred)
    }


def predict_batch(model, file_list):
    return [predict_single(model, img) for img in file_list]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_url", help="图片链接或本地路径")
    parser.add_argument("--image_list", nargs="+", help="多张图片")
    parser.add_argument("--model_path", default="GradientBoostingClassifier_best_new.pkl")
    args = parser.parse_args()

    model = joblib.load(args.model_path)

    if args.image_url:
        result = predict_single(model, args.image_url)
        print(result)

    if args.image_list:
        result = predict_batch(model, args.image_list)
        print(result)
