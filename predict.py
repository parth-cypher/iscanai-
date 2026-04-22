import argparse

from model_utils import predict_external_cataract, resolve_default_image_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict visible cataract from an external eye photo.")
    parser.add_argument("image_path", nargs="?", default=None, help="Optional image path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = resolve_default_image_path(args.image_path)
    result = predict_external_cataract(image_path)

    print("\nPrediction Result")
    print(f"Image: {result['image_path']}")
    print(f"AI Prediction: {result['ai_prediction']}")
    print(f"Visual Analysis: {result['visual_prediction']}")
    print(f"Final Result: {result['final_result']}")
    print(f"Confidence: {result['confidence']:.2f}%")
    print(f"ML Confidence: {result['ml_confidence']:.2f}%")
    print(f"Brightness Score: {result['brightness_score']:.2f}")
    print(f"Normal Probability: {result['normal_probability']:.2f}%")
    print(f"Cataract Probability: {result['cataract_probability']:.2f}%")
    if result["warning"]:
        print(f"Warning: {result['warning']}")


if __name__ == "__main__":
    main()
