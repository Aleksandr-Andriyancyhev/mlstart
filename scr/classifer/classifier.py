import argparse
from typing import Any
import torchvision.models as models
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from PIL import Image

# 3rdarty
import cv2

# project

def inference_classifier(classifier: Any, path_to_image) -> str:
    image = Image.open(path_to_image)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image)
    image_tensor = image.unsqueeze(0)

    with torch.no_grad():
        output = classifier(image_tensor)
        print(output)
        nul, predicted = torch.max(output, 1)
        print(nul,predicted)

    return "aircraft" if predicted.item() == 0 else "ship"


def load_classifier(name_of_classifier: str, path_to_pth_weights: str, device: str):
    state_dict = torch.load(path_to_pth_weights, map_location=device).state_dict()
    model = getattr(models, name_of_classifier)()

    model.fc = nn.Sequential(

    nn.Linear(model.fc.in_features, 500),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(500, 2)
) 
    model.load_state_dict(state_dict)
    model.eval()

    return model

def arguments_parser() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Скрипт для выполнения классификатора на единичном изображении или папке c изображениями")
    parser.add_argument(
        "--name_of_classifier", 
        "-nc", 
        type=str, 
        help="Название классификатора"
    )
    parser.add_argument(
        "--path_to_weights",
        "-wp",
        type=str,
        help="Путь к PTH-файлу c весами классификатора",
    )
    parser.add_argument(
        "--path_to_content",
        "-cp",
        type=str,
        help="Путь к одиночному изображению/папке c изображениями",
    )
    parser.add_argument(
        "--use_cuda",
        "-uc",
        action="store_true",
        help="Использовать ли CUDA для инференса",
    )
    args = parser.parse_args()

    return args


def main() -> None:
    args = arguments_parser()

    name_of_classifier = args.name_of_classifier
    path_to_weights = args.path_to_weights
    path_to_content = args.path_to_content
    use_cuda = args.use_cuda

    if use_cuda:
        device = "cuda:0"
    else:
        device = "cpu"
    
    mod = load_classifier(name_of_classifier,path_to_weights,device)
    class_id = inference_classifier(mod, path_to_content)
    image = cv2.imread(path_to_content)
    cv2.imshow(class_id, image)
    cv2.resizeWindow(class_id, 200, 200)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()