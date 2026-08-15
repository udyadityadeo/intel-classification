import torch
from PIL import Image
from torchvision import transforms

from app.model import ImprovedCNN

CLASS_NAMES = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street",
]


class ImageClassifier:

    def __init__(self, checkpoint_path):

        self.device = torch.device("cpu")

        self.model = ImprovedCNN(
            num_classes=len(CLASS_NAMES)
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device
        )

        self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

    def predict(self, image: Image.Image):

        image = image.convert("RGB")

        tensor = self.transform(image)

        tensor = tensor.unsqueeze(0)

        with torch.no_grad():

            logits = self.model(tensor)

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            confidence, prediction = (
                probabilities.max(dim=1)
            )

        return {
            "class": CLASS_NAMES[prediction.item()],
            "confidence": float(confidence.item()),
        }