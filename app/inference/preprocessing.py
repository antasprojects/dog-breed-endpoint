import torchvision.transforms as transforms
from PIL import Image

preprocess = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def prepare_image(img: Image.Image):
    """
    Takes a PIL image -> returns a torch tensor with shape (1, C, H, W)
    """
    img = img.convert("RGB")
    return preprocess(img).unsqueeze(0)