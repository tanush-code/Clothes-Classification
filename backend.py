from model import NeuralNetwork
import torch
from PIL import Image
from torchvision import transforms

transforms = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(), #Convert it to tesnsor
])


model = NeuralNetwork()
model.load_state_dict(torch.load("model_weight.pth"))
model.eval()

def predict_image(image_path, model):
    img = Image.open(image_path)
    img_tensor = transforms(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        predicted_class = output.argmax(1)
        label = model.labels[predicted_class.item()]
        return label

print(predict_image('Imagedata/shirt.jpg', model))
