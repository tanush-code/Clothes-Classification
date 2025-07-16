# 🧠 Fashion Image Classification with PyTorch

This is a backend-only project for classifying fashion images using a custom-trained PyTorch neural network on the FashionMNIST dataset.

## 📦 What's Included

- `model.py` – Defines the neural network (`NeuralNetwork`) and class labels.
- `backend.py` – Loads the saved model and predicts labels for new images.
- `model_weight.pth` – Trained model weights.
- `Imagedata/` – Folder for input images to classify.

## 🛠 How It Works

1. Model is trained on FashionMNIST and saved using:
   ```python
   torch.save(model.state_dict(), "model_weight.pth")
