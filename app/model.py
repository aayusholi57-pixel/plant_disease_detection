import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LeafClassifier:
    def __init__(self, model_path: str, data_dir: str = "data/raw"):
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
            
        # Filter out hidden files like .gitkeep or system files (.DS_Store)
        self.class_names = sorted([
            d for d in os.listdir(data_dir) 
            if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')
        ])
        
        # Load State Dict First to double check stored num_classes
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            # Inspect output weight tensor shape dynamically
            checkpoint_num_classes = state_dict['fc.weight'].shape[0]
            self.num_classes = checkpoint_num_classes
            
            # Load Architecture with exact matching classes
            self.model = models.resnet18(weights=None)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, self.num_classes)
            
            self.model.load_state_dict(state_dict)
            self.model.to(device)
            self.model.eval()
        else:
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ])

    def predict(self, image_bytes: bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor_img = self.transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = self.model(tensor_img)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_prob, top_class = torch.topk(probabilities, 1)

        raw_label = self.class_names[top_class.item()]
        clean_label = raw_label.replace("___", " - ").replace("_", " ")
        confidence = round(top_prob.item() * 100, 2)

        return raw_label, clean_label, confidence