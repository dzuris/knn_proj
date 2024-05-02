import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.models import mobilenet_v2
from ..eval.eval import get_model
import yaml

# Define the Student Model
class StudentModel(nn.Module):
    def __init__(self, num_classes):
        super(StudentModel, self).__init__()
        self.mobilenet = mobilenet_v2(pretrained=False)
        self.mobilenet.classifier[1] = nn.Linear(1280, num_classes) # Adjust last layer for number of classes

    def forward(self, x):
        return self.mobilenet(x)
    
# Define the Distillation Loss
class DistillationLoss(nn.Module):
    def __init__(self, temperature=3):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=1)
        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_outputs, teacher_outputs):
        student_probs = self.softmax(student_outputs / self.temperature)
        teacher_probs = self.softmax(teacher_outputs / self.temperature)
        return self.kl_div_loss(student_probs, teacher_probs)
    
# Train the Student Model
def train_student(student_model, teacher_model, train_loader, optimizer, distillation_loss, device):
    student_model.train()
    teacher_model.eval()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        student_outputs = student_model(inputs)
        with torch.no_grad():
            teacher_outputs = teacher_model(inputs)
        loss = distillation_loss(student_outputs, teacher_outputs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(train_loader.dataset)

# Evaluate the Student Model
def evaluate_student(student_model, test_loader, device):
    student_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = student_model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return correct / total

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Distillation train')

    args = parser.parse_args()

    with open(args.path_weights + "config.yaml", "r") as stream:
        data = yaml.safe_load(stream)

    # Check if the GPU is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Example usage
    num_classes = data['n_classes'] # Number of classes in your dataset
    student_model = StudentModel(num_classes=num_classes)
    teacher_model = get_model(data, device) # Your original teacher model
    optimizer = optim.Adam(student_model.parameters(), lr=data['lr'])
    distillation_loss = DistillationLoss(temperature=3)

    # Train the student model
    # TODO: train and test loaders I do not know how to obtain them and chatbot kept his secrets
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train_student(student_model, teacher_model, train_loader, optimizer, distillation_loss, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")

    # Evaluate the student model
    accuracy = evaluate_student(student_model, test_loader, device)
    print(f"Student Model Accuracy: {accuracy:.4f}")
