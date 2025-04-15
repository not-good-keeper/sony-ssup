import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import gc

def main():
    torch.backends.cudnn.benchmark = True
    
    torch.cuda.empty_cache()
    gc.collect()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    model.classifier[-1] = nn.Linear(in_features=1280, out_features=10)
    model.to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    scaler = torch.amp.GradScaler()

    def load_data():
        transform_train = transforms.Compose([
            transforms.Resize((96, 96)),  
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        ])
        transform_test = transforms.Compose([
            transforms.Resize((96, 96)),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        ])
        train_dataset = CIFAR10(root='./data', train=True, transform=transform_train, download=True)
        val_dataset = CIFAR10(root='./data', train=False, transform=transform_test, download=True)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, 
                                 num_workers=2, pin_memory=True, 
                                 persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, 
                               num_workers=2, pin_memory=True, 
                               persistent_workers=True)
        return train_loader, val_loader

    print("Loading dataset...")
    train_loader, val_loader = load_data()
    print("Dataset loaded successfully.")

    num_epochs = 30
    accumulation_steps = 4  
    
    best_acc = 0
    early_stop_patience = 5
    no_improve_count = 0
    
    model.train()
    print("Starting training loop...")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs} started")
        running_loss = 0.0
        correct, total = 0, 0
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
        
        for batch_idx, (inputs, labels) in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            if (batch_idx % accumulation_steps == 0) or (batch_idx + 1 == len(train_loader)):
                optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels) / accumulation_steps  
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1 == len(train_loader)):
                scaler.step(optimizer)
                scaler.update()
            
            running_loss += loss.item() * accumulation_steps  
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': running_loss/(batch_idx+1), 'acc': 100. * correct / total})
        
        scheduler.step()
        train_accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%")
        
        model.eval()
        correct, total = 0, 0
        val_loss = 0.0
        print("Evaluating...")
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * correct / total
        print(f'Validation Loss: {val_loss/len(val_loader):.4f}, Accuracy: {val_accuracy:.2f}%')
        
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            no_improve_count = 0
            torch.save(model.state_dict(), "mobilenetv3_cifar10_best.pth")
            print(f"New best accuracy: {best_acc:.2f}%. Model saved.")
        else:
            no_improve_count += 1
            print(f"No improvement for {no_improve_count} epochs. Best accuracy: {best_acc:.2f}%")
            
        if no_improve_count >= early_stop_patience:
            print(f"Early stopping after {epoch+1} epochs.")
            break
            
        model.train()
        
        torch.cuda.empty_cache()
        gc.collect()
        
    torch.save(model.state_dict(), "mobilenetv3_cifar10_final.pth")
    print("Final model saved successfully.")
    print(f"Best validation accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()