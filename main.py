import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import json
import copy
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


def save_weights_to_json(weights, path, filename):
    Path(path).mkdir(parents=True, exist_ok=True)
    weights_dict = {}
    
    for name, tensor in weights.items():
        array = tensor.cpu().numpy()
        weights_dict[name] = {
            # 'dtype': str(array.dtype),
            'shape': list(array.shape),
            'data': array.tolist()
        }
    
    with open(Path(path)/filename, 'wb') as f:
        pickle.dump(weights_dict, f)

def load_weights_from_json(filepath, device):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    weights = {}
    for name, meta in data.items():
        # array = np.array(meta['data'], dtype=meta['dtype'])
        array = np.array(meta['data'])
        weights[name] = torch.from_numpy(array).to(device)
        
    return weights


# Data handling
def create_non_iid_partition(dataset, num_clients, classes_per_client=2):
    sorted_indices = torch.argsort(torch.tensor(dataset.targets))
    partitions = []
    class_size = len(dataset) // 10
    
    for i in range(num_clients):
        class_indices = np.random.choice(10, classes_per_client, replace=False)
        client_indices = []
        for c in class_indices:
            start = c * class_size
            end = (c + 1) * class_size
            client_indices.extend(sorted_indices[start:end][i::num_clients])
        partitions.append(Subset(dataset, client_indices))
    
    return partitions


# Model definition
class MNISTCNN(nn.Module):
    def __init__(self):
        super(MNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize and save genesis model
genesis_model = MNISTCNN().to(device)
save_weights_to_json(
            genesis_model.state_dict(),
            f"fl_weights/round_{0}",
            "global_weights.pkl"
        )

# Federated Trainer
class FLTrainer:
    def __init__(self, trainer_id, eth_address, data_loader, device, save_dir):
        self.trainer_id = trainer_id
        self.eth_address = eth_address
        self.data_loader = data_loader
        self.device = device
        self.save_dir = save_dir
        self.criterion = nn.CrossEntropyLoss()

    def train(self, global_weights, lr=0.01, epochs=1):
        model = MNISTCNN().to(self.device)
        model.load_state_dict(global_weights)
        model.train()
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        for _ in range(epochs):
            for images, labels in self.data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        return model.state_dict()
    
# Federated Aggregator
class FLAggregator:
    def __init__(self, aggregator_id, eth_address, device, save_dir):
        self.device = device
        self.aggregator_id = aggregator_id
        self.eth_address = eth_address
        self.save_dir = Path(save_dir)
        self.metrics = defaultdict(list)
        self.current_round = 0
        self.global_model = genesis_model

    def save_global_model(self, round_num):
        save_weights_to_json(
            self.global_model.state_dict(),
            self.save_dir / f"round_{round_num}",
            "global_weights.pkl"
        )

    def load_global_model(self, round_num):
        return load_weights_from_json(
            self.save_dir / f"round_{round_num}" / "global_weights.pkl",
            self.device
        )

    def aggregate(self, client_updates):
        aggregated = {}
        global_weights = self.global_model.state_dict()
        
        for key in global_weights.keys():
            updates = torch.stack([update[key] for update in client_updates])
            aggregated[key] = torch.mean(updates, dim=0)
        
        self.global_model.load_state_dict(aggregated)
        return aggregated





# Federated Evaluator
class FLEvaluator:
    def __init__(self, evaluator_id, eth_address, device, save_dir):
        self.device = device
        self.evaluator_id = evaluator_id
        self.eth_address = eth_address
        self.save_dir = Path(save_dir)
        self.metrics = defaultdict(list)
        self.current_round = 0
        self.global_model = genesis_model

    def load_gbl_model(self, path):
        weights = load_weights_from_json(path, self.device)
        self.global_model.load_state_dict(weights)
    
    def evaluate(self, test_loader):
        self.global_model.eval()
        total_loss, correct = 0.0, 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.global_model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(test_loader)
        accuracy = 100 * correct / len(test_loader.dataset)
        return avg_loss, accuracy

    def load_global_model(self, round_num):
        return load_weights_from_json(
            self.save_dir / f"round_{round_num}" / "global_weights.pkl",
            self.device
        )
        

def main(args):
    print(f"USING DEVIVE: {device}")
    
    
    # Prepare data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Initialize the Participants
    
    aggregator = FLAggregator(1, "Agg Eth Addr", device, args.save_dir)
    evaluator = FLEvaluator(1, "Ev eth Addr", device, args.save_dir)
    trainer_datasets = create_non_iid_partition(train_dataset, args.num_trainers)
    
    trainers = [
        FLTrainer(
            i, 
            "Tr Eth Addr", 
            DataLoader(ds, batch_size=args.batch_size, shuffle=True),
            device, 
            args.save_dir
        )   for i, ds in enumerate(trainer_datasets)
    ]
    
    for round_num in range(1, args.num_rounds + 1):
        global_weights = aggregator.load_global_model(round_num - 1)
        
        trainer_updates = []
        for trainer in tqdm(trainers, desc="Clients Training"):
            update = trainer.train(global_weights, lr=args.learning_rate, epochs=args.local_epochs)
            
            save_path = f"{trainer.save_dir}/round_{round_num}"
            save_weights_to_json(
                update,
                save_path,
                f"client_{trainer.trainer_id}_weights.pkl"
            )
            trainer_updates.append(update)
            
        # Aggregation
        aggregator.aggregate(trainer_updates)
        aggregator.save_global_model(round_num)
        
        #Evaluation
        evaluator.load_gbl_model(evaluator.save_dir / f"round_{round_num}" / "global_weights.pkl")
        test_loss, test_acc = evaluator.evaluate(test_loader)
        print(f"Round {round_num} - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
        evaluator.metrics['test_loss'].append(test_loss)
        evaluator.metrics['test_acc'].append(test_acc)
        # Save final metrics
        
    with open(evaluator.save_dir / "metrics.json", 'w') as f:
        json.dump(evaluator.metrics, f)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning with MNIST")
    parser.add_argument("--num-trainers", type=int, default=10)
    parser.add_argument("--num-rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--save-dir", type=str, default="fl_weights")
    
    args = parser.parse_args()
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    main(args)
