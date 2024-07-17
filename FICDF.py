# FICDF
# author Shengli Ding, assisted by ChatGPT 4o

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset, TensorDataset
from torchvision import transforms
from PIL import Image
import os
import random
import copy
import flwr as fl
import numpy as np
import time
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import pandas as pd
from sklearn.cluster import KMeans
from ResNet import resnet18_cbam



#---------------------------------------------- Parameter Selections ---------------------------------------------------
setting = 2  # Setting in Table III
max_samples_per_centroid = 1  # Coefficient a in formular (6)
cuda_usage = 'cuda:0'

# Initialize datasets
path_iot_dataset_root = 'YOUR PATH TO/processedData_IoTSentinel'
path_annotation_train = 'YOUR PATH TO/train_annotations_IoTSentinel.csv'
path_annotation_val   = 'YOUR PATH TO/val_annotations_IoTSentinel.csv'
path_annotation_test  = 'YOUR PATH TO/test_annotations_IoTSentinel.csv'
log_dir               = 'YOUR PATH TO/log'  # You might need to creat your own folder to store running log (e.g.
                                            # named 'log')

num_prototypes_per_class = 40
learning_rate = 0.01
num_global_rounds = 30
local_epochs_per_round = 5  #  Number of local epochs per client per round
local_epochs_first_round = 10
seed_for_cuda = 2024  # Different from myseed, which is for IoT selection, in line 49

#--------------------------------------------- Parameter Selections END-------------------------------------------------


if setting == 1:
    tasks = [5, 10, 15, 20, 25, 30]
    clientnumber = 3
    myseed = 2024
    version = 'FICDF_version1_v2.6_1time'
elif setting == 2:
    tasks = [5, 10, 15, 20, 25, 30]
    clientnumber = 3
    myseed = 3024
    version = 'FICDF_version2_v2.6_1time'
elif setting == 3:
    tasks = [7, 14, 21, 28]
    clientnumber = 3
    myseed = 2024
    version = 'FICDF_version3_v2.6_1time'
elif setting == 4:
    tasks = [5, 10, 15, 20, 25, 30]
    clientnumber = 4
    myseed = 2024
    version = 'FICDF_version4_v2.6_1time'

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
def log_test_accuracy(global_round, test_accuracy, F1, current_time=current_time, log_dir=log_dir, version=version):
    # Get the current time in the specified format
    log_file_name = f"{version}_{current_time}.txt"
    log_file_path = os.path.join(log_dir, log_file_name)

    # Create the log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Check if the file exists, if not, create it
    if not os.path.exists(log_file_path):
        with open(log_file_path, "w") as log_file:
            log_file.write(f"Test Accuracy after round {global_round + 1}: {test_accuracy:.2f}% , F1: {F1:.2f}%\n")
    else:
        with open(log_file_path, "a") as log_file:
            log_file.write(f"Test Accuracy after round {global_round + 1}: {test_accuracy:.2f}% , F1: {F1:.2f}%\n")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(seed_for_cuda)
np.random.seed(myseed)  # for class splitting

# Get the current date and time
now = datetime.now()
current_time = now.strftime("%Y-%m-%d %H:%M:%S")  # Format the date and time
print("FICDF: ", current_time)

# Start the timer
start_time = time.time()

class IoTDeviceDataset(Dataset):
    def __init__(self, annotation, root_dir, selected_classes, data_type='', transform=None, transform_noise=None):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_type = data_type
        self.selected_classes = selected_classes
        self.annotations = annotation
        self.root_dir = root_dir
        self.transform = transform
        self.transform_noise = transform_noise
        self.class_group_of_iot = []

        if self.data_type != 'fisher':
            self.filter_annotations()
        else:
            print("\n fisher dataset initialized \n")


    def get_annotation(self):
        return self.annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        img_path = self.root_dir + self.annotations.iloc[idx, 0]
        image = Image.open(img_path)
        label = int(self.annotations.iloc[idx, 1])

        if self.transform:
            if (self.data_type == 'train') and (label not in self.class_group_of_iot):
                image = self.transform_noise(image)
            else:
                image = self.transform(image)

        return image, label

    def filter_annotations(self):
        self.annotations = self.annotations[self.annotations[1].isin(self.selected_classes)]

    def  filter_exemplars(self):
        self.annotations = self.annotations[~self.annotations[1].isin(self.selected_classes)]

    def update_annotation_for_fisher(self, current_class_group_of_iot):
        self.annotations = self.annotations[self.annotations[1].isin(current_class_group_of_iot)]  # for all unique class
                                                                                                # data

    # update the training datset by filtered exemplar data 50*20centroids
    def update_traindate_prototype(self, prototype_df_traindata_current):
        prototype_df_traindata_current.columns = self.annotations.columns
        self.annotations = prototype_df_traindata_current

    def add_exemplars(self, annotation_old_exemplars):
        print('')
        annotation_old_exemplars.columns = self.annotations.columns
        self.annotations = pd.concat([annotation_old_exemplars, self.annotations], ignore_index=True)

    def display_annotations(self):
        label_counts = self.annotations.iloc[:, 1].value_counts().sort_index()
        if  self.data_type == 'fisher':
            print('\nFisher data')
            for label, count in label_counts.items():
                    print(f"IoT               : {label} {count}")
        else:
            print('\nTraining data with old exemplars')
            for label, count in label_counts.items():
                if label in self.class_group_of_iot:
                    print(f"Own IoT           : {label} {count}")
                else:
                    print(f"Sharing Noised IoT: {label} {count}")


def select_annotation_prototypes(dataset, client_model, device, current_classes, num_prototypes_per_class=40,
                                 max_samples_per_centroid=max_samples_per_centroid):
    """
    Select prototype data for each class using the KNN multi-centroid exemplar algorithm.

    :param dataset: The dataset containing the images and labels.
    :param client_model: The client model containing the feature extractor.
    :param num_prototypes_per_class: Number of prototype images to select per class.
    :param device: Device to perform computation ('cpu' or 'cuda:0').
    :param max_samples_per_centroid: Maximum number of samples to select closest to each centroid.
    :return: Two pandas DataFrames containing the prototypes and the training data prototypes in annotation format.
    """
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    client_model.to(device)
    client_model.eval()

    # Extract the feature extractor part from the client model
    feature_extractor = client_model.feature

    # Step 1: Extract features for all images
    features = []
    labels = []
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            outputs = feature_extractor(images)
            features.append(outputs.cpu().numpy())
            labels.append(targets.cpu().numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    prototypes = []
    prototypes_traindata = []

    for cls in current_classes:
        class_indices = np.where(labels == cls)[0]
        class_features = features[class_indices]

        # Perform K-means clustering to find centroids
        kmeans = KMeans(n_clusters=num_prototypes_per_class, random_state=0).fit(class_features)
        centroids = kmeans.cluster_centers_

        # Select the closest sample for each centroid for prototypes
        selected_indices = []

        for centroid in centroids:
            distances = np.linalg.norm(class_features - centroid, axis=1)
            selected_idx = np.argmin(distances)
            selected_indices.append(class_indices[selected_idx])
            class_features = np.delete(class_features, selected_idx, axis=0)
            class_indices = np.delete(class_indices, selected_idx)

        prototypes.extend([(dataset.annotations.iloc[idx, 0], labels[idx]) for idx in selected_indices])

        # Select at most max_samples_per_centroid for prototypes_traindata
        class_indices = np.where(labels == cls)[0]
        class_features = features[class_indices]
        for _ in range(max_samples_per_centroid-1):
            for _, centroid in enumerate(centroids):
                if len(class_features) == 0:
                    break
                else:
                    distances = np.linalg.norm(class_features - centroid, axis=1)
                    selected_idx = np.argmin(distances)
                    selected_indices.append(class_indices[selected_idx])
                    class_features = np.delete(class_features, selected_idx, axis=0)
                    class_indices = np.delete(class_indices, selected_idx)

        prototypes_traindata.extend([(dataset.annotations.iloc[idx, 0], labels[idx]) for idx in selected_indices])

    # Create DataFrames from the prototypes
    prototype_df = pd.DataFrame(prototypes, columns=['image', 'label'])
    prototype_df_traindata = pd.DataFrame(prototypes_traindata, columns=['image', 'label'])

    def print_statistics(df, df_name):
        # unique_labels = df['label'].unique()
        # print(f"Number of unique labels in {df_name}: {len(unique_labels)}")
        label_counts = df['label'].value_counts()
        # print(f"Number of items for each label in {df_name}:")
        for label, count in label_counts.items():
            print(f"Label {label}: {count} items")

    print_statistics(prototype_df, 'prototype_df')
    print_statistics(prototype_df_traindata, 'prototype_df_traindata')

    return prototype_df, prototype_df_traindata


class Network(nn.Module):
    def __init__(self, num_classes, feature_extractor):
        super(Network, self).__init__()
        self.feature = feature_extractor
        self.fc = nn.Linear(feature_extractor.fc.in_features, num_classes, bias=True)

    def forward(self, input):
        x = self.feature(input)
        x = self.fc(x)
        return x

    def incremental_learning(self, num_classes):
        weight = self.fc.weight.data
        bias = self.fc.bias.data
        in_features = self.fc.in_features
        out_features = self.fc.out_features

        self.fc = nn.Linear(in_features, num_classes, bias=True)
        self.fc.weight.data[:out_features] = weight
        self.fc.bias.data[:out_features] = bias
        print('Global Model Updated, classification output #: ', num_classes)

    def feature_extractor(self, inputs):
        return self.feature(inputs)


def model_to_device(model, device):
    return model.to(device)


def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def evaluate_confusion_matrix(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    num_classes = len(np.unique(all_labels))
    conf_matrix = confusion_matrix(all_labels, all_predictions, labels=np.arange(num_classes))

    # print(f'Overall Accuracy: {accuracy:.2f}%')
    print('Confusion Matrix:')
    conf_matrix_df = pd.DataFrame(conf_matrix, index=np.arange(num_classes), columns=np.arange(num_classes))
    print(conf_matrix_df.to_string(index=True, header=True))

    class_correct = np.diag(conf_matrix)
    class_total = conf_matrix.sum(axis=1)

    class_accuracies = []
    for i in range(num_classes):
        if class_total[i] > 0:
            class_accuracy = 100 * class_correct[i] / class_total[i]
            class_accuracies.append(class_accuracy)
            print(f'Accuracy of class {i}: {class_accuracy:.2f}%')
        else:
            print(f'Accuracy of class {i}: N/A (no samples)')

    # Calculate average class accuracy
    avg_class_accuracy = np.mean(class_accuracies)

    # Assuming all_labels and all_predictions are defined and num_classes is set
    # Calculate precision, recall, and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions,
                                                               labels=np.arange(num_classes), average='macro',
                                                               zero_division=0)

    print(f'Overall Accuracy: {avg_class_accuracy:.2f}%')
    print(f'Precision: {precision*100:.2f}%')
    print(f'Recall: {recall*100:.2f}%')
    print(f'F1 Score: {f1*100:.2f}%')

    return avg_class_accuracy, f1*100


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise


# Example usage of AddGaussianNoise in the train_transform
train_transform_noise = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    AddGaussianNoise(mean=0., std=0.1),
    transforms.Normalize((0.5,), (0.5,))
])
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Device selection
device = torch.device(cuda_usage if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Server definition for federated learning
class FLServer(fl.server.Server):
    def __init__(self, client_manager, model, device):
        super().__init__(client_manager=client_manager)
        self.global_model = model
        self.val_loader = None
        self.device = device

    def fit(self, num_global_rounds, local_epochs_per_round, clients, test_loader, exemplar_fl):
        self.test_loader = test_loader
        old_test_accuracy = 0
        count = 0
        for global_round in range(num_global_rounds):
            print(f"Global round {global_round + 1}/{num_global_rounds}")
            # Send global model to clients and get updated local models
            for current_client in clients:
                current_client.local_epochs_per_round = local_epochs_per_round

            updated_local_models = [current_client.train_and_update(self.global_model, num_global_rounds, global_round)
                                    for current_client in clients]

            # Aggregate updated local models to form the new global model
            self.aggregate_models(updated_local_models)
            # Evaluate the global model on validation data after each round
            test_accuracy, F1_score_in100 = evaluate_confusion_matrix(self.global_model, self.test_loader, self.device)
            # print(f"Validation Accuracy after aggregation: {val_accuracy:.2f}%")
            print(f"Test Accuracy after round {num_global_rounds + 1}: {test_accuracy:.2f}%")

            difference = abs(test_accuracy - old_test_accuracy)
            print(difference, 'count: ', count)

            if not exemplar_fl:
                log_test_accuracy(global_round, test_accuracy, F1_score_in100)
            if  difference <= 0.5:
                count+=1
            else:
                count = 0

            old_test_accuracy = test_accuracy

            # Select the exemplars for each class
            if global_round == 0 and exemplar_fl == True:
                for client in clients:
                    (client.prototypes_annotation_current,
                     client.prototype_df_traindata_current) = select_annotation_prototypes(dataset=client.train_dataset,
                                                              client_model=client.model, device=self.device,
                                                              current_classes=client.current_classes,
                                                              num_prototypes_per_class=num_prototypes_per_class)

    # FedAvg
    def aggregate_models(self, client_models):
        # Deep copy the state dict of the first client model to initialize the global state dict
        global_state_dict = copy.deepcopy(client_models[0].state_dict())

        # Iterate over each parameter in the state dict
        for key in global_state_dict.keys():
            # Sum the parameters from all client models
            for i in range(1, len(client_models)):
                global_state_dict[key] += client_models[i].state_dict()[key]

            # Divide by the number of models to get the average
            global_state_dict[key] = torch.div(global_state_dict[key], len(client_models))

        # Load the averaged parameters into the global model
        self.global_model.load_state_dict(global_state_dict)

    def evaluate_confusion_matrix(self, data_loader):
        return evaluate_confusion_matrix(self.global_model, data_loader, self.device)


class FLClient:
    # def __init__(self, model, client_id, train_dataset, val_dataset, test_dataset):
    def __init__(self, model, client_id, learning_rate, device):
        self.model = model
        self.old_model = None
        self.client_id = client_id
        self.train_dataset = None  # train_dataset
        self.val_dataset = None  # val_dataset
        self.test_dataset = None  # test_dataset
        self.prototypes_annotation = pd.DataFrame(columns=['image', 'label'])
        self.prototypes_annotation_current = pd.DataFrame(columns=['image', 'label'])
        self.prototype_df_traindata_current = pd.DataFrame(columns=['image', 'label'])
        self.unique_train_data = pd.DataFrame(columns=['image', 'label'])
        self.class_group_of_iot = []
        self.prototypes_indices = []
        self.local_epochs_per_round = 0
        self.learning_rate = learning_rate
        self.device = device
        self.current_classes = []

    def update_old_model(self, old_model):
        self.old_model = copy.deepcopy(old_model)

    def compute_fisher_information(self):
        fisher_train_data = IoTDeviceDataset(annotation=self.train_dataset.get_annotation(),
                                                    root_dir=self.train_dataset.root_dir, data_type='fisher',
                                                    selected_classes=self.current_classes,
                                                    transform=train_transform, transform_noise=train_transform_noise)
        # fisher_train_data.display_annotations()
        print('self.current_classes: ', self.current_classes)
        # fisher_train_data = copy.deepcopy(client.train_dataset)
        fisher_train_data.update_annotation_for_fisher(self.current_classes)
        # print('the data in update_annotation_for_fisher is')
        fisher_train_data.display_annotations()
        fisher_data_loader = DataLoader(fisher_train_data, batch_size=128, shuffle=False)


        fisher_information = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        self.model.eval()

        for inputs, labels in fisher_data_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher_information[name] += param.grad.data ** 2


        # Normalize fisher information
        for name in fisher_information:
            fisher_information[name] /= len(fisher_data_loader)

        return fisher_information

    def train_and_update(self, global_model, num_global_rounds, global_round):
        self.model.load_state_dict(global_model.state_dict())
        self.model.to(self.device)  # Ensure the model is on the correct device

        train_loader = DataLoader(self.train_dataset, batch_size=128, shuffle=True)  #
        val_loader = DataLoader(self.val_dataset, batch_size=128, shuffle=False)
        exemplars_dataset = copy.deepcopy(self.train_dataset)
        exemplars_dataset.filter_exemplars()
        exemplars_loader = DataLoader(exemplars_dataset, batch_size=128, shuffle=False)

        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=0.00001)
        criterion = nn.CrossEntropyLoss()

        self.old_model = copy.deepcopy(self.model)
        self.old_model.eval()

        # This variable will be used for dynamic learning rate adjustment
        current_global_round_portion = global_round / num_global_rounds

        for epoch in range(self.local_epochs_per_round):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                # # Knowledge Distillation
                # for replay_inputs, replay_labels in exemplars_loader:
                #     replay_inputs = replay_inputs.to(self.device)
                #     replay_labels = replay_labels.to(self.device)
                #
                #     replay_outputs = self.model(replay_inputs)
                #     with torch.no_grad():
                #         old_outputs = self.old_model(replay_inputs)
                #
                #     # Set the temperature for distillation
                #     temperature = 2.0
                #
                #     # Compute the distillation loss
                #     distillation_loss = nn.KLDivLoss(reduction='batchmean')(
                #         nn.functional.log_softmax(replay_outputs / temperature, dim=1),
                #         nn.functional.softmax(old_outputs / temperature, dim=1)
                #     )
                #
                #     # Add the distillation loss to the total loss
                #     loss += distillation_loss

                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            training_accuracy = 100 * correct / total
            val_accuracy = evaluate(self.model, val_loader, self.device)
            print(
                f"Client {self.client_id + 1} Epoch [{epoch + 1}/{self.local_epochs_per_round}], "
                f"Loss: {running_loss / len(train_loader):.4f}, "
                f"Training Accuracy: {training_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%")

        return self.model

annotation_train = pd.read_csv(path_annotation_train, header=None)
annotation_val = pd.read_csv(path_annotation_val, header=None)
annotation_test = pd.read_csv(path_annotation_test, header=None)

# Initialize model and clients
feature_extractor = resnet18_cbam()
global_model = Network(tasks[0], feature_extractor)
global_model = model_to_device(global_model, device)

clients = [FLClient(global_model, i, learning_rate, device=device) for i in range(clientnumber)]

# Create and run the federated learning server
client_manager = fl.server.client_manager.SimpleClientManager()
fl_server = FLServer(client_manager, global_model,device=device)

print("=" * 120)

# Divide new classes among clients for each task
for task_index, num_classes in enumerate(tasks):
    print(f"Training Task {task_index + 1} with {num_classes} classes")

    # ----------------------------------------------  update global model  ---------------------------------------------

    global_model.incremental_learning(num_classes)
    global_model = model_to_device(global_model, device)

    # ------------------------------------  assign IoT devices to each client (router)  --------------------------------

    # Determine the number of classes for the current task e.g. [4,5,6,7]
    if clientnumber == 3:
        num_classes = tasks[task_index]
        if task_index > 0:
            class_indices_current_task = np.arange(tasks[task_index - 1], num_classes)
        else:
            class_indices_current_task = np.arange(num_classes)

        # Shuffle the class indices for a random split
        np.random.shuffle(class_indices_current_task)

        while True:
            # Initialize three empty lists for the splits
            split1, split2, split3 = [], [], []

            # Randomly assign each class index to one of the three splits
            for index in class_indices_current_task:
                choice = np.random.choice([0, 1, 2])
                if choice == 0:
                    split1.append(index)
                elif choice == 1:
                    split2.append(index)
                else:
                    split3.append(index)
            loop = False
            class_splits = [split1, split2, split3]
            for split in class_splits:
                if len(split) == 0:
                    loop = True
            if not loop:
                break

    elif clientnumber == 4:
        # Define number of classes for the current task
        num_classes = tasks[task_index]

        if task_index > 0:
            class_indices_current_task = np.arange(tasks[task_index - 1], num_classes)
        else:
            class_indices_current_task = np.arange(num_classes)

        # Shuffle the class indices for a random split
        np.random.shuffle(class_indices_current_task)

        while True:
            # Initialize four empty lists for the splits
            split1, split2, split3, split4 = [], [], [], []

            # Randomly assign each class index to one of the four splits
            for index in class_indices_current_task:
                choice = np.random.choice([0, 1, 2, 3])
                if choice == 0:
                    split1.append(index)
                elif choice == 1:
                    split2.append(index)
                elif choice == 2:
                    split3.append(index)
                else:
                    split4.append(index)

            class_splits = [split1, split2, split3, split4]

            loop = False
            for split in class_splits:
                if len(split) == 0:
                    loop = True
            if not loop:
                break

    # ------------------------------------------  data set preparation  ------------------------------------------------

    # Create dataset for val and test in current task (e.g. [0,1,2,3,4,5,6,7])
    class_indices = np.arange(num_classes)
    val_dataset  = IoTDeviceDataset(annotation=annotation_val, root_dir=path_iot_dataset_root, data_type='val',
                                    selected_classes=class_indices, transform=test_transform)
    test_dataset = IoTDeviceDataset(annotation=annotation_test, root_dir=path_iot_dataset_root, data_type='test',
                                    selected_classes=class_indices, transform=test_transform)
    print('\n', '-' * 20, ' Client Dataset Info', '-' * 20, '\n')

    # ----------------------- initial client preparations and first FL round with prototype selection  -----------------

    active_clients = []
    for i, client in enumerate(clients):
        client_selected_classes = class_splits[i]
        client.current_classes = client_selected_classes
        client.class_group_of_iot = client.class_group_of_iot + client_selected_classes

        print(f"Client {i + 1} IoT device classes: {client.class_group_of_iot}; new: {client_selected_classes}")

        # Only include clients with assigned classes; The clients with old data will also be included
        if client.class_group_of_iot:
            print(f"Client {i + 1} is activated")
            # Set the client validation dataset to total dataset at that task e.g. [0,1,2,3]
            client.val_dataset = val_dataset

            # if client_selected_classes:
            client.train_dataset = IoTDeviceDataset(annotation=annotation_train,
                                                    root_dir=path_iot_dataset_root, data_type='train',
                                                    selected_classes=client_selected_classes,
                                                    transform=train_transform, transform_noise=train_transform_noise)
            #
            client.train_dataset.class_group_of_iot = client.class_group_of_iot  # Update the iot group for taring data
            client.train_dataset.add_exemplars(client.prototypes_annotation)  # Add old exemplar
            client.train_dataset.display_annotations()  # Display the training dataset

            # this code will get the unique training that only this client has for fisher info matrix

            active_clients.append(client)
            print('')
        else:
            print(f"Client {i + 1} is NOT activated")
            print('')

    # Perform first federated training to get the exemplars for each class
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    fl_server.fit(num_global_rounds=1, local_epochs_per_round=local_epochs_first_round,
                  clients=active_clients, test_loader=test_loader, exemplar_fl=True)

    print("=" * 120)

    # -------------------------------- update the clients' datasets with prototype sharing  ----------------------------

    active_clients = []
    for i, client in enumerate(clients):
        temp_annotation = pd.DataFrame(columns=['image', 'label'])
        # data sharing
        for other_client in clients:
            if client != other_client:
                temp_annotation.columns = other_client.prototypes_annotation_current.columns
                temp_annotation = pd.concat([temp_annotation, other_client.prototypes_annotation_current],
                                            ignore_index=True)

        if client.train_dataset:
            client.train_dataset.update_traindate_prototype(client.prototype_df_traindata_current)  # filter the dataset
            client.train_dataset.add_exemplars(client.prototypes_annotation)  # Add old exemplar again

            client.train_dataset.add_exemplars(temp_annotation)  # update the dataset with sharing data (noised will
                                                                 # be added when using DataLoader)
            active_clients.append(client)
            print(f'Client {i + 1}:')
            client.train_dataset.display_annotations()
        else:
            print(f'Client {i + 1}:\nNo data')

        # add client's own exemplar for next task
        client.prototypes_annotation_current.columns = client.prototypes_annotation.columns
        client.prototypes_annotation = pd.concat([client.prototypes_annotation,
                                                  client.prototypes_annotation_current], ignore_index=True)
        temp_annotation.columns = client.prototypes_annotation.columns
        client.prototypes_annotation = pd.concat([client.prototypes_annotation,
                                                  temp_annotation], ignore_index=True)
        print('')

    # ---------------------------------------- start the formal federated learning  ------------------------------------

    fl_server.fit(num_global_rounds=num_global_rounds, local_epochs_per_round=local_epochs_per_round,
                  clients=active_clients, test_loader=test_loader, exemplar_fl=False)

    # Evaluate the global model on the test dataset at the end of each task
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    test_accuracy, F1_score_in100 = fl_server.evaluate_confusion_matrix(test_loader)
    print('')
    print(f"Test Accuracy after Task {task_index + 1}: {test_accuracy:.2f}%")
    print("=" * 120)

# Final evaluation of the global model on the test dataset after all tasks
class_indices_final = np.arange(tasks[-1])
print('class_indices_final = np.arange(tasks[-1])', class_indices_final)

test_dataset = IoTDeviceDataset(annotation=annotation_test, root_dir=path_iot_dataset_root, data_type='final test',
                                selected_classes=class_indices_final, transform=test_transform)

test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
test_accuracy, F1_score_in100 = fl_server.evaluate_confusion_matrix(test_loader)
print('')
print('')
print(f"Final Test Accuracy after all Tasks: {test_accuracy:.2f}%")

# Print the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time:.2f} seconds")
