import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from data_processing import load_and_preprocess_data, filter_labels, get_standardized_features, set_seed
from evaluation import compute_roc_auc_score, compute_dauprc_score
from models import NeuralNet, get_random_forest_classifier
import config
import pickle


# --- Random Forest Experiment ---
def run_rf_experiment(features, label_matrix, seeds, tasks):
    auc_results = {seed: [] for seed in seeds}
    dauprc_results = {seed: [] for seed in seeds}
    label_matrix = label_matrix[:, 13:16]
    filtered_matrix = filter_labels(label_matrix)

    for seed in seeds:
        np.random.seed(seed)
        for task in tasks:
            data = filtered_matrix[filtered_matrix[:, 1] == task]
            X = features[data[:, 0]]
            y = data[:, 2]
            pos_indices = np.where(y == 1)[0]
            neg_indices = np.where(y == 0)[0]
            pos_train_indices = np.random.choice(pos_indices, size=5, replace=False)
            neg_train_indices = np.random.choice(neg_indices, size=5, replace=False)
            train_indices = np.concatenate([pos_train_indices, neg_train_indices])
            test_indices = [i for i in range(len(y)) if i not in train_indices]
            X_train, y_train = X[train_indices], y[train_indices]
            X_test, y_test = X[test_indices], y[test_indices]

            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            clf = get_random_forest_classifier(seed)
            clf.fit(X_train, y_train)
            y_pred_proba = clf.predict_proba(X_test)[:, 1]
            predictions_tensor = torch.tensor(y_pred_proba)
            labels_tensor = torch.tensor(y_test)
            target_ids_tensor = torch.zeros_like(labels_tensor)

            auc_score = compute_roc_auc_score(y_test, y_pred_proba)
            mean_dauprc, dauprcs, target_id_list = compute_dauprc_score(
                predictions_tensor,
                labels_tensor,
                target_ids_tensor
            )

            auc_results[seed].append(auc_score)
            dauprc_results[seed].append(mean_dauprc)

    mean_auc_seeds, std_auc_seeds = [], []
    mean_dauprc_seeds, std_dauprc_seeds = [], []
    for seed in seeds:
        auc_scores = auc_results[seed]
        mean_auc = np.mean(auc_scores)
        mean_auc_seeds.append(mean_auc)
        print(f"Seed {seed}:")
        print(f"Mean roc_auc_score = {mean_auc:.4f}")

        dauprcs = dauprc_results[seed]
        mean_dauprc = np.nanmean(dauprcs)
        mean_dauprc_seeds.append(mean_dauprc)
        print(f"Mean dauprc_score = {mean_dauprc:.4f}\n")

    print(f"\nMean AUC Score over all seeds: {np.mean(mean_auc_seeds):.4f},"
          f"Standard Deviation of AUC Score over all seeds: {np.std(mean_auc_seeds):.4f}")
    print(f"Mean DAUPRC Score over all seeds: {np.nanmean(mean_dauprc_seeds):.4f},"
          f"Standard Deviation of DAUPRC Score over all seeds: {np.nanstd(mean_dauprc_seeds):.4f}")

    return [mean_auc_seeds, mean_dauprc_seeds]


# --- Dataset Class ---
class MoleculeDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx, 2]
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


# --- Frequent Hitters Experiment ---
def run_fh_experiment(features, label_matrix, seeds):
    print("-------------------------------------------------------------------------------------------------------------")

    mean_roc_auc_seeds, std_roc_auc_seeds = [], []
    mean_dauprc_seeds, std_dauprc_seeds = [], []

    labels_train = label_matrix[:, :10]
    labels_val = label_matrix[:, 10:13]
    labels_test = label_matrix[:, 13:16]

    labels_train = filter_labels(labels_train)
    labels_val = filter_labels(labels_val)
    labels_test = filter_labels(labels_test)

    features_train = features[labels_train[:, 0]]
    features_val = features[labels_val[:, 0]]
    features_test = features[labels_test[:, 0]]

    # Standardization
    features_train, features_val, features_test = get_standardized_features(features_train, features_val, features_test)

    for seed in seeds:
        set_seed(seed)
        batch_size = config.BATCH_SIZE

        # Dataset and Dataloader
        train_dataset = MoleculeDataset(features_train, labels_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))

        val_tasks = config.VAL_TASKS
        val_datasets = {}
        val_loaders = {}
        for task in val_tasks:
            val_datasets[task] = MoleculeDataset(features_val[labels_val[:, 1] == task], labels_val[labels_val[:, 1] == task])
        for task, dataset in val_datasets.items():
            val_loaders[task] = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, generator=torch.Generator().manual_seed(seed))

        test_tasks = config.TEST_TASKS
        test_datasets = {}
        test_loaders = {}
        for task in test_tasks:
            test_datasets[task] = MoleculeDataset(features_test[labels_test[:, 1] == task], labels_test[labels_test[:, 1] == task])
        for task, dataset in test_datasets.items():
            test_loaders[task] = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, generator=torch.Generator().manual_seed(seed))

        # Model and hyperparameters
        input_size = features_train.shape[1]
        hs1 = config.HS1
        hs2 = config.HS2
        hs3 = config.HS3
        output_size = 1
        num_epochs = config.NUM_EPOCHS
        num_steps = len(train_loader)
        learning_rate = config.LEARNING_RATE
        model = NeuralNet(input_size, hs1, hs2, hs3, output_size)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Training
        print(f"\nSeed {seed}:")
        for epoch in range(num_epochs):
            model.train()
            for i, (features, labels) in enumerate(train_loader):
                labels = labels.unsqueeze(1)
                outputs = model(features)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (i+1) % 5 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Step {i+1}/{num_steps}, Loss = {loss}")
            print()

            # Validation
            model.eval()
            with torch.no_grad():
                for task, val_loader in val_loaders.items():
                    val_labels = []
                    val_preds = []
                    for features, labels in val_loader:
                        labels = labels.unsqueeze(1)
                        outputs = model(features)
                        val_preds.extend(outputs.numpy().flatten())
                        val_labels.extend(labels.numpy().flatten())

                    val_labels = np.array(val_labels, dtype=int)
                    val_preds = np.array(val_preds)

                    # ROC AUC Score
                    roc_auc_val = compute_roc_auc_score(val_labels, val_preds)

                    # DAUPRC Score
                    val_labels_tensor = torch.tensor(val_labels)
                    val_preds_tensor = torch.tensor(val_preds)
                    target_ids_tensor_val = torch.zeros_like(val_labels_tensor)
                    mean_dauprc_val, dauprcs_val, target_id_list_val = compute_dauprc_score(
                        val_preds_tensor, val_labels_tensor, target_ids_tensor_val
                    )
                    print(f"Validation Task {task}, ROC AUC Score: {roc_auc_val:.4f}, Mean DAUPRC Score: {mean_dauprc_val:.4f}")
                print()

        # Testing
        print(f"\nResults on the test set:")
        auc_results_fh = []
        dauprc_results_fh = []
        model.eval()
        with torch.no_grad():
            for task, test_loader in test_loaders.items():
                test_preds = []
                test_labels = []
                for features, labels in test_loader:
                    labels = labels.unsqueeze(1)
                    outputs = model(features)
                    test_preds.extend(outputs.numpy().flatten())
                    test_labels.extend(labels.numpy().flatten())

                test_labels = np.array(test_labels, dtype=int)
                test_preds = np.array(test_preds)

                # ROC AUC Score
                roc_auc_test = compute_roc_auc_score(test_labels, test_preds)

                # DAUPRC Score
                test_preds_tensor = torch.tensor(test_preds)
                test_labels_tensor = torch.tensor(test_labels)
                target_ids_tensor_test = torch.zeros_like(test_labels_tensor)
                mean_dauprc_test, dauprcs_test, target_id_list_test = compute_dauprc_score(
                    test_preds_tensor, test_labels_tensor, target_ids_tensor_test
                )

                auc_results_fh.append(roc_auc_test)
                dauprc_results_fh.append(mean_dauprc_test)
                print(f"Test task {task}, ROC AUC Score: {roc_auc_test:.4f}, Mean DAUPRC Score: {mean_dauprc_test:.4f}")

        mean_roc_auc_seeds.append(np.mean(auc_results_fh))
        mean_dauprc_seeds.append(np.nanmean(dauprc_results_fh))

        print(f"\nMean ROC AUC Score across all test tasks: {np.mean(auc_results_fh):.4f}")
        print(f"Mean DAUPRC Score across all test tasks: {np.nanmean(dauprc_results_fh):.4f}")

    print("\n############################################################################################################################\n")
    print(f"Mean ROC AUC Score over all seeds: {np.mean(mean_roc_auc_seeds):.4f},"
          f"Standard Deviation of ROC AUC Score over all seeds: {np.std(mean_roc_auc_seeds):.4f}")
    print(f"Mean DAUPRC Score over all seeds: {np.mean(mean_dauprc_seeds):.4f},"
          f"Standard Deviation of DAUPRC Score over all seeds: {np.std(mean_dauprc_seeds):.4f}")
    return [mean_roc_auc_seeds, mean_dauprc_seeds]


# --- Main Execution Block ---
if __name__ == "__main__":
    # 1. Load and process data
    all_features, muv_matrix = load_and_preprocess_data(config.DATA_PATH)

    # 2. Run experiments
    rf_results = run_rf_experiment(all_features, muv_matrix, config.SEEDS, config.RF_TASKS)
    fh_results = run_fh_experiment(all_features, muv_matrix, config.SEEDS)

    with open("results.pkl", "wb") as f:
        pickle.dump({"rf_results": rf_results, "fh_results": fh_results}, f)
