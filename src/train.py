import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import data_processing
import models
import evaluation
import config


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


# --- Random Forest Experiment ---
def run_rf_experiment(features, filtered_test, seeds, tasks):
    auc_results = {seed: [] for seed in seeds}
    dauprc_results = {seed: [] for seed in seeds}

    print("Random Forest Experiment: \n")

    for seed in seeds:
        np.random.seed(seed)
        for task in tasks:
            data = filtered_test[filtered_test[:, 1] == task]
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

            clf = models.get_random_forest_classifier(seed)
            clf.fit(X_train, y_train)
            y_pred_proba = clf.predict_proba(X_test)[:, 1]
            predictions_tensor = torch.tensor(y_pred_proba)
            labels_tensor = torch.tensor(y_test)
            target_ids_tensor = torch.zeros_like(labels_tensor)

            auc_score = evaluation.compute_roc_auc_score(y_test, y_pred_proba)
            mean_dauprc, dauprcs, target_id_list = evaluation.compute_dauprc_score(
                predictions_tensor,
                labels_tensor,
                target_ids_tensor)

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

    print(
        f"\nMean AUC Score over all seeds: {np.mean(mean_auc_seeds):.4f}, Standard Deviation over all seeds: {np.std(mean_auc_seeds):.4f}")
    print(
        f"Mean DAUPRC Score over all seeds: {np.nanmean(mean_dauprc_seeds):.4f}, Standard Deviation over all seeds: {np.nanstd(mean_dauprc_seeds):.4f}")
    return np.mean(mean_auc_seeds), np.std(mean_auc_seeds), np.nanmean(mean_dauprc_seeds), np.nanstd(mean_dauprc_seeds)

# -- Frequent Hitters Experiment --
def run_fh_experiment(features, labels_train, labels_val, labels_test, seeds):

    print("Frequent Hitter Experiment: \n")

    features_train = features[labels_train[:, 0]]
    features_val = features[labels_val[:, 0]]
    features_test = features[labels_test[:, 0]]

    mean_roc_aucs_seeds, std_roc_aucs_seeds = [], []
    mean_dauprcs_seeds, std_dauprcs_seeds = [], []

    for seed in seeds:
        set_seed(seed)

        batch_size = 8192

        train_dataset = MoleculeDataset(features_train, labels_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                  generator=torch.Generator().manual_seed(seed))

        val_tasks = [10, 11, 12]
        val_datasets = {}
        val_loaders = {}
        for task in val_tasks:
            val_datasets[task] = MoleculeDataset(features_val[labels_val[:, 1] == task],
                                                 labels_val[labels_val[:, 1] == task])
        for task, dataset in val_datasets.items():
            val_loaders[task] = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False,
                                           generator=torch.Generator().manual_seed(seed))

        test_tasks = [13, 14, 15]
        test_datasets = {}
        test_loaders = {}
        for task in test_tasks:
            test_datasets[task] = MoleculeDataset(features_test[labels_test[:, 1] == task],
                                                  labels_test[labels_test[:, 1] == task])
        for task, dataset in test_datasets.items():
            test_loaders[task] = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False,
                                            generator=torch.Generator().manual_seed(seed))

        input_size = features_train.shape[1]
        hs1 = 64
        hs2 = 32
        hs3 = 16
        output_size = 1
        num_epochs = 5
        num_steps = len(train_loader)
        learning_rate = 0.01
        model = models.NeuralNet(input_size, hs1, hs2, hs3, output_size)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Training
        print(f"Seed {seed}:")
        for epoch in range(num_epochs):
            model.train()
            for i, (features, labels) in enumerate(train_loader):
                labels = labels.unsqueeze(1)
                outputs = model(features)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (i + 1) % 5 == 0:
                    print(f"Epoch {epoch + 1}/{num_epochs}, Step {i + 1}/{num_steps}, Loss = {loss}")
            print()

            # Validation
            model.eval()
            with torch.no_grad():
                for task, val_loader in val_loaders.items():
                    all_labels = []
                    all_preds = []
                    for features, labels in val_loader:
                        labels = labels.unsqueeze(1)
                        outputs = model(features)
                        all_preds.extend(outputs.numpy().flatten())
                        all_labels.extend(labels.numpy().flatten())

                    all_labels = np.array(all_labels, dtype=int)
                    all_preds = np.array(all_preds)

                    # roc_auc score
                    roc_auc = evaluation.compute_roc_auc_score(all_labels, all_preds)

                    # dauprc score
                    all_labels_tensor = torch.tensor(all_labels)
                    all_preds_tensor = torch.tensor(all_preds)
                    target_ids_tensor = torch.zeros_like(all_labels_tensor)
                    mean_dauprc, dauprcs, target_id_list = evaluation.compute_dauprc_score(all_preds_tensor, all_labels_tensor,
                                                                                target_ids_tensor)

                    print(f"Validation Task {task}, ROC AUC Score: {roc_auc:.4f}, Mean DAUPRC Score: {mean_dauprc:.4f}")
                print()

        # Testing
        print("\nResults on the test set:")
        auc_results_fh = []
        dauprc_results_fh = []

        model.eval()
        with torch.no_grad():
            for task, test_loader in test_loaders.items():
                all_preds = []
                all_labels = []
                for features, labels in test_loader:
                    labels = labels.unsqueeze(1)
                    outputs = model(features)
                    all_preds.extend(outputs.numpy().flatten())
                    all_labels.extend(labels.numpy().flatten())

                all_labels = np.array(all_labels, dtype=int)
                all_preds = np.array(all_preds)

                # ROC AUC Score
                roc_auc = evaluation.compute_roc_auc_score(all_labels, all_preds)

                # DAUPRC Score
                all_preds_tensor = torch.tensor(all_preds)
                all_labels_tensor = torch.tensor(all_labels)
                target_ids_tensor = torch.zeros_like(all_labels_tensor)
                mean_dauprc, dauprcs, target_id_list = evaluation.compute_dauprc_score(all_preds_tensor, all_labels_tensor,
                                                                            target_ids_tensor)

                auc_results_fh.append(roc_auc)
                dauprc_results_fh.append(mean_dauprc)

                print(f"Test Task {task}, ROC AUC Score: {roc_auc:.4f}, Mean DAUPRC Score: {mean_dauprc:.4f}")

        mean_roc_aucs_seeds.append(np.mean(auc_results_fh))
        mean_dauprcs_seeds.append(np.nanmean(dauprc_results_fh))

        print(f"\n Mean ROC AUC Score across all test tasks: {np.mean(auc_results_fh):.4f}")
        print(f"Mean DAUPRC Score across all test tasks: {np.mean(dauprc_results_fh):.4f}")
        print(
            "\n#########################################################################################################################\n")

    print()
    print(
        f"\nMean ROC AUC Score over all seeds: {np.mean(mean_roc_aucs_seeds):.4f}, Standard Deviation: {np.std(mean_roc_aucs_seeds):.4f}")
    print(
        f"Mean DAUPRC Score over all seeds: {np.mean(mean_dauprcs_seeds):.4f}, Standard Deviation:{np.std(mean_dauprcs_seeds):.4f}")

    return [np.mean(mean_roc_aucs_seeds), np.std(mean_roc_aucs_seeds), np.mean(mean_dauprcs_seeds), np.std(mean_dauprcs_seeds)]


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


# --- Main execution block ---
if __name__ == "__main__":
    # 1. Load and process data
    all_features, label_matrix = data_processing.load_and_preprocess_data(config.DATA_PATH)

    # 2. Split labels into train/val/test sets by task
    train_label_set = data_processing.filter_labels(label_matrix[:, config.TRAIN_TASKS_SLICE])
    val_label_set = data_processing.filter_labels(label_matrix[:, config.VALIDATION_TASKS_SLICE])
    test_label_set = data_processing.filter_labels(label_matrix[:, config.TEST_TASKS_SLICE])

    print(f"Train samples: {len(train_label_set)}, Val samples: {len(val_label_set)}, Test samples: {len(test_label_set)}")

    # 3. Run Experiments
    rf_results = run_rf_experiment(all_features, test_label_set, config.SEEDS, config.RF_TASKS)
    fh_results = run_fh_experiment(all_features, train_label_set, val_label_set, test_label_set, config.SEEDS)
