import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(data_path):
    muv = pd.read_csv(data_path)

    # 1. Compute molecule features (ECFP fingerprints and descriptors)
    molecules = list(muv["smiles"])
    # create mol objects
    mols = list()
    for smiles in molecules:
        mol = Chem.MolFromSmiles(smiles)
        mols.append(mol)

    # ECFP fingerprints
    ecfps = list()
    for mol in mols:
        fp_sparseVec = rdFingerprintGenerator.GetCountFPs(
            [mol], fpType=rdFingerprintGenerator.MorganFP
        )[0]
        fp = np.zeros((0,), np.int8)  # Generate target pointer to fill
        DataStructs.ConvertToNumpyArray(fp_sparseVec, fp)
        ecfps.append(fp)
    ecfps = np.array(ecfps)

    # RDKit descriptors
    real_200_descr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                      35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
                      59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
                      83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105,
                      106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124,
                      125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
                      144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162,
                      163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181,
                      182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200,
                      201, 202, 203, 204, 205, 206, 207]
    rdkit_descriptors = list()
    for mol in mols:
        descrs = list()
        for descr in Descriptors._descList:
            _, descr_calc_fn = descr
            descrs.append(descr_calc_fn(mol))
        descrs = np.array(descrs)
        descrs = descrs[real_200_descr]
        rdkit_descriptors.append(descrs)
    rdkit_descriptors = np.array(rdkit_descriptors)

    features = np.hstack((ecfps, rdkit_descriptors))
    print(f"Computed features with shape: {features.shape}")

    # 2. Restructure labels matrix
    labels = muv.values[:, :-2]
    muv_matrix = np.empty(labels.shape, dtype=object)
    for r in range(len(labels)):
        for c in range(len(labels[0])):
            label = labels[r, c]
            muv_matrix[r, c] = (r, c, int(label) if label in [0.0, 1.0] else -1)

    return features, muv_matrix


def filter_labels(matrix):
    """Filters out triplets with a label of -1 (NaNs)"""
    matrix_copy = matrix.copy()
    flattened = matrix_copy.reshape(-1)
    filtered = []
    for triplet in flattened:
        if triplet[2] != -1:
            filtered.append(triplet)
    return np.array(filtered)


def get_standardized_features(features_train, features_val, features_test):
    """Standardizes features based on the training set"""
    scaler = StandardScaler()
    features_train = scaler.fit_transform(features_train)
    features_val = scaler.transform(features_val)
    features_test = scaler.transform(features_test)
    return features_train, features_val, features_test, scaler