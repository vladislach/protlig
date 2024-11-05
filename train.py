import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from collections import Counter
from constants import x_map, e_map
from model import Net

from Bio import SeqIO
from rdkit import Chem

import torch
from torch import nn

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import one_hot


# Load the data
df = pd.read_csv('ALL/DATA/NR-DBIND.csv', delimiter=';')
df['ID'] = df['ID'].astype(int)
df.dropna(subset=['accession'], inplace=True)
df.reset_index(drop=True, inplace=True)

row_indexes = {}

for index, row in df.iterrows():
    key = (row['accession'], row['canonical_smile'])
    if key in row_indexes:
        row_indexes[key].append(index)
    else:
        row_indexes[key] = [index]


# Create a dataset with the median p_binding_value for each unique protein-canonical_smiles pair
dataset = []

for (protein_id, smiles), indexes in row_indexes.items():
    values = [df.loc[index]['p_binding_value'] for index in indexes if not np.isnan(df.loc[index]['p_binding_value'])]
    if len(values) > 0:
        p_binding_value = np.median(values).item()

        if p_binding_value < 5.0 or p_binding_value >= 8:
            continue
        y_class = int(p_binding_value) - 5
        y_one_hot = torch.zeros(3)
        y_one_hot[y_class] = 1

        dataset.append(Data(
            protein_id=protein_id,
            smiles=smiles,
            y=p_binding_value,
            # y_class=y_one_hot,
            y_class=y_class)
        )


# Load the protein sequences
input_fasta_file = 'seqs_raw.fasta'
output_fasta_file = 'seqs.fasta'
protein_ids = []

records = SeqIO.parse(input_fasta_file, 'fasta')
renamed_records = []

for record in records:
    protein_id = record.id.split('|')[1]
    protein_ids.append(protein_id)
    
    record.id = protein_id
    record.name = protein_id
    record.description = protein_id
    renamed_records.append(record)

SeqIO.write(renamed_records, output_fasta_file, 'fasta')

protein_embeds = {}
for protein_id in protein_ids:
    protein_embeds[protein_id] = torch.load(f'./seq_embeds/{protein_id}.pt')['representations'][33]

for data in dataset:
    protein_id = data.protein_id
    data.protein_embed = protein_embeds[protein_id]


# Create the node and edge features
for data in dataset:
    mol = Chem.MolFromSmiles(data.smiles)
    ringinfo = mol.GetRingInfo()
    
    x = []
    atoms = [atom for atom in mol.GetAtoms()] # type: ignore

    x.append(one_hot(
        torch.tensor([x_map['atomic_num'].index(atom.GetAtomicNum()) for atom in atoms]),
        num_classes=len(x_map['atomic_num']), dtype=torch.float)
    )
    x.append(one_hot(
        torch.tensor([x_map['degree'].index(atom.GetTotalDegree()) for atom in atoms]),
        num_classes=len(x_map['degree']), dtype=torch.float)
    )
    x.append(one_hot(
        torch.tensor([x_map['num_hs'].index(atom.GetTotalNumHs()) for atom in atoms]),
        num_classes=len(x_map['num_hs']), dtype=torch.float)
    )
    x.append(one_hot(
        torch.tensor([x_map['hybridization'].index(str(atom.GetHybridization())) for atom in atoms]),
        num_classes=len(x_map['hybridization']), dtype=torch.float)
    )
    x.append(
        torch.tensor([[x_map['is_aromatic'].index(atom.GetIsAromatic())] for atom in atoms], dtype=torch.float)
    )
    x.append(
        torch.tensor([[x_map['is_in_ring'].index(atom.IsInRing())] for atom in atoms], dtype=torch.float)
    )
    x.append(one_hot(
        torch.tensor([x_map['formal_charge'].index(atom.GetFormalCharge()) for atom in atoms]),
        num_classes=len(x_map['formal_charge']), dtype=torch.float)
    )
    x.append(one_hot(
        torch.tensor([x_map['implicit_valence'].index(atom.GetImplicitValence()) for atom in atoms]),
        num_classes=len(x_map['implicit_valence']), dtype=torch.float)
    )
    x.append(one_hot(
        torch.tensor([x_map['num_radical_electrons'].index(atom.GetNumRadicalElectrons()) for atom in atoms]),
        num_classes=len(x_map['num_radical_electrons']), dtype=torch.float)
    )
    x.append(one_hot(
        torch.tensor([x_map['num_atom_rings'].index(ringinfo.NumAtomRings(atom.GetIdx())) for atom in atoms]),
        num_classes=len(x_map['num_atom_rings']), dtype=torch.float)
    )
    x.append(
        torch.tensor([[x_map['is_in_ring_3'].index(ringinfo.IsAtomInRingOfSize(atom.GetIdx(), 3))] for atom in atoms], dtype=torch.float)
    )
    x.append(
        torch.tensor([[x_map['is_in_ring_4'].index(ringinfo.IsAtomInRingOfSize(atom.GetIdx(), 4))] for atom in atoms], dtype=torch.float)
    )
    x.append(
        torch.tensor([[x_map['is_in_ring_5'].index(ringinfo.IsAtomInRingOfSize(atom.GetIdx(), 5))] for atom in atoms], dtype=torch.float)
    )
    x.append(
        torch.tensor([[x_map['is_in_ring_6'].index(ringinfo.IsAtomInRingOfSize(atom.GetIdx(), 6))] for atom in atoms], dtype=torch.float)
    )
    x.append(
        torch.tensor([[x_map['is_in_ring_7'].index(ringinfo.IsAtomInRingOfSize(atom.GetIdx(), 7))] for atom in atoms], dtype=torch.float)
    )
    x.append(
        torch.tensor([[x_map['is_in_ring_8'].index(ringinfo.IsAtomInRingOfSize(atom.GetIdx(), 8))] for atom in atoms], dtype=torch.float)
    )

    x = torch.cat(x, dim=1)
    data.x = x

    edge_index = torch.tensor(Chem.GetAdjacencyMatrix(mol), dtype=torch.long).to_sparse().indices()
    edge_attr = []

    for i in range(edge_index.shape[1]):
        bond = mol.GetBondBetweenAtoms(edge_index[0][i].item(), edge_index[1][i].item())
        edge_attr.append(e_map['bond_type'].index(str(bond.GetBondType())))
    
    edge_attr = torch.tensor(edge_attr, dtype=torch.long)
    edge_attr =  one_hot(edge_attr, num_classes=len(e_map['bond_type']), dtype=torch.float)
    data.edge_index = edge_index
    data.edge_attr = edge_attr


# Split the dataset
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# Train the model
model = Net()
num_epochs = 10
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
        optimizer.zero_grad()
        logits = model(batch)
        loss = loss_fn(logits, batch.y_class)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * len(batch)

    avg_train_loss = train_loss / len(train_loader.dataset)
    
    model.eval()
    test_loss = 0.0
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            logits = model(batch)
            loss = loss_fn(logits, batch.y_class)
            test_loss += loss.item() * len(batch)
            
            _, preds = logits.max(dim=1)
            num_correct += (preds == batch.y_class).sum().item()
            num_samples += batch.y_class.size(0)

    avg_test_loss = test_loss / len(test_loader.dataset)
    acc = num_correct / num_samples

    print(f'Epoch: {epoch+1}/{num_epochs}, '
          f'Train Loss: {avg_train_loss:.4f}, '
          f'Test Loss: {avg_test_loss:.4f}, '
          f'Accuracy: {acc:.4f}')
    print()

