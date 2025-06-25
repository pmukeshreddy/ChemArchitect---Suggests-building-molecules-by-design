class MolecularDataset(Dataset):
    def __init__(self,smiles_list,applications_list,tokenizer,property_encoder,max_lenght=128):
        self.tokenizer = tokenizer
        self.property_encoder = property_encoder
        self.max_lenght = max_lenght
        #filter smiles
        valid_data = []
        for smiles , apps in zip(smiles_list,applications_list):
            if self.is_valid_smiles(smiles):
                valid_data.append((smiles , apps))
        self.data = valid_data
    def is_valid_smiles(self, smiles):
        """Check if SMILES string is valid"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None and len(smiles) > 2 and len(smiles) < 120
        except:
            return False
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        smiles , applications = self.data[idx]
        tokens = self.tokenizer.encoder(smiles)
        properties = self.property_encoder.encode_application(applications)
        return {
            'tokens': tokens,
            'properties': properties,
            'smiles': smiles
        }
