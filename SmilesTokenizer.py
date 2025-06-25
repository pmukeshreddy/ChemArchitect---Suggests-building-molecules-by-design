class SmilesTokenizer:
    def __init__(self):
        self.special_tokens = ['<pad>', '<start>', '<end>', '<unk>']
        self.atom_tokens = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']
        self.bond_tokens = ['-', '=', '#', ':']
        self.structure_tokens = ['(', ')', '[', ']', '+', '-', '@', '/', '\\']
        self.number_tokens = [str(i) for i in range(10)]

        all_tokens = (self.special_tokens+self.atom_tokens+self.bond_tokens+self.structure_tokens+self.number_tokens)

        self.char_to_idx = {char: idx for idx, char in enumerate(all_tokens)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(all_tokens)

        self.unk_idx = self.char_to_idx['<unk>']
        self.pad_idx = self.char_to_idx['<pad>']
        self.start_idx = self.char_to_idx['<start>']
        self.end_idx = self.char_to_idx['<end>']

    def expan_vocab(self,smiles_list):
        all_chars = set()
        for smiles in smiles_list:
            all_chars.update(list(smiles))
        new_chars = all_chars - self.char_to_idx(self.char_to_idx.keys())

        for char in new_chars:
            if char not in self.char_to_idx:
                idx = len(self.char_to_idx)
                self.char_to_idx[char] = idx
                self.idx_to_char[idx] = char
                
        self.vocab_size = len(self.char_to_idx)
        print(f"Vocabulary expanded to {self.vocab_size} tokens")
        print(f"New characters added: {new_chars}")

    def encode(self,smiles,max_lenght=128):
        tokens = ["<start>"] + list(smiles) + ["<end>"]

        indicies = []
        for token in tokens:
            if token in self.char_to_idx:
                indicies.append(self.char_to_idx[token])
            else:
                indicies.append(self.unk_idx)

        if len(tokens)>max_lenght:
            indices = indicies[:max_lenght]
            indices[-1] = self.end_idx
        else:
            indices.extend([self.pad_idx]*(max_lenght-len(indicies)))
        return torch.tensor(indices, dtype=torch.long)
        
    def decode(self,indices):
        chars = []
        for idx in indices:
            if idx == self.pad_idx:
                break
            if idx == self.end_idx:
                break
            if idx == self.start_idx:
                continue
            chars.append(self.idx_to_char[idx])
        return "".join(chars)
        
