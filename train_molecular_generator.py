def train_molecular_generator(model, train_loader, val_loader, tokenizer, 
                            num_epochs=50, learning_rate=1e-4, device='cuda'):
    """Train the molecular generator"""
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch in pbar:
            tokens = batch['tokens'].to(device)
            properties = batch['properties'].to(device)
            
            # Teacher forcing: input = tokens[:-1], target = tokens[1:]
            input_tokens = tokens[:, :-1]
            target_tokens = tokens[:, 1:]
            
            # Forward pass
            logits = model(input_tokens, properties)
            
            # Compute loss
            loss = criterion(logits.reshape(-1, model.vocab_size), 
                           target_tokens.reshape(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                tokens = batch['tokens'].to(device)
                properties = batch['properties'].to(device)
                
                input_tokens = tokens[:, :-1]
                target_tokens = tokens[:, 1:]
                
                logits = model(input_tokens, properties)
                loss = criterion(logits.reshape(-1, model.vocab_size), 
                               target_tokens.reshape(-1))
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')
        
        # Generate sample molecules
        if (epoch + 1) % 10 == 0:
            print("\nSample generations:")
            test_properties = torch.tensor([1, 0, 1, 0, 0, 0, 0], dtype=torch.float).to(device)
            for i in range(3):
                sample = model.generate(test_properties, tokenizer, temperature=0.8)
                is_valid = "✓" if is_valid_molecule(sample) else "✗"
                print(f"  {is_valid} {sample}")
            print()
    
    return train_losses, val_losses


def is_valid_molecule(smiles):
    """Check if generated SMILES is a valid molecule"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False
