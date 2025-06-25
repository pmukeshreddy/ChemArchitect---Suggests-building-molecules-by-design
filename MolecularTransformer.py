class MolecularTransformer(nn.Module):
    def __init__(self,vocab_size,property_dim,d_model=256, nhead=8, 
                 num_layers=6, max_length=128):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.property_dim = property_dim
        self.max_length = max_length

        #embeddings
        self.token_embeddings = nn.Embedding(vocab_size,d_model)
        self.pos_encoding = PostionalEnocding(d_model)

        #property conditioning
        self.property_proj = nn.Linear(property_dim,d_model)

        #transformer decoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=d_model*4,dropout=0.1,batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer,num_layers)
        #output
        self.output_projection = nn.Linear(d_model,vocab_size)
        #initize weights
        self._init_weights()
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module,nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module,nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
    def create_casual_mask(self,seq_len):
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        return mask.bool()
        
    def forward(self,tokens,properties,target_tokens=None):
        batch_size , seq_len = tokens.shape
        #token embeddings
        token_emb = self.token_embeddings(tokens)*np.sqrt(self.d_model)
        token_emb = self.pos_encoding(token_emb)
        #property embeddings
        prop_embed = self.property_proj(properties).unsqueeze(1)
        prop_embed = prop_embed.expand(-1,seq_len,-1)
        #combined embeddings
        combined_emb = token_emb + prop_embed
        #create casual mask
        causal_mask = self.create_causal_mask(seq_len).to(tokens.device)
        #transformer encoder
        output = self.transformer(combined_emb,src_mask=causal_mask)
        logits = self.output_projection(output)
        return logits
