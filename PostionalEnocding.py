class PostionalEnocding(nn.Module):
    def __init__(self,d_model,max_length=512):
        super().__init__()
        
        pe = torch.zeros(max_length,d_model)
        position = torch.arrange(0,max_length).unsqueeze(1).float()

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(np.log(10000.0) / d_model))

        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)

        self.register_buffer('pe', pe)
    def forward(self,x):
        return x + self.pe[:x.size(1),:].unsqueeze(0)
