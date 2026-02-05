import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len):
        super(SinusoidalPositionalEmbedding, self).__init__()
        embedding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        embedding[:, 0::2] = torch.sin(position * div_term)
        embedding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('embedding', embedding)

    def forward(self, x):
        return x + self.embedding[:x.shape[1]].unsqueeze(0)

class Connector(nn.Module):
    def __init__(
            self,
            in_feature_dim,
            out_proj_dim,
            num_layers=2,
            attn_heads=16,
            hidden_size=1024,
            intermediate_size=4096,
            downsampling_factor=6,
            dtype=torch.bfloat16,
            norm_first=True,
            use_positional_embeddings=False,
            dropout=0.1,
            **kwargs
        ):
        self.in_feature_dim = in_feature_dim
        self.out_proj_dim = out_proj_dim
        self.num_layers = num_layers
        self.attn_heads = attn_heads
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.downsampling_factor = downsampling_factor
        self.dtype = dtype
        self.norm_first = norm_first
        self.use_positional_embeddings = use_positional_embeddings
        self.dropout = dropout

        super(Connector, self).__init__()

        # Construct the transformer encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=self.attn_heads,
                dim_feedforward=self.intermediate_size,
                dropout=dropout,
                activation='gelu',
                dtype=self.dtype,
                batch_first=True,
                norm_first=self.norm_first,
            ),
            num_layers=self.num_layers
        )

        if self.use_positional_embeddings:
            self.positional_embeddings = SinusoidalPositionalEmbedding(
                d_model=self.hidden_size,
                max_len=300,
            )

        self.in_projection = nn.Linear(self.in_feature_dim * self.downsampling_factor, self.hidden_size, dtype=self.dtype)
        self.attention_pooling = nn.MaxPool1d(downsampling_factor, stride=downsampling_factor)

        self.lm_projection = nn.Linear(self.hidden_size, self.out_proj_dim, dtype=self.dtype)

    def stacking_downsampler(self, embeds, attention_mask=None):
        mod = embeds.shape[-2] % self.downsampling_factor
        if mod != 0:
            # append zeros to both the embeddings and the mask if the sequences are not divisible
            # by downsampling_factor
            appendix = torch.zeros((embeds.shape[0], self.downsampling_factor - mod,
                                    embeds.shape[-1]), device=embeds.device)
            embeds = torch.hstack((embeds, appendix))

            if attention_mask is not None:
                mask_appendix = attention_mask[...,-1].unsqueeze(1).repeat(1, self.downsampling_factor - mod)
                attention_mask = torch.cat((attention_mask, mask_appendix), dim=1)

        # perform the stacking downsampling
        embeds = embeds.contiguous().view(
            embeds.shape[0],
            embeds.shape[1] // self.downsampling_factor,
            embeds.shape[2] * self.downsampling_factor
        )

        # downsample the attention mask too
        if attention_mask is not None:
            attention_mask = self.attention_pooling(attention_mask.float()).long()
        else:
            attention_mask = None
        
        return embeds, attention_mask
    
    def forward(self, x, attention_mask=None):
        # downsample the input features
        x, attention_mask = self.stacking_downsampler(x, attention_mask=attention_mask)

        x = self.in_projection(x)

        if self.use_positional_embeddings:
            x = self.positional_embeddings(x)

        x = self.transformer(x, src_key_padding_mask=attention_mask.bool())

        x = self.lm_projection(x)

        return x, attention_mask

