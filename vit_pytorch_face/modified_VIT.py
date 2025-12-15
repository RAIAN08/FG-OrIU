import torch
from torch import nn


class ModifiedViT(nn.Module):
    """
    Modified ViT model to return both classification output and embeddings

    Output shape: torch.Size([bs, 1000])
    Embeddings shape: torch.Size([bs, 768])
    """

    def __init__(self, vit_model):
        super(ModifiedViT, self).__init__()
        # Split the original ViT model and keep the submodules
        self.conv_proj = vit_model.conv_proj  # Convolutional projection
        self._process_input = vit_model._process_input  # Patch embedding
        self.class_token = vit_model.class_token

        # self.encoder_pos_embedding = vit_model.encoder.pos_embedding  # Transformer encoder
        # self.encoder_dropout = vit_model.encoder.dropout
        # self.encoder_ln = vit_model.encoder.ln

        self.encoder = vit_model.encoder  # Transformer encoder
        self.heads = vit_model.heads  # Classification head

    def forward(self, x, label, prepare_feature=False, training=False):
        # label is not used in this model
        # Patch embedding
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to match the batch size
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        if prepare_feature:
            features = {}
            idx = 0

            x = x + self.encoder.pos_embedding
            x = self.encoder.dropout(x)

            for block in self.encoder.layers:
                # attn
                residual = x
                x = block.ln_1(x)  
                x, _ = block.self_attention(x, x, x, need_weights=False)  
                x = block.dropout(x)
                x = x + residual  

                if not training:
                    features[f"layer_{idx}_attn"] = x.detach().cpu()
                else:
                    features[f"layer_{idx}_attn"] = x

                residual = x
                x = block.ln_2(x)  
                x = block.mlp(x)  
                x = x + residual  

                if not training:
                    features[f"layer_{idx}_ff"] = x.detach().cpu()
                else:
                    features[f"layer_{idx}_ff"] = x

                idx += 1

            x = self.encoder.ln(x)
        else:
            features = None
            x = self.encoder(x)  # Transformer encoder

        embeddings = x[:, 0]  # [CLS] token's embedding
        output = self.heads(
            embeddings
        )  # The classification head calculates the final output

        if label is not None:
            # x = self.loss(emb, label)
            if prepare_feature:
                return output, features, embeddings
            else:
                return output, embeddings
        else:
            return embeddings
        

        return output, embeddings
