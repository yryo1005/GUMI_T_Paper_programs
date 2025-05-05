import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTFeatureExtractor, ViTModel, PreTrainedModel, PretrainedConfig, PreTrainedTokenizerFast, AutoConfig, AutoModelForCausalLM


class SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, input_length, num_heads = 8):
        super(SelfAttentionBlock, self).__init__()

        self.hidden_dim = hidden_dim
        self.input_length = input_length

        self.layer_norm_1 = nn.LayerNorm(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            embed_dim = hidden_dim, 
            num_heads = num_heads, 
            batch_first = True
        )

        self.layer_norm_2 = nn.LayerNorm(hidden_dim)
        self.fc_1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)

        self.attn_mask = torch.triu(torch.ones(self.input_length, self.input_length), diagonal = 1).bool()
    
    def forward(self, x):
        attn_mask = self.attn_mask.to(x.device)

        z = self.layer_norm_1(x)
        z, attn_output_weights  = self.self_attention(z, z, z, attn_mask = attn_mask)
        z = x + z

        y = self.layer_norm_2(z)
        y = nn.LeakyReLU()( self.fc_1(y) )
        y = self.fc_2(y)
        y = y + z

        return y, attn_output_weights

class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads = 8):
        super(CrossAttentionBlock, self).__init__()

        self.hidden_dim = hidden_dim

        self.layer_norm_1 = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim = hidden_dim,
            num_heads = num_heads,
            batch_first = True
        )

        self.layer_norm_2 = nn.LayerNorm(hidden_dim)
        self.fc_1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.LeakyReLU()

    def forward(self, x, y):
        """
        x: Decoder側の入力 (batch_size, seq_len, hidden_dim)
        y: Encoder側の出力 (batch_size, num_patch, hidden_dim)
        """

        z = self.layer_norm_1(x)
        z, attn_output_weights = self.cross_attention(
            query = z,
            key = y,
            value = y
        )
        z = x + z

        y = self.layer_norm_2(z)
        y = self.activation(self.fc_1(y))
        y = self.fc_2(y)
        y = y + z

        return y, attn_output_weights

class TransformerSentenceGeneratorViTFC(nn.Module):
    def __init__(self, num_transformer_layers, num_heads, vocab_size, input_length, hidden_dim, patch_dim, ):
        super(TransformerSentenceGeneratorViTFC, self).__init__()

        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.fc_1 = nn.Linear(patch_dim, hidden_dim)

        self.self_attention_blocks = nn.ModuleList([
            SelfAttentionBlock(hidden_dim, input_length, num_heads) for _ in range(num_transformer_layers)
        ])
        self.cross_attention_blocks = nn.ModuleList([
            CrossAttentionBlock(hidden_dim) for _ in range(num_transformer_layers)
        ])
        
        self.fc_2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, y):
        """
            x: 文章(batch_size, seq_len)
            y: 画像["pixel_values"](batch_size, num_patch, hidden_dim)
        """
        
        x = self.embedding(x)

        y = self.vit(**y)
        y = y.last_hidden_state[:, 1:, :] # CLSトークンを除外
        y = self.fc_1(y)

        for SA, CA in zip(self.self_attention_blocks, self.cross_attention_blocks):
            x, _ = SA(x)
            x, _ = CA(x, y)

        x = self.fc_2(x)

        return x

def GUMI_T_ViTFC_generate_ohgiri(image_preprocesser, generator, image_paths, tokenizer, sentence_length, argmax = False, k = 5, temp = 1.0):
    """
        image_paths: 画像のパスのリスト
    """

    device = next(generator.parameters()).device

    images = [Image.open(path).convert("RGB") for path in image_paths]
    preprocessed_images = image_preprocesser(images, return_tensors="pt").to(device)

    gen_texts = torch.ones(size = (len(image_paths), 1)).to(torch.int32).to(device)

    for i in range(0, sentence_length):
        tmp_texts = F.pad(gen_texts, (0, sentence_length - i), value = 0).to(torch.int32).to(device)
        outputs = generator(tmp_texts, preprocessed_images)

        logits = outputs[:, i, :]

        if argmax:
            gathered_indices = torch.argmax(logits, dim = -1, keepdim = True)
        else:  
            probs = F.softmax(logits / temp, dim = -1)
            top_k_probs, top_k_indices = torch.topk(probs, k = k, dim = -1)
            top_k_probs = top_k_probs / top_k_probs.sum(dim = -1, keepdim = True) 
            chosen_indices = torch.multinomial(top_k_probs, 1).squeeze(-1)
            gathered_indices = top_k_indices.gather(-1, chosen_indices.unsqueeze(-1))

        gen_texts = torch.cat([gen_texts, gathered_indices], dim = 1)
    
    fig = plt.figure(figsize = (20, 20))
    for i, (I, IP) in enumerate(zip(images, image_paths)):
        IP = IP.split("/")[-1]
        ax = fig.add_subplot(1, len(image_paths), i + 1)
        ax.imshow(I)
        ax.axis("off")
        ax.set_title(f"{i}: {IP}")
    plt.show()
    
    tmp_gen_texts = list()
    for G in gen_texts:
        tmp_gen_text = list()
        for I in G[1:]:
            if int(I) in [0, 1, 2]:
                break
            tmp_gen_text.append( tokenizer.decode([int(I)]) )
        tmp_gen_texts.append("".join(tmp_gen_text))
    
    for i, GT in enumerate(tmp_gen_texts):
        print(f"{i}: {GT}")

