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

class TransformerSentenceGenerator(nn.Module):
    def __init__(self, num_transformer_layers, num_heads, vocab_size, input_length, hidden_dim, patch_dim, ):
        super(TransformerSentenceGenerator, self).__init__()

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
            y: 画像の特徴量(batch_size, num_patch, hidden_dim)
        """
        
        x = self.embedding(x)
        y = self.fc_1(y)

        for SA, CA in zip(self.self_attention_blocks, self.cross_attention_blocks):
            x, _ = SA(x)
            x, _ = CA(x, y)

        x = self.fc_2(x)

        return x

def GUMI_T_generate_ohgiri(vit, image_feature_extractor, generator, image_paths, tokenizer, sentence_length, argmax = False, k = 5, temp = 1.0):
    """
        image_paths: 画像のパスのリスト
    """

    device = next(generator.parameters()).device

    images = [Image.open(path).convert("RGB") for path in image_paths]
    tmp_images = image_feature_extractor(images, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = vit(**tmp_images)
    image_features = outputs.last_hidden_state[:, 1:, :]

    gen_texts = torch.ones(size = (len(image_paths), 1)).to(torch.int32).to(device)

    for i in range(0, sentence_length):
        tmp_texts = F.pad(gen_texts, (0, sentence_length - i), value = 0).to(torch.int32).to(device)
        outputs = generator(tmp_texts, image_features)

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

### HuggingFace用の設定

class GUMI_T_Config(PretrainedConfig):
    model_type = "gumi_t"

    def __init__(self, 
            num_transformer_layers = 1, 
            num_heads = 8, 
            vocab_size = 17363, 
            input_length = 32, 
            hidden_dim = 1024, 
            patch_dim = 768, **kwargs):
        super().__init__(**kwargs)

        self.num_transformer_layers = num_transformer_layers
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.input_length = input_length
        self.hidden_dim = hidden_dim
        self.patch_dim = patch_dim

class GUMI_T(PreTrainedModel):
    config_class = GUMI_T_Config
    
    def __init__(self, config):
        super(GUMI_T, self).__init__(config)

        self.input_length = config.input_length

        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.image_feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

        self.generator = TransformerSentenceGenerator(
            num_transformer_layers = config.num_transformer_layers,
            num_heads = config.num_heads,
            vocab_size = config.vocab_size, 
            input_length = config.input_length, 
            hidden_dim = config.hidden_dim, 
            patch_dim = config.patch_dim
        )
    
    def load_weights(self, weight_path):
        self.generator.load_state_dict(torch.load(weight_path))

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok = True)

        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        self.config.save_pretrained(save_directory)

    def forward(self, input_ids, image_features):
        """
            input_ids: 単語のIDのリスト(batch_size, input_length)
            image_features: 画像の特徴量のリスト(batch_size, 196, 768)
        """
        
        outputs = self.generator(input_ids, image_features, )
        
        return outputs # [batch_size, input_length, vocab_size]

    def generate(self, inputs, argmax = False, k = 5, temp = 1.0):
        device = next(self.generator.parameters()).device

        pil_images = inputs["pixel_values"]

        tmp_images = self.image_feature_extractor(pil_images, return_tensors = "pt").to(device)
        with torch.no_grad():
            outputs = self.vit(**tmp_images)
        image_features = outputs.last_hidden_state[:, 1:, :]

        gen_texts = torch.ones(size = (len(pil_images), 1)).to(torch.int32).to(device)

        for i in range(0, self.input_length - 1):
            tmp_texts = F.pad(gen_texts, (0, self.input_length - 1 - i), value = 0).to(torch.int32).to(device)
            outputs = self.generator(tmp_texts, image_features)

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
        
        return gen_texts
    
    def decode(self, gen_texts, tokenizer):
        """
            gen_texts: 生成された文章のIDのリスト(batch_size, input_length)
        """
        tmp_gen_texts = list()
        for G in gen_texts:
            tmp_gen_text = list()
            for I in G[1:]:
                if int(I) in [0, 1, 2]:
                    break
                tmp_gen_text.append( tokenizer.decode([int(I)]) )
            tmp_gen_texts.append("".join(tmp_gen_text))
        
        return tmp_gen_texts