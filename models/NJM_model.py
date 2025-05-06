import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import PreTrainedModel, PretrainedConfig, PreTrainedTokenizerFast, AutoConfig, AutoModelForCausalLM

class LSTMSentenceGenerator(nn.Module):
    def __init__(self, vocab_size, input_length, hidden_dim, image_feature_dim, ):
        super(LSTMSentenceGenerator, self).__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.fc_1 = nn.Linear(image_feature_dim, hidden_dim)

        self.lstm = nn.LSTM(
            input_size = hidden_dim,
            hidden_size = hidden_dim,
            batch_first = True,
        )
        
        self.fc_2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, y):
        """
            x: 文章(batch_size, seq_len)
            y: 画像の特徴量(batch_size, image_feature_dim)
        """
        
        x = self.embedding(x)
        y = self.fc_1(y)

        h_0 = y.unsqueeze(0)  # (1, batch_size, hidden_dim)
        c_0 = torch.zeros_like(h_0)  # (1, batch_size, hidden_dim)
        x, _ = self.lstm(x, (h_0, c_0))


        x = self.fc_2(x)

        return x

def NJM_generate_ohgiri(resnet152, image_preprocesser, generator, image_paths, tokenizer, sentence_length, argmax = False, k = 5, temp = 1.0):
    """
        image_paths: 画像のパスのリスト
    """

    device = next(generator.parameters()).device

    images = [Image.open(path).convert("RGB") for path in image_paths]
    preprocessed_images = torch.stack([image_preprocesser(I) for I in images]).to(device)
    
    with torch.no_grad():
        image_features = resnet152(preprocessed_images)

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

class NJM_Config(PretrainedConfig):
    model_type = "NJM"
    def __init__(self, 
                 vocab_size = 17363, 
                 input_length = 32,
                 hidden_dim = 1024,
                 image_feature_dim = 2048,
                 **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.input_length = input_length
        self.hidden_dim = hidden_dim
        self.image_feature_dim = image_feature_dim

class NJM(PreTrainedModel):
    config_class = NJM_Config

    def __init__(self, config):
        super().__init__(config)

        self.input_length = config.input_length

        self.image_preprocesser = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(  # ImageNetの平均と標準偏差で正規化
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        resnet152 = models.resnet152(pretrained=True)
        self.resnet152 = torch.nn.Sequential(*list(resnet152.children())[:-1], nn.Flatten(start_dim = 1))

        self.generator = LSTMSentenceGenerator(
            vocab_size = config.vocab_size, 
            input_length = config.input_length, 
            hidden_dim = config.hidden_dim, 
            image_feature_dim = config.image_feature_dim
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
            image_features: 画像の特徴量のリスト(batch_size, 2048)
        """
        
        outputs = self.generator(input_ids, image_features, )
        
        return outputs # [batch_size, input_length, vocab_size]
    
    def generate(self, inputs, argmax = False, k = 5, temp = 1.0):
        device = next(self.generator.parameters()).device

        pil_images = inputs["pixel_values"]

        preprocessed_images = torch.stack([self.image_preprocesser(I) for I in pil_images]).to(device)
        with torch.no_grad():
            image_features = self.resnet152(preprocessed_images)

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