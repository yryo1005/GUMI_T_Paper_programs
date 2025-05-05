import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import PreTrainedModel, PretrainedConfig, PreTrainedTokenizerFast, AutoConfig, AutoModelForCausalLM

class ImageEncoder(nn.Module):
    def __init__(self, feature_dim = 16384):
        super(ImageEncoder, self).__init__()
        self.feature_dim = feature_dim
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 2, padding = 1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size = 3, stride = 2, padding = 1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size = 3, stride = 2, padding = 1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size = 3, stride = 2, padding = 1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size = 3, stride = 2, padding = 1)

        self.fc1 = nn.Linear(256 * 4 * 4, self.feature_dim)
        # self.fc2 = nn.Linear(self.feature_dim, self.feature_dim)

    def forward(self, x):
        """
            x : (batch_size, 3, 128, 128)
        """
        
        x = F.leaky_relu(self.conv1(x))
        # (batch_size, 64, 64, 64)

        x = F.leaky_relu(self.conv2(x))
        # (batch_size, 128, 32, 32)

        x = F.leaky_relu(self.conv3(x))
        # (batch_size, 256, 16, 16) 

        x = F.leaky_relu(self.conv4(x))
        # (batch_size, 512, 8, 8)

        x = F.leaky_relu(self.conv5(x))
        # (batch_size, 1024, 4, 4)

        x = nn.Flatten()(x)
        # (batch_size, 1024 * 4 * 4)
        x = F.leaky_relu(self.fc1(x))
        # x = F.leaky_relu(self.fc2(x))

        return x

class ImageDecoder(nn.Module): 
    def __init__(self, feature_dim = 16384):
        super(ImageDecoder, self).__init__()
        self.feature_dim = feature_dim

        self.fc1 = nn.Linear(self.feature_dim, 256 * 4 * 4)

        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size = 3, stride = 2, padding = 1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 128, kernel_size = 3, stride = 2, padding = 1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size = 3, stride = 2, padding = 1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, 64, kernel_size = 3, stride = 2, padding = 1, output_padding=1)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size = 3, stride = 2, padding = 1, output_padding=1)
        self.conv1 = nn.Conv2d(32, 3, kernel_size = 3, stride = 1, padding = 1)

    def forward(self, x):
        """
            x : (batch_size, feature_dim)
        """
        
        x = F.leaky_relu(self.fc1(x))
        # (batch_size, 512 * 4 * 4)
        x = x.reshape(-1, 256, 4, 4)
        # (batch_size, 512, 4, 4)

        x = F.leaky_relu(self.deconv1(x))
        # (batch_size, 256, 8, 8)

        x = F.leaky_relu(self.deconv2(x))
        # (batch_size, 128, 16, 16) 

        x = F.leaky_relu(self.deconv3(x))
        # (batch_size, 64, 32, 32) 

        x = F.leaky_relu(self.deconv4(x))
        # (batch_size, 32, 64, 64)

        x = F.leaky_relu(self.deconv5(x))
        # (batch_size, 16, 128, 128)

        x = torch.sigmoid(self.conv1(x))
        # (batch_size, 3, 128, 128)

        return x

class Autoencoder(nn.Module):
    def __init__(self, feature_dim = 16384):
        super(Autoencoder, self).__init__()
        self.encoder = ImageEncoder(feature_dim)
        self.decoder = ImageDecoder(feature_dim)

    def forward(self, x):
        """
            x : (batch_size, 3, 128, 128)
        """
        
        x = self.encoder(x)
        x = self.decoder(x)

        return x

def autoencoder_predict(image_preprocesser, autoencoder, image_paths):

    device = next(autoencoder.parameters()).device

    images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
    images = [image_preprocesser(image) for image in images]
    preprocessed_images = torch.stack(images).to(device)

    with torch.no_grad():
        predicts = autoencoder(preprocessed_images)
    predicts = predicts.cpu().numpy()

    fig = plt.figure(figsize = (5 * len(images), 10))
    for i, (I, P) in enumerate(zip(images, predicts)):

        ax = fig.add_subplot(2, len(images), i + 1)
        ax.imshow(np.transpose(I, (1, 2, 0)))
        ax.axis("off")
        ax.set_title("Input")

        ax = fig.add_subplot(2, len(images), i + 1 + len(images))
        ax.imshow(np.transpose(P, (1, 2, 0)))
        ax.axis("off")
        ax.set_title("Predict")
    plt.show()

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

def GUMI_AE_generate_ohgiri(encoder, image_preprocesser, generator, image_paths, tokenizer, sentence_length, argmax = False, k = 5, temp = 1.0):
    """
        image_paths: 画像のパスのリスト
    """

    device = next(generator.parameters()).device

    images = [Image.open(path).convert("RGB") for path in image_paths]
    preprocessed_images = torch.stack([image_preprocesser(I) for I in images]).to(device)
    
    with torch.no_grad():
        image_features = encoder(preprocessed_images)

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

class GUMI_AE_Config(PretrainedConfig):
    model_type = "NJM"
    def __init__(self, 
                 vocab_size = 17363, 
                 input_length = 32,
                 hidden_dim = 1024,
                 image_feature_dim = 16384,
                 **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.input_length = input_length
        self.hidden_dim = hidden_dim
        self.image_feature_dim = image_feature_dim

class GUMI_AE(PreTrainedModel):
    def __init__(self, config):
        super(GUMI_AE, self).__init__(config)

        self.image_feature_dim = config.image_feature_dim
        self.input_length = config.input_length

        self.autoencoder = Autoencoder(config.image_feature_dim)
        self.image_preprocesser = transforms.Compose([
            transforms.Resize( (128, 128) ),
            transforms.ToTensor()
        ])
        self.encoder = self.autoencoder.encoder

        self.generator = LSTMSentenceGenerator(
            vocab_size = config.vocab_size, 
            input_length = config.input_length, 
            hidden_dim = config.hidden_dim, 
            image_feature_dim = config.image_feature_dim
        )
    
    def load_generator_weights(self, weight_path):
        self.generator.load_state_dict(torch.load(weight_path))
    
    def load_autoencoder_weights(self, weight_path):
        self.autoencoder.load_state_dict(torch.load(weight_path))
        self.encoder = self.autoencoder.encoder

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok = True)

        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        self.config.save_pretrained(save_directory)
    
    def forward(self, input_ids, image_features):
        """
            input_ids: 単語のIDのリスト(batch_size, input_length)
            image_features: 画像の特徴量のリスト(batch_size, 16384)
        """
        
        outputs = self.generator(input_ids, image_features, )
        
        return outputs # [batch_size, input_length, vocab_size]
    
    def generate(self, inputs, argmax = False, k = 5, temp = 1.0):
        device = next(self.generator.parameters()).device

        pil_images = inputs["pixel_values"]

        preprocessed_images = torch.stack([self.image_preprocesser(I) for I in pil_images]).to(device)
        with torch.no_grad():
            image_features = self.encoder(preprocessed_images)

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