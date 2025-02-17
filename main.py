import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Generator の定義（Colab の実装と同一）
class Generator(nn.Module):
    def __init__(self, latent_dim, n_classes, img_size=28):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.init_size = img_size // 4  # 28//4 = 7
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim + n_classes, 128 * self.init_size ** 2)
        )
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),  # 7 -> 14
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),  # 14 -> 28
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh()  # 出力は [-1, 1]
        )

    def forward(self, noise, labels):
        label_input = self.label_emb(labels)
        gen_input = torch.cat((noise, label_input), -1)
        out = self.l1(gen_input)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# ハイパーパラメータ
latent_dim = 10
n_classes = 10
img_size = 28

# Streamlit アプリのタイトルと入力フォーム
st.title("Conditional WGAN-GP による MNIST 画像生成")
st.write("生成したい数字（0～9）を入力してください。")

# ユーザー入力（テキスト入力でもOKですが、ここでは selectbox を使用）
target_digit = st.selectbox("数字を選択", list(range(10)))
n_images = st.number_input("生成する画像の枚数", min_value=1, max_value=20, value=5)

# モデルの初期化と重みのロード
@st.cache_resource(show_spinner=False)
def load_generator(model_path="generator.pth"):
    model = Generator(latent_dim, n_classes, img_size).to("cpu")
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

generator = load_generator()

# 画像生成ボタン
if st.button("画像生成"):
    # ランダムノイズ生成（n_images 枚）
    z = torch.randn(n_images, latent_dim)
    # 指定されたラベル（全て target_digit に固定）
    labels = torch.full((n_images,), target_digit, dtype=torch.long)
    # 画像生成
    with torch.no_grad():
        gen_imgs = generator(z, labels)
    # 生成画像のテンソルは [-1, 1] の範囲なので [0, 1] に変換
    gen_imgs = gen_imgs.cpu().numpy()
    gen_imgs = 0.5 * (gen_imgs + 1)
    
    # 画像を横に並べて表示するためのプロット
    cols = n_images
    fig, axs = plt.subplots(1, cols, figsize=(2 * cols, 2))
    if n_images == 1:
        axs = [axs]
    for i, ax in enumerate(axs):
        # 生成画像は (1, 28, 28) なので squeeze() でチャネル次元を除去
        img = gen_imgs[i].squeeze()
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    st.pyplot(fig)
