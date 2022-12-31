import argparse
from features import FCN50, Dino
import h5py
import numpy as np
import os
import pickle
import math
import cv2
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.io.image import read_image
from PIL import Image
from autolabel.utils import Scene
from sklearn import decomposition
from tqdm import tqdm
from PIL import Image
import tinycudann as tcnn

class Autoencoder(nn.Module):

    def __init__(self, in_features, bottleneck):
        super().__init__()
        self.encoder = tcnn.Network(n_input_dims=in_features,
                                    n_output_dims=bottleneck,
                                    network_config={
                                        "otype": "CutlassMLP",
                                        "activation": "ReLU",
                                        "output_activation": "ReLU",
                                        "n_neurons": 128,
                                        "n_hidden_layers": 1
                                    })
        self.decoder = tcnn.Network(n_input_dims=bottleneck,
                                    n_output_dims=in_features,
                                    network_config={
                                        "otype": "CutlassMLP",
                                        "activation": "ReLU",
                                        "output_activation": "None",
                                        "n_neurons": 128,
                                        "n_hidden_layers": 1
                                    })

    def forward(self, x, p=0.1):
        code = self.encoder(x)
        out = self.decoder(F.dropout(code, 0.1))
        return out, code


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--autoencode', action='store_true')
    return parser.parse_args()


def compress_features(features, dim):
    N, H, W, C = features.shape
    coder = Autoencoder(C, dim).cuda()
    optimizer = torch.optim.Adam(coder.parameters(), lr=1e-3)
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(features.reshape(N * H * W, C)))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=2048, shuffle=True)  #changed batch size from 2048
    for _ in range(5):
        bar = tqdm(loader)
        for batch in bar:
            batch = batch[0].cuda()
            reconstructed, code = coder(batch)
            loss = F.mse_loss(reconstructed,
                              batch) + 0.01 * torch.abs(code).mean()
            bar.set_description(f"Loss: {loss.item()}")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    with torch.inference_mode():
        features_out = np.zeros((N, H, W, dim), dtype=np.float16)
        for i, feature in enumerate(features):
            feature = torch.tensor(feature).view(H * W, C).cuda()
            _, out = coder(feature.view(H * W, C))
            features_out[i] = out.detach().cpu().numpy().reshape(H, W, dim)
    return features_out


def extract_features(extractor, output_file, flags):
    root = "/home/asl-student/Desktop/Smazzucco/ssn-pytorch/BSR/BSDS500/data/images/train"

    paths = os.listdir(root)
    paths.remove('Thumbs.db')
    paths = [os.path.join(root, path) for path in paths]
    extracted = []
    len_paths = len(paths)
    
    with torch.inference_mode():
        batch_size = 2
        for i in tqdm(range(int(len_paths/batch_size))):
            if i == 100:
                break
            else:
                batch = paths[i * batch_size:(i + 1) * batch_size]
                image = torch.stack([read_image(p) for p in batch]).cuda()
                image = F.interpolate(image, [720, 960])
                features = extractor(image / 255.)

                extracted += [f for f in features]
    extracted = np.stack(extracted)

    if flags.autoencode:
        features = compress_features(extracted, flags.dim)
    else:
        features = extracted[:, :, :, :flags.dim]

    dataset = output_file.create_dataset(
        'dino', (len(features), *extractor.shape, flags.dim),
        dtype=np.float16,
        compression='lzf')
    dataset[:] = features

    N, H, W, C = dataset[:].shape
    X = dataset[:].reshape(N * H * W, C)
    pca = decomposition.PCA(n_components=3)
    indices = np.random.randint(0, X.shape[0], size=50000)
    subset = X[indices]
    transformed = pca.fit_transform(subset)
    minimum = transformed.min(axis=0)
    maximum = transformed.max(axis=0)
    diff = maximum - minimum

    dataset.attrs['pca'] = np.void(pickle.dumps(pca))
    dataset.attrs['min'] = minimum
    dataset.attrs['range'] = diff

def visualize_features(features):
    pca = pickle.loads(features.attrs['pca'].tobytes())
    N, H, W, C = features[:].shape

    from matplotlib import pyplot
    feature_maps = features[:]
    for i, fm in enumerate(feature_maps[::10]):
        mapped = pca.transform(fm.reshape(H * W, C)).reshape(H, W, 3)
        normalized = np.clip(
            (mapped - features.attrs['min']) / features.attrs['range'], 0, 1)
        pyplot.imsave(
            f"/home/asl-student/Desktop/Smazzucco/ssn-pytorch/BSR/img{i}.png",
            normalized)


def main():
    flags = read_args()
    path = "/home/asl-student/Desktop/Smazzucco/ssn-pytorch/BSR/BSDS500/data/images/"
    output_file = h5py.File(os.path.join(path, 'features.hdf'),
                            'w',
                            libver='latest')
    group = output_file.create_group('features')

    extractor = Dino()

    extract_features(extractor, group, flags)

    #visualize_features(group['dino'])


if __name__ == "__main__":
    main()
