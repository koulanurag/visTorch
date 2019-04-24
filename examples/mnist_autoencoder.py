import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from visTorch import visboard
from torchvision.utils import save_image
from torchvision.utils import make_grid
from PIL import Image


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.latent_size = 12

        self._encoder = nn.Sequential(nn.Linear(28 * 28, 128),
                                      nn.Tanh(),
                                      nn.Linear(128, 64),
                                      nn.Tanh(),
                                      nn.Linear(64, self.latent_size),
                                      nn.Tanh())
        self._decoder = nn.Sequential(nn.Linear(self.latent_size, 64),
                                      nn.Tanh(),
                                      nn.Linear(64, 128),
                                      nn.Tanh(),
                                      nn.Linear(128, 28 * 28),
                                      nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encoder(self, x):
        return self._encoder(x)

    def decoder(self, x):
        return self._decoder(x)


def min_max_normalization(tensor, min_value, max_value):
    min_tensor = tensor.min()
    tensor = (tensor - min_tensor)
    max_tensor = tensor.max()
    tensor = tensor / max_tensor
    tensor = tensor * (max_value - min_value) + min_value
    return tensor


def tensor_round(tensor):
    return torch.round(tensor)


def process_to_img(x):
    grid = make_grid(x)
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    return im


def to_img(x):
    x = x.view(x.size(0), 1, 28, 28)
    return x


def add_noise(img):
    noise = torch.randn(img.size()) * 0.4
    noisy_img = img + noise
    return noisy_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MNIST AutoEncoder Training and Latent Visualization')
    parser.add_argument('--train', action='store_true', default=False,
                        help='Train the AutoEncoder (default: %(default)s)')
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='Visualize / Interact with the latent Space (default: %(default)s)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size used for training (default: %(default)s)')
    parser.add_argument('--host', default='127.0.0.1',
                        help='IP address for hosting the visualization app (default: %(default)s)')
    parser.add_argument('--port', default='8051',
                        help='hosting port (default: %(default)s)')
    args = parser.parse_args()

    # create relative paths
    dataset_dir = os.path.join(os.getcwd(), 'dataset')
    result_dir = os.path.join(os.getcwd(), 'results', 'mnist')
    ae_path = os.path.join(result_dir, 'ae_model.p')
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    # create model
    ae_model = AutoEncoder()

    # dataset
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda tensor: min_max_normalization(tensor, 0, 1)),
        transforms.Lambda(lambda tensor: tensor_round(tensor))

    ])

    dataset = MNIST(dataset_dir, download=True, transform=img_transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    if args.train:
        num_epochs = 100
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(
            ae_model.parameters(), lr=1e-3, weight_decay=1e-5)

        for epoch in range(num_epochs):
            for data in dataloader:
                img, _ = data
                img = img.view(img.size(0), -1)
                noisy_img = add_noise(img)
                # ===================forward=====================
                output = ae_model(noisy_img)
                loss = criterion(output, img)
                MSE_loss = nn.MSELoss()(output, img)
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # ===================log========================
            print('epoch [{}/{}], loss:{:.4f}, MSE_loss:{:.4f}'
                  .format(epoch + 1, num_epochs, loss.item(), MSE_loss.item()))
            if epoch % 10 == 0:
                x = to_img(img.cpu().data)
                x_hat = to_img(output.cpu().data)
                save_image(x, os.path.join(result_dir, 'x_{}.png'.format(epoch)))
                save_image(x_hat, os.path.join(result_dir, 'x_hat{}.png'.format(epoch)))

            torch.save(ae_model.state_dict(), ae_path)

    if args.visualize:
        ae_model.load_state_dict(torch.load(ae_path))  # load pre-trained model

        # initialize visualization app
        vis_board = visboard()
        vis_board.add_ae(ae_model,
                         dataset,
                         latent_options={'n': ae_model.latent_size, 'min': -1, 'max': 1, 'step': 0.05},
                         model_paths={'Best': ae_path},
                         pre_process=process_to_img)
        vis_board.run_server(args.host, args.port)
