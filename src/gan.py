import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer

# batch_size lives here
from hw3.answers import part3_gan_hyperparams


def new_width(width, size, padding, stride):
    return (width - size + 2*padding) // stride + 1


class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        #  To extract image features you can use the EncoderCNN from the VAE
        #  section or implement something new.
        #  You can then use either an affine layer or another conv layer to
        #  flatten the features.
        # ====== YOUR CODE: ======
        in_channels, width, height = self.in_size
        channels = [128, 256, 512, 1024]
        modules = []
        modules.append(nn.Conv2d(
            in_channels, out_channels=channels[0], kernel_size=4, stride=2,
            padding=1))
        modules.append(nn.BatchNorm2d(channels[0]))
        modules.append(nn.LeakyReLU())
        modules.append(nn.Dropout(0.1))
        modules.append(nn.Conv2d(
            channels[0], out_channels=channels[1], kernel_size=4, stride=2,
            padding=1))
        modules.append(nn.BatchNorm2d(channels[1]))
        modules.append(nn.LeakyReLU())
        modules.append(nn.Dropout(0.1))
        modules.append(nn.Conv2d(
            channels[1], out_channels=channels[2], kernel_size=4, stride=2,
            padding=1))
        modules.append(nn.BatchNorm2d(channels[2]))
        modules.append(nn.LeakyReLU())
        modules.append(nn.Dropout(0.1))
        modules.append(nn.Conv2d(
            channels[2], out_channels=channels[3], kernel_size=4, stride=2,
            padding=1))
        modules.append(nn.BatchNorm2d(channels[3]))
        modules.append(nn.ReLU())
        # ========================
        self.cnn = nn.Sequential(*modules)
        self.linear = nn.Linear(4 * 4 * channels[3], 1)
        # ========================

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        #  No need to apply sigmoid to obtain probability - we'll combine it
        #  with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        encoded_batch = self.cnn(x)
        y = self.linear(encoded_batch.flatten(start_dim=1))
        # ========================
        return y


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim

        # TODO: Create the generator model layers.
        #  To combine image features you can use the DecoderCNN from the VAE
        #  section or implement something new.
    #  You can assume a fixed image size.
        # ====== YOUR CODE: ======
        # Convert the sample from the underlying distribution to features
        channels = [1024, 512, 256, 128]
        # self.projector = nn.Linear(
        #     z_dim, featuremap_size * featuremap_size * channels[0])
        self.projector = nn.ConvTranspose2d(
            z_dim, channels[0], kernel_size=4, stride=1, padding=0, output_padding=0)

        # Decode the features
        modules = [
            nn.BatchNorm2d(channels[0]),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                channels[0], channels[1], kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(channels[1]),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                channels[1], channels[2], kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(channels[2]),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                channels[2], channels[3], kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(channels[3]),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                channels[3], out_channels, kernel_size=4, stride=2, padding=1, output_padding=0),
        ]
        # ========================
        self.cnn = nn.Sequential(*modules)
        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        #  Generate n latent space samples and return their reconstructions.
        #  Don't use a loop.
        # ====== YOUR CODE: ======
        samples = torch.randn((n, self.z_dim), device=device)
        # samples = samples.to(device)
        if with_grad:
            return self.forward(samples)
        else:
            with torch.no_grad():
                return self.forward(samples)
        # ========================

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        #  Don't forget to make sure the output instances have the same
        #  dynamic range as the original (real) images.
        # ====== YOUR CODE: ======
        sausages = self.projector(z.reshape(-1, self.z_dim, 1, 1))
        x = torch.tanh(self.cnn(sausages))
        # ========================
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the discriminator loss.
    #  See pytorch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======
    loss_function = nn.BCEWithLogitsLoss()

    data_size = y_data.size(0)
    generated_size = y_generated.size(0)

    # Enable the loss_fn to handle empty data or generated batches
    offset_1 = torch.rand(data_size) * \
        label_noise - label_noise / 2
    real_target = data_label + offset_1
    y_data = y_data.to('cuda')
    real_target = real_target.to('cuda')
    loss_data = loss_function(y_data, real_target)
    if generated_size == 0:
        return loss_data

    offset_2 = torch.rand(generated_size) * \
        label_noise - label_noise / 2
    generated_target = 1 - data_label + offset_2
    y_generated = y_generated.to('cuda')
    generated_target = generated_target.to('cuda')
    loss_generated = loss_function(y_generated, generated_target)
    if data_size == 0:
        return loss_generated
    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the Generator loss.
    #  Think about what you need to compare the input to, in order to
    #  formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    loss_function = nn.BCEWithLogitsLoss()
    target = torch.zeros(y_generated.size(0), device='cuda')
    target[:] = data_label
    y_generated = y_generated.to('cuda')
    loss = loss_function(y_generated, target)
    # ========================
    return loss


def train_batch(
    dsc_model: Discriminator,
    gen_model: Generator,
    dsc_loss_fn: Callable,
    gen_loss_fn: Callable,
    dsc_optimizer: Optimizer,
    gen_optimizer: Optimizer,
    x_data: DataLoader,
):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    #  1. Show the discriminator real and generated data
    #  2. Calculate discriminator loss
    #  3. Update discriminator parameters
    # ====== YOUR CODE: ======
    dsc_optimizer.zero_grad()
    x_data = x_data + \
        torch.normal(mean=0.0, std=0.05, size=x_data.size(), device='cuda')
    real_scores = dsc_model.forward(x_data).flatten()
    generated_images = gen_model.sample(x_data.size(0))
    # generated_images = generated_images + \
    #     torch.normal(mean=0.0, std=0.05, size=x_data.size(), device='cuda')
    generated_scores = dsc_model.forward(generated_images).flatten()
    dsc_loss = dsc_loss_fn(real_scores, generated_scores)
    dsc_loss.backward()
    dsc_optimizer.step()
    # ========================

    # TODO: Generator update
    #  1. Show the discriminator generated data
    #  2. Calculate generator loss
    #  3. Update generator parameters
    # ====== YOUR CODE: ======
    # Help the generator by giving it more training:
    # it should win over the discriminator in the end,
    # so we train it until it catches up.
    # while True:
    gen_optimizer.zero_grad()
    generated_images = gen_model.sample(x_data.size(0), with_grad=True)
    image_scores = dsc_model.forward(generated_images).flatten()
    gen_loss = gen_loss_fn(image_scores)
    gen_loss.backward()
    gen_optimizer.step()
    # if gen_loss < dsc_loss:
    #     break
    # ========================

    return dsc_loss.item(), gen_loss.item()


def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    """
    Saves a checkpoint of the generator, if necessary.
    :param gen_model: The Generator model to save.
    :param dsc_losses: Avg. discriminator loss per epoch.
    :param gen_losses: Avg. generator loss per epoch.
    :param checkpoint_file: Path without extension to save generator to.
    """

    saved = False
    checkpoint_file = f"{checkpoint_file}.pt"

    # TODO:
    #  Save a checkpoint of the generator model. You can use torch.save().
    #  You should decide what logic to use for deciding when to save.
    #  If you save, set saved to True.
    # ====== YOUR CODE: ======
    import numpy as np
    if np.std(dsc_losses[-20:]) < 0.12 and np.std(gen_losses[-20:]) < 0.04:
        torch.save(gen_model, checkpoint_file)
        saved = True
    # ========================

    return saved
