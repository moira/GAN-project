r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=128,
        seq_len=448,
        h_dim=684,
        n_layers=6,
        dropout=0.19,
        learn_rate=0.001,
        lr_sched_factor=0.2,
        lr_sched_patience=1,
    )
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "ACT I. SCENE I."
    temperature = 0.00625
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**<br/>
We cannot train on the entire corpus because it will take too much memory, which needs fit memory size of the GPU.<br/>
Additionally, training on the text serially will not use the benefits of parallel computing abilities of the GPU, and by<br/>
running in batch i.e. splitting the text to sequences we can train paralleled.

"""

part1_q2 = r"""
**Your answer:**<br/>
The reason that we can generate text longer then sequance length is the use of hidden states inside the model, i.e. the use of network memory.<br/>
By using Hidden-States we are able to feed the network the last generated character with the previous hidden states,<br/>
and by that create different new characters. With the previous hidden state the model remebers its past input and can preduce the nex character.

"""

part1_q3 = r"""
**Your answer:**<br/>
Since the dataset is sequential by shuffling between batches well damage the integrity of the dataset.<br/>
If we'll train out of order,we will be unable to pass the hidden state between batches, and worsen our training effort. 

"""

part1_q4 = r"""
**Your answer:<br/>
1.The temperature hyperparameter controls the the noise of the soft-max function as seen in the previous section,  
while in training we would like to have noise so the model will be able to learn better, when generating we would<br/>
like to see low noise and low variance of the model, which gives the model probability less uniform properties.<br/>
2.When the temperature is very high we can see that softmax behave like uniform probablity, and the variance is very high. <br/>
It causes the the model to choose the next character randomly which is not what we desire.<br/>
3.When the temperature is very low we can see that softmax behave like delta function, and the variance is very low. <br/>
there for the highest score out of the model has the value 1, and It causes the the model to choose the most likely next character.<br/>

"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=4, h_dim=256, z_dim=2,
        x_sigma2=0.00001, learn_rate=0.00002, betas=(0.9, 0.999)
    )
    # ========================
    return hypers


part2_q1 = r"""
The parameter $\sigma^2$ controls the relative importance of data loss
in the total loss. Low values of $\sigma^2$ increase the data loss, because
it is multiplied by $1/\sigma^2$, and force the autoencoder to replicate
the original images more precisely. Large values of $\sigma^2$ make data loss
less relevant, and images become more fuzzy.
"""

part2_q2 = r"""
1. The reconstruction loss term penalises the autoencoder for incorrectly
reconstructing images. The KL loss term penalises the encoder part of the
autoencoder for inaccurately approximating the conditional distribution
$p(z|X)$.
2. The bigger the KL loss term, the closer the points in the latent space are
brought together.
3. This effect is beneficial for sampling: when projections of original data
are very sparse, there are no smooth transitions between different categories
present in the input, and the decoder may encounter samples from the regions
it has very rarely or never seen before. Packing projections more closely
together helps the decoder to interpolate between classes.
"""

part2_q3 = r"""
We want to create a model that will be able to mimic the reference dataset.
In order to do this, the model has to sample relevant input classes from the
latent space, i.e. maximise $p(z)$, and then successfully reconstruct images
based on them, i.e. maximise $p(x|z)$.  
$p(z)\cdot p(x|z) = p(x)$, therefore this is the term we want to maximise.
"""

part2_q4 = r"""
This is done to achieve numerical stability. Variances should be strictly
positive, and they are often very close to zero; these two factors
introduce discontinuities and make gradients unstable. Transforming them
to logs makes it possible for the network to look for a good value in
an unconstrained interval.
"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=32,
        z_dim=4,
        data_label=0,
        label_noise=0.2,
        discriminator_optimizer=dict(
            type="SGD",  # Any name in nn.optim like SGD, Adam
            lr=0.002,
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type="SGD",  # Any name in nn.optim like SGD, Adam
            lr=0.005,
            momentum=0.9,
            # type="Adam",  # Any name in nn.optim like SGD, Adam
            # lr=0.001,
            # betas=(0.5, 0.999)
            # # You an add extra args for the optimizer here
        ),
    )
    # ========================
    return hypers


part3_q1 = r"""
When training the discriminator network, we use samples produced
by the generator. The generator is assumed to be fixed at this stage.
Therefore, we do not maintain gradients in its tensors because they
will not be used in the backwards step.

Then we start training the generator, and it is not fixed any more
but is optimised. Therefore we preserve gradients in its network for
backpropagation.
"""

part3_q2 = r"""
1. We should not stop training the network based solely on the value of the
generator loss because it is not informative: a bad discriminator may give rise
to very low generator loss while the generator itself is still very far from
optimal.
2. This might mean that the generator does not produce diverse enough images.
If it converges on a small set of image types that it can produce well, the
discriminator at some point may become unable to tell them from real images,
while the generator will continue perfecting these images even further.
"""

part3_q3 = r"""
VAEs produce images that look like smooth interpolations between different
image types found in the training dataset. This is the effect of using the
KL loss term to force the model to tightly pack the points in the latent
space together. Because of this, the central part of the image is underlined,
as it is stable across data points, and the background is more difficult to
captures.

GANs' images are not smoothed because there is no need for the generator
to produce an "average" image in order to fool the discriminator. The generator
tries to mimic all aspects of the target image individually, but cannot
reproduce them exactly, which sometimes leads to "patchwork"-like effect.
"""

# ==============
