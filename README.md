# CS236781: Deep Learning on Computational Accelerators
# Homework Assignment 3

Faculty of Computer Science, Technion.

## Introduction

In this assignment we'll learn to generate text with a deep multilayer RNN network based on GRU cells.
Then we'll focus our attention on image generation and implement two different generative models:
A variational autoencoder and a generative adversarial network.


## General Guidelines

- Please read the [getting started page](https://vistalab-technion.github.io/cs236781/assignments/getting-started) on the course website. It explains how to **setup, run and submit** the assignment.
- This assignment requires running on GPU-enabled hardware. Please read the [course servers usage guide](https://vistalab-technion.github.io/cs236781/assignments/hpc-servers). It explains how to use and run your code on the course servers to benefit from training with GPUs.
- The text and code cells in these notebooks are intended to guide you through the
  assignment and help you verify your solutions.
  The notebooks **do not need to be edited** at all (unless you wish to play around).
  The only exception is to fill your name(s) in the above cell before submission.
  Please do not remove sections or change the order of any cells.
- All your code (and even answers to questions) should be written in the files
  within the python package corresponding the assignment number (`hw1`, `hw2`, etc).
  You can of course use any editor or IDE to work on these files.

## Contents
- [Part1: Sequence Models](#part1)
    - [Text generation with a char-level RNN](#part1_1)
    - [Obtaining the corpus](#part1_2)
    - [Data Preprocessing](#part1_3)
    - [Dataset Creation](#part1_4)
    - [Model Implementation](#part1_5)
    - [Generating text by sampling](#part1_6)
    - [Training](#part1_7)
    - [Generating a work of art](#part1_8)
    - [Questions](#part1_9)
- [Part 2: Variational Autoencoder](#part2):
    - [Obtaining the dataset](#part2_1)
    - [The Variational Autoencoder](#part2_2)
    - [Model Implementation](#part2_3)
    - [Loss Implementation](#part2_4)
    - [Sampling](#part2_5)
    - [Training](#part2_6)
    - [Questions](#part2_7)
- [Part 3: Generative Adversarial Networks](#part3)
    - [Obtaining the dataset](#part3_1)
    - [Generative Adversarial Nets (GANs)](#part3_2)
    - [Training GANs](#part3_3)
    - [Model Implementation](#part3_4)
    - [Loss Implementation](#part3_5)
    - [Sampling](#part3_6)
    - [Training](#part3_7)
    - [Questions](#part3_8)
