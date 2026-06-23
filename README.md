# Deep Multi-Agent Reinforcement Learning for Cooperative Multi-UAV Payload Transportation

## Overview

This repository contains the implementation of my undergraduate thesis at Institut Teknologi Bandung (ITB). This thesis aims to investigate cooperative payload transportation of multi-UAV systems with Multi-Agent RL using NVIDIA's Isaac Lab, navigating through constrained environments.

## Environment

![Environment in Isaac Lab](env_isaaclab.png)

![Top-View 2D Environment Representation](2Denv_1.png)

## Results

### Videos

<video controls src="results/videos/WORLD_NEGATIVE_2026-06-08_23-07-17_mappo_torch_direct_marl.mp4" title="Left Initial Condition"></video>

<video controls src="results/videos/WORLD_ZERO_2026-06-08_23-07-17_mappo_torch_direct_marl.mp4" title="Middle Initial Condition"></video>

<video controls src="results/videos/WORLD_POSITIVE_2026-06-08_23-07-17_mappo_torch_direct_marl.mp4" title="Right Initial Condition"></video>

### Trajectory Plots

![Mid-Training 400K Episodes (Upper row) and Post-Training 1.5M Episodes (Lower row) Representation](results/figures/2dtrajectory.png)

### Learning Curve

![Total Reward Graph](results/figures/learningcurve_1.png)

## Comments

The structure of this repository is currently disordered, it is a cloned repo from ([Isaac Lab](https://github.com/isaac-sim/IsaacLab)), which was then added with another cloned repo from ([Zeng2025](https://github.com/Aerial-Manipulation-Lab/MARL_cooperative_aerial_manipulation_ext)) to implement the MAPPO skrl framework and drone model.