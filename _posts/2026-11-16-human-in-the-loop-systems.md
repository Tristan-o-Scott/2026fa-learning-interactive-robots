---
layout: distill
title: "Human-in-the-loop Systems"
description: "Interactive reinforcement learning, human feedback loops, and human-in-the-loop cyber-physical robotic systems."
date: 2026-11-16
future: true
htmlwidgets: true
ready: true

# Reporter team authors
authors:
  - name: "Grace Bergquist, Daniel Evans, Jessi Morris, Atharva Tilak, Megan Vu"

bibliography: 2026-11-16-human-in-the-loop-systems.bib

toc:
  - name: "Introduction"
  - name: "Diff-DAgger"
    subsections:
    - name: "Motivation"
    - name: "Background"
    - name: "Method"
    - name: "Results"
    - name: "Limitations"
    - name: "Takeaways"
  - name: "Student Q&A"
---

## Introduction


---

## Diff-DAgger: Uncertainty Estimation with Diffusion Policy for Robotic Manipulation <d-cite key="lee_diff-dagger_2025"></d-cite> {#diff-dagger}

Imagine using behavioral cloning to teach a robot how to perform basic household tasks. Your training data contains many examples (demonstrations) of picking up cups of water. At test time, the robot goes to pick up a cup, but its grip begins slightly off-center. Because this case was never addressed in the demonstrations, the robot continues to grip with the same pressure, pushing the cup out of the gripper and onto the floor. All your investors are laughing, and your shoes are soaked. 

Dataset aggregation (DAgger) aims to prevent catastrophes like these. The idea is to make the supervised training process more robust by allowing it to fill in its own gaps. After initial training, the robot is allowed to move in a simulated or real environment that can progress out of the training distribution. When the environment drifts far enough away from what the robot has been trained for, control is given to an expert -- typically a human demonstrator -- who steps in and corrects the action sequence. Then, the robot is re-trained with the new gaps filled in, and the process repeats. 

While this technique works quite well in situations where reinforcement learning is infeasible (at least initially), a significant challenging is choosing when intervention should occur. As discussed in the following section, multimodal action distributions can make this process especially difficult. <d-cite key="lee_diff-dagger_2025"></d-cite> note that modern robot policies often use diffusion models to produce actions, from which they extrapolate an improved method for choosing when to intervene, which they call Diff-DAgger.

### Motivation



### Method


### Results


### Limitations

### Takeaways


---

## Student Q&A

<!-- TODO: group questions under ### topic headings, as in the Foundation Models for Language report -->

### Topic

**Q: Question?**

A: Answer.
<br><br>
