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

### Motivation & Background

The key issue that caused the embarassing investor demo in the introduction is called "covariate shift," when actions taken by the agent result in shifted input variable distributions <d-cite key="ross_reduction_2011"></d-cite>. Here is a summary of different DAgger methods:

| Method | Description |
|---|---|
| **DAgger** | The original DAgger method. New labels are collected at out-of-distribution (OOD) states, and the dataset is re-aggregated. <d-cite key="ross_reduction_2011"></d-cite> |
| **Human-gated DAgger** | A human watches the robot take actions in its environment, then chooses when to intervene. <d-cite key="kelly_hg-dagger_2019"></d-cite> |
| **Robot-gated DAgger** | The robot decides when an intervention should happen, typically by estimating uncertainty. <d-cite key="hoque_fleet-dagger_2023"></d-cite> |
| **Ensemble DAgger** | A robot-gated method that computes uncertainty as disagreement between multiple policies. <d-cite key="menda_ensembledagger_2019"></d-cite> |

{% include figure.liquid 
   path="assets/img/2026-11-16-human-in-the-loop-systems/distribution-modes.png" 
   class="img-fluid rounded z-depth-1" 
   caption="Figure X: Demonstrations from start to goal. Most states are in-distribution and uni-modal (red/orange). At the fork, valid demonstrations split between the upper and lower paths, creating a multi-modal region (green). States away from the demonstrations are out-of-distribution (blue)." 
%}

Figure X illustrates a key problem with Ensemble DAgger. At the multi-modal region where the agent must traverse the circle, two very different actions are possible: the agent may turn left or right. However, because different policies can choose either option, this would register as uncertainty, and the expert would be asked to intervene regardless of whether or not the robot was OOD. 

<d-cite key="lee_diff-dagger_2025"></d-cite> proposes a new method for deciding when interventions should happen that uses a metric inherent to a popular policy architecture: diffusion models. Denoising Diffusion Probabilistic Models (DDPMs) work by iteratively producing a sequence of actions from noise. Its loss looks like 

$$\mathcal{L}_\pi(o,a,\epsilon,t)=\left\|f(\epsilon,a,t)-f_\theta\!\left(o,\sqrt{\alpha_t}\,a+\sqrt{1-\alpha_t}\,\epsilon,\,t\right)\right\|^2,$$

where $\mathcal{L}_\pi(o,a,\epsilon,t)$ represents the loss for a policy $\pi$ given observations $o$, an action label $a$, some noise $\epsilon$, and noising step $t$. The right-hand side simply calculates the distance of the diffusion model $f_\theta$'s predicted action from the action label. The intuition of Diff-DAgger is that this loss function presents a better way to proxy uncertainty than asking multiple policies for potentially conflicting answers. Instead, the authors propose that the agent attempt to reconstruct its own action, the loss of which can provide a less-conflicting uncertainty estimate. 

### Method
The authors of Diff-DAgger use diffusion loss to estimate uncertainty. In order to compute loss in unseen states, they use the following estimation:
$$\mathcal{L}(o, a, \pi) = \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I),\; t \sim \mu(1, T)} \, \mathcal{L}_\pi(o, a, \epsilon, t).$$

In fact, this simply reuses the loss equation from earlier. The key idea is that the input action $a$ is now the action already generated by the model. If $\mathcal{L}(o, a, \pi)$ gives a loss higher than the $\alpha$ percentile observed in training for $K$ consecutive time steps, the robot will trigger an intervention. Because the decision originally generated by the policy is preserved, the noised-input should already be on the slope of a local extrema in a multi-modal decision point scenario. This has an additional benefit over ensemble-DAgger: instead of running multiple policies at once, only one action needs to be computed at each time step. 

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
