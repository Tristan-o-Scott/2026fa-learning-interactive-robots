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
  - name: "KnowNo"
    subsections:
    - name: "Method"
    - name: "Experimental Setup"
    - name: "Baselines"
    - name: "Results"
    - name: "Conclusions"
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

## KnowNo: Robots That Ask For Help <d-cite key="ren_robots_2023"></d-cite> {#knowno}


### Method

KnowNo prompts the LLM to generate candidate next steps, formatted as multiple-choice question answering (MCQA) with options A–D and an added "E: an option not listed here" choice. The LLM's next-token probability for each letter serves as its confidence in that option.

Using a calibration set of example scenarios, Conformal Prediction finds a confidence threshold that achieves a user-specified success level $1 - \epsilon$. At test time, every option scoring above the threshold goes into a *prediction set*:

- If the set contains exactly one option, the robot executes it.
- If the set contains more than one option, the robot asks the human to choose.

### Experimental Setup

Four environments spanning simulation, single-arm manipulation, mobile manipulation, and bimanual coordination:

| Environment | Task | Ambiguity |
|---|---|---|
| **Simulation:** PyBullet Tabletop Rearrangement | A simulated robot arm rearranges colored blocks and bowls on a tabletop | Attribute, numeric, spatial |
| **Hardware:** Multi-Step Tabletop Rearrangement (UR5 arm) | A physical UR5 arm with a suction gripper sorts 28 toy food items into blue and green plates based on human dietary preferences | Only a subset of preferences is given |
| **Hardware:** Mobile Manipulator in an Office Kitchen | A mobile robot operates in an office kitchen with a countertop, drawers, a microwave, and waste bins | Spatial, safety |
| **Hardware:** Bimanual Tabletop Manipulation (dual Kuka IIWA 7 arms) | Two arms with limited reachability must coordinate object handovers and bin sorting | Vague commands (e.g., which arm should act) |

### Baselines

KnowNo was primarily evaluated with PaLM-2L as the base planner, and compared against:

| Baseline | Description |
|---|---|
| **Simple Set** | Adds options in order of LLM confidence until their total reaches $1 - \epsilon$, without calibration |
| **Ensemble Set** | Same as Simple Set, but scores come from how often the LLM picks each option across 20 randomized prompts |
| **Prompt Set** | Asks the LLM to list the prediction set directly |
| **Binary** | Asks the LLM to say whether it is certain or uncertain |
| **No Help** | Always executes the top-scoring option |

### Results

1. **Reliable task success guarantees.** In simulation, with a target success level of 85% ($\epsilon = 0.15$), KnowNo had the smallest deviation from the target across all three ambiguity settings. The uncalibrated baselines either missed the target or gave the user no way to set one.
2. **Minimizing human help rate.** For the same success level, KnowNo asked for help less often than the baselines. The gap grew as instructions became more vague. Ensemble Set also performed well, but needed more inference time.
3. **Quantitative hardware performance.** On the UR5, KnowNo reduced human help by 14% per step and 8% per trial compared to Simple Set. On the mobile manipulator, it reduced help by 14%. Overall, KnowNo reduced human help by 10–24% compared to the baselines.
4. **Ablation on scoring functions.** Replacing the MCQA letter scores with sentence-level scores (raw sentence probability or perplexity) made performance worse, especially when candidate options varied a lot in length. Scoring a single letter avoids this length bias.

### Conclusions

Conformal prediction paired with MCQA allows operators to tune a statistical guarantee on task completion, while keeping human help to a minimum. KnowNo works with off-the-shelf LLMs without fine-tuning.

The key limitation is that the guarantee currently assumes:

- 1. the environment's scene state is accurately parsed into text.
- 2. low-level robotic controllers execute the selected primitive without failure.
- 3. a clean environment, that is, the method does not account for human error, so a wrong answer from the human breaks the guarantee.

In addition, calibration requires a labeled dataset of scenarios. The authors used 400 examples, which took about 4 hours to label in the multi-step setting, and this must be repeated for each new environment.

---

## Student Q&A

<!-- TODO: group questions under ### topic headings, as in the Foundation Models for Language report -->

### Topic

**Q: Question?**

A: Answer.
<br><br>
