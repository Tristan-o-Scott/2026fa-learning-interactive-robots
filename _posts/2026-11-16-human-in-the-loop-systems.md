---
layout: distill
title: "Human-in-the-loop Systems"
description: "When should a robot ask a person for help? We cover Diff-DAgger, which uses diffusion loss to decide when an expert should take over, and KnowNo, which uses conformal prediction to decide when an LLM planner should ask for clarification."
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
    subsections:
    - name: "Overview of Papers"
  - name: "Diff-DAgger"
    subsections:
    - name: "Motivation and Background"
    - name: "Method"
    - name: "Results"
    - name: "Limitations"
    - name: "Takeaways"
  - name: "KnowNo"
    subsections:
    - name: "Related Work"
    - name: "How KnowNo Works"
    - name: "Experimental Setup"
    - name: "Baselines"
    - name: "KnowNo Results"
    - name: "Conclusions"
  - name: "Student Q&A"
---

## Introduction

This lecture report covers the **Human-in-the-loop Systems** session in *Learning for Interactive Robots (CS 6501, Fall 2026)* at the University of Virginia.

**The main question:** When should a robot stop and ask a person for help?

Most robot learning puts the human at the start of the pipeline. People collect demonstrations or design a reward function, the robot trains, and after that the robot is on its own. A human-in-the-loop system keeps a person involved while the robot is actually running. That person might take over the controls, correct a mistake, or answer a question. Any of these makes the robot more reliable, and any of them costs the person time and attention.

That trade-off is the main design problem. A robot that asks too rarely fails in situations it was never trained for. A robot that asks too often isn't saving anyone any work, since the person might as well do the task themselves. Striking the balance depends on the robot having a decent sense of its own uncertainty, which is hard for the large learned models robots use today. Diffusion policies and LLM planners can both be confidently wrong.

### Overview of Papers

Both papers in this report are robot-gated, meaning the robot decides on its own when to ask for help. They differ in what kind of help the robot asks for and in how it measures its own uncertainty.

**Diff-DAgger** <d-cite key="lee_diff-dagger_2025"></d-cite> works at the level of motor control. A diffusion policy uses its own training loss to notice when its actions have drifted away from the expert data, then hands control to an expert who shows it what to do. Those corrections go into the dataset, and the policy is retrained on the bigger dataset before the next round.

**KnowNo** <d-cite key="ren_robots_2023"></d-cite> works at the level of task planning. An LLM planner writes out its possible next steps as a multiple-choice question and uses conformal prediction to figure out which options it can't rule out. If more than one is left, it asks the human to pick. The LLM is never retrained. What the user gets instead is a statistical guarantee on how often the task succeeds.

| | Diff-DAgger | KnowNo |
|---|---|---|
| **Level** | Low-level control | High-level planning |
| **Uncertainty signal** | Diffusion loss on the robot's own action | LLM confidence in each multiple-choice option |
| **When the robot asks** | Loss stays above a threshold for $K$ steps in a row | More than one option survives conformal prediction |
| **What the human does** | Takes over and demonstrates | Picks the correct option |
| **What the help is used for** | Training data for the next DAgger round | Finishing the current task |
| **Guarantee** | None, results are empirical | Task success of at least $1 - \epsilon$ |

*Table 1: The two papers side by side.*

Between them, the two papers cover two different reasons a robot might need a person. Diff-DAgger asks for help when the robot's actions stop looking like the expert's, and KnowNo asks when the robot can't tell what the person meant.

---

## Diff-DAgger: Uncertainty Estimation with Diffusion Policy for Robotic Manipulation <d-cite key="lee_diff-dagger_2025"></d-cite> {#diff-dagger}

Imagine using behavioral cloning to teach a robot how to perform basic household tasks. Your training data contains many examples (demonstrations) of picking up cups of water. At test time, the robot goes to pick up a cup, but its grip begins slightly off-center. Because this case was never addressed in the demonstrations, the robot continues to grip with the same pressure, pushing the cup out of the gripper and onto the floor. All your investors are laughing, and your shoes are soaked. 

Dataset aggregation (DAgger) aims to prevent catastrophes like these. The idea is to make the supervised training process more robust by allowing it to fill in its own gaps. After initial training, the robot is allowed to move in a simulated or real environment that can progress out of the training distribution. When the environment drifts far enough away from what the robot has been trained for, control is given to an expert -- typically a human demonstrator -- who steps in and corrects the action sequence. Then, the robot is re-trained with the new gaps filled in, and the process repeats. 

While this technique works quite well in situations where reinforcement learning is infeasible (at least initially), a significant challenge is choosing when intervention should occur. As discussed in the following section, multimodal action distributions can make this process especially difficult. <d-cite key="lee_diff-dagger_2025"></d-cite> note that modern robot policies often use diffusion models to produce actions, from which they extrapolate an improved method for choosing when to intervene, which they call Diff-DAgger.

### Motivation and Background

The key issue that caused the embarrassing investor demo in the introduction is called "covariate shift," when actions taken by the agent result in shifted input variable distributions <d-cite key="ross_reduction_2011"></d-cite>. Here is a summary of different DAgger methods:

| Method | Description |
|---|---|
| **DAgger** | The original DAgger method. New labels are collected at out-of-distribution (OOD) states, and the dataset is re-aggregated. <d-cite key="ross_reduction_2011"></d-cite> |
| **Human-gated DAgger** | A human watches the robot take actions in its environment, then chooses when to intervene. <d-cite key="kelly_hg-dagger_2019"></d-cite> |
| **Robot-gated DAgger** | The robot decides when an intervention should happen, typically by estimating uncertainty. <d-cite key="hoque_fleet-dagger_2023"></d-cite> |
| **Ensemble DAgger** | A robot-gated method that computes uncertainty as disagreement between multiple policies. <d-cite key="menda_ensembledagger_2019"></d-cite> |

{% include figure.liquid 
   path="assets/img/2026-11-16-human-in-the-loop-systems/distribution-modes.png" 
   class="img-fluid rounded z-depth-1" 
   caption="Figure 1: Demonstrations from start to goal. Most states are in-distribution and uni-modal (red/orange). At the fork, valid demonstrations split between the upper and lower paths, creating a multi-modal region (green). States away from the demonstrations are out-of-distribution (blue)." 
%}

Figure 1 illustrates a key problem with Ensemble DAgger. At the multi-modal region where the agent must traverse the circle, two very different actions are possible: the agent may turn left or right. However, because different policies can choose either option, this would register as uncertainty, and the expert would be asked to intervene regardless of whether or not the robot was OOD. 

<d-cite key="lee_diff-dagger_2025"></d-cite> proposes a new method for deciding when interventions should happen that uses a metric inherent to a popular policy architecture: diffusion models. Denoising Diffusion Probabilistic Models (DDPMs) work by iteratively producing a sequence of actions from noise. Its loss looks like 

$$\mathcal{L}_\pi(o,a,\epsilon,t)=\left\|f(\epsilon,a,t)-f_\theta\!\left(o,\sqrt{\alpha_t}\,a+\sqrt{1-\alpha_t}\,\epsilon,\,t\right)\right\|^2,$$

where $\mathcal{L}_\pi(o,a,\epsilon,t)$ represents the loss for a policy $\pi$ given observations $o$, an action label $a$, some noise $\epsilon$, and noising step $t$. The right-hand side simply calculates the distance between the diffusion model $f_\theta$'s prediction and the training target $f(\epsilon,a,t)$ (the noise, the clean action, or a "velocity," depending on how the model is trained) for the noised action label. The intuition of Diff-DAgger is that this loss function presents a better way to proxy uncertainty than asking multiple policies for potentially conflicting answers. Instead, the authors propose that the agent attempt to reconstruct its own action, the loss of which can provide a less-conflicting uncertainty estimate. 

### Method
The authors of Diff-DAgger use diffusion loss to estimate uncertainty. In order to compute loss in unseen states, they use the following estimation:

$$\mathcal{L}(o, a, \pi) = \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I),\; t \sim \mu(1, T)} \, \mathcal{L}_\pi(o, a, \epsilon, t).$$

In fact, this simply reuses the loss equation from earlier. The key idea is that the input action $a$ is now the action already generated by the model. If $\mathcal{L}(o, a, \pi)$ gives a loss higher than the $\alpha$ percentile observed in training for $K$ consecutive time steps, the robot will trigger an intervention. Because the action being scored is the one the policy itself generated, it already sits near one of the learned modes, so at a multi-modal decision point the loss stays low instead of being inflated by the other valid mode. This has an additional benefit over ensemble-DAgger: instead of running multiple policies at once, only one action needs to be computed at each time step. 

### Results

#### What We Are Testing

**Robot-gated DAgger** lets the robot decide when to ask an expert for help. This only works if the robot can tell when it is in trouble.

Diff-DAgger does this with one trick. A diffusion policy is trained to predict noise. At rollout time, we reuse that same training loss to score the action the robot just generated. A high loss means "this looks unlike my training data."

{% include figure.liquid
   path="assets/img/2026-11-16-human-in-the-loop-systems/fig1_overview.png"
   class="img-fluid rounded z-depth-1"
   caption="Figure 2 (paper Fig. 1): Overview of Diff-DAgger. (a) A diffusion policy is trained on expert data. (b) During rollout, the robot generates an action, computes the diffusion loss on it, and continues only if the loss is low. Otherwise the expert takes over."
%}

The loss comes from the standard diffusion policy objective:

$$
\mathcal{L}_\pi(o, a, \epsilon, t) = \left\| f(\epsilon, a, t) - f_\theta\!\left(o,\ \sqrt{\alpha_t}\,a + \sqrt{1-\alpha_t}\,\epsilon,\ t\right) \right\|^2
$$

Here $o$ is the observation, $a$ is the action sequence, $\epsilon$ is Gaussian noise, and $t$ is the denoising step. Because $\epsilon$ and $t$ are random, the paper averages the loss over a batch of $N_b$ samples:

$$
\bar{\mathcal{L}}(o, a, \pi) = \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I),\; t \sim \mu(1, T)} \; \mathcal{L}_\pi(o, a, \epsilon, t)
$$

The decision rule has two knobs:

- $\alpha$ is a quantile. We compute $\bar{\mathcal{L}}$ on every training pair. The $\alpha$-quantile of those values is the threshold.
- $K$ is the patience. The expert takes over only if the loss exceeds the threshold for $K$ timesteps in a row.

$$
\text{query at step } s \iff \bar{\mathcal{L}}(o_\tau, a_\tau, \pi) > q_\alpha \quad \text{for all } \tau \in \{s-K+1, \dots, s\}
$$

{% include figure.liquid
   path="assets/img/2026-11-16-human-in-the-loop-systems/decision_rule.svg"
   class="img-fluid rounded z-depth-1"
   caption="Figure 3: The query rule in action (our own diagram). One noisy spike is ignored. Two high-loss steps in a row, with K = 2, hand control to the expert."
%}

The experiments ask three questions:

1. **RQ1.** Can diffusion loss predict task failure?
2. **RQ2.** With the same number of expert interventions, does Diff-DAgger beat existing methods?
3. **RQ3.** Does it cut the wall-clock time of each DAgger iteration?

#### Tasks and Experts

There are three simulated tasks (built on ManiSkill) and two real-world tasks.

{% include figure.liquid
   path="assets/img/2026-11-16-human-in-the-loop-systems/fig3_tasks.png"
   class="img-fluid rounded z-depth-1"
   caption="Figure 4 (paper Fig. 3): The five tasks. Top: stacking, pushing, and plugging in simulation. Bottom: stacking and loading on a real Franka robot."
%}

- **Stacking.** Pick up a red cube and place it on a green cube. A basic perception and manipulation test.
- **Pushing.** Use a stick to push a red T-shaped block to a target pose. This is contact-rich and needs multimodal planning.
- **Plugging.** Insert a charger into a receptacle. This needs fine alignment.
- **Loading (real).** Pick up soft bread and place it on a toaster. This tests handling deformable objects.

The simulated experts were designed on purpose. Stacking and plugging use a rule-based RRT motion planner, which plans over a long horizon. Pushing uses **two RL agents** trained with PPO. One is rewarded for rotating the object clockwise. The other is rewarded for counter-clockwise. They alternate when giving demonstrations, so the data truly has two modes for the same state. The authors also test learning from the counter-clockwise agent alone.

The real-world experts are humans teleoperating a robot with a VR controller.

| Type | Task | # Obj | # Cams | Contact rich | High precision | Time limit | Expert | Avg. expert duration | Expert success |
|---|---|---|---|---|---|---|---|---|---|
| Sim | Stacking | 2 | 3 | No | No | 250 | Motion planner | 60.3 | 1.00 |
| Sim | Pushing | 1 | 1 | Yes | No | 250 | RL agent | 65.7 | 0.93 |
| Sim | Plugging | 2 | 3 | No | Yes | 500 | Motion planner | 147 | 0.98 |
| Real | Stacking | 2 | 3 | No | No | 600 | Human | 174.2 | 1.00 |
| Real | Loading | 2 | 3 | Yes | Yes | 600 | Human | 210.5 | 1.00 |

*Table 2 (paper Table I): Task and expert summary. Time limits are in timesteps, at 20 Hz in simulation and 30 Hz in the real world.*

#### Observations, Models, and Baselines

**Cameras.** Pick-and-place tasks use three cameras (left, right, in-hand) to avoid occlusion. Pushing uses one top-down camera. The real robot is a Franka Research 3 with three RealSense D435 cameras.

**Model.** All methods use the same diffusion policy: a U-Net head with FiLM conditioning. The paper tests two input types:

- **State-based.** Robot joint positions, end-effector pose, and the true object poses.
- **Image-based.** An R3M ResNet18 encoder with spatial softmax, plus proprioception. Random crop augmentation is used.

The observation horizon is set to one to keep batches small. The diffusion model uses V-objective prediction (see "Choosing the Prediction Target" below).

| Task | Action space | Initial demos | Final demos | New demos per update | $\alpha$ | $K$ | $N_b$ |
|---|---|---|---|---|---|---|---|
| Stacking (sim) | Abs. pose | 20 | 60 | 4 | 0.99 | 2 | 512 |
| Pushing (1 expert) | Rel. joint | 20 | 100 | 4 | 0.99 | 1 | 512 |
| Pushing (2 experts) | Rel. joint | 20 | 100 | 4 | 0.9 | 2 | 512 |
| Plugging | Abs. pose | 20 | 100 | 8 | 0.99 | 2 | 512 |
| Stacking (real) | Abs. pose | 40 | 120 | 20 | 0.99 | 2 | 512 |
| Loading (real) | Abs. pose | 40 | 120 | 20 | 0.99 | 2 | 512 |

*Table 3 (paper Table IV, trimmed to the DAgger columns): Key hyperparameters.*

**Baselines.** Three comparisons, all using the same diffusion policy so expressiveness is equal:

- **Behavior Cloning (BC).** Train once on offline demonstrations.
- **Ensemble-DAgger.** Five policies. Their disagreement (action variance) measures uncertainty.
- **ThriftyDAgger.** A strong robot-gated method. It adds a learned risk estimate on top of the ensemble.

#### RQ1: Can Diffusion Loss Predict Task Failure?

##### Why Ensembles Struggle: A Toy Example

Before the robot tasks, the paper uses a 2D navigation problem. An agent goes from start to goal around a circle. It can go clockwise or counter-clockwise. Both are correct, so the data has two modes.

{% include figure.liquid
   path="assets/img/2026-11-16-human-in-the-loop-systems/fig2_toy_example.png"
   class="img-fluid rounded z-depth-1"
   caption="Figure 5 (paper Fig. 2): (a) The toy problem. Red/orange dots are the two modes. Green marks the fork where the paths split. The dark blue area is out-of-distribution. (b) Left: action standard deviation across the ensemble. Right: diffusion loss. Both measured over 100 runs."
%}

Look at the left chart. The ensemble's disagreement is highest at the **fork**, which is a normal, in-distribution spot. It is even a bit higher than in the truly out-of-distribution region. Each policy is confident, they just pick different paths. The ensemble reads that as uncertainty.

Now the right chart. Diffusion loss is low and nearly equal for the single-mode region and the fork. It jumps only in the out-of-distribution region. That is exactly the behavior we want.

##### The Evaluation Setup

For each task, the policy is trained on enough demos to land near a **50% success rate**. Then it runs 100 random episodes with no expert help. For each episode, we record two things: did the task fail, and did the robot ask for help? All methods use the same $\alpha$ and $K$.

{% include figure.liquid
   path="assets/img/2026-11-16-human-in-the-loop-systems/confusion_matrix.svg"
   class="img-fluid rounded z-depth-1"
   caption="Figure 6: How query decisions line up with outcomes (our own diagram)."
%}
We then report standard scores:

$$
\text{TPR} = \frac{TP}{TP + FN}, \qquad \text{TNR} = \frac{TN}{TN + FP}, \qquad F_1 = \frac{2\,TP}{2\,TP + FP + FN}, \qquad \text{Acc} = \frac{TP + TN}{TP + TN + FP + FN}
$$

TPR is the share of failing episodes where the robot asked for help. TNR is the share of successful episodes where it stayed independent.

##### Results

| Task | Demos | Method | TPR | TNR | F1 | Acc |
|---|---|---|---|---|---|---|
| Stacking | 30 | **Ours** | 0.88 | 0.77 | **0.83** | **0.84** |
| | | Ensemble-DA | 0.81 | 0.31 | 0.52 | 0.59 |
| | | Thrifty-DA | 0.95 | 0.21 | 0.54 | 0.65 |
| Pushing (1 expert) | 40 | **Ours** | 0.83 | 0.92 | **0.88** | **0.85** |
| | | Ensemble-DA | 1.00 | 0.11 | 0.60 | 0.73 |
| | | Thrifty-DA | 1.00 | 0.02 | 0.57 | 0.72 |
| Pushing (2 experts) | 25/25 | **Ours** | 1.00 | 0.52 | **0.77** | **0.82** |
| | | Ensemble-DA | 0.29 | 0.90 | 0.58 | 0.42 |
| | | Thrifty-DA | 0.58 | 0.66 | 0.62 | 0.62 |
| Plugging | 100 | **Ours** | 0.68 | 0.82 | **0.75** | **0.73** |
| | | Ensemble-DA | 0.22 | 0.27 | 0.14 | 0.37 |
| | | Thrifty-DA | 0.26 | 0.31 | 0.22 | 0.21 |

*Table 4 (paper Table II): Query decisions versus task outcomes over 100 rollouts, visuomotor tasks.*

Diff-DAgger has the best F1 and accuracy on every task. The paper reports averages across the simulated visuomotor tasks:

- Against Ensemble-DAgger: **+42.7% F1** and **+38.4% accuracy**.
- Against ThriftyDAgger: **+22.6% F1** and **+32.8% accuracy**.
- Across all tasks, the abstract summarizes this as a **39.0%** gain in failure prediction.

Notice the pattern in the baselines. Ensemble-DAgger and ThriftyDAgger often have a TPR of 1.00 but a TNR near zero on single-expert pushing. They ask for help almost every time. On two-expert pushing, the picture flips for Ensemble-DAgger: a TPR of 0.29 means it misses most failures.

**Videos.** The authors' project page shows diffusion-loss behavior on two tasks. (We link them rather than copy them.)

<video controls width="100%" src="https://diffdagger.github.io/static/videos/stack_dloss.mp4"></video>

*Video 1: Stacking with diffusion loss ([source](https://diffdagger.github.io/static/videos/stack_dloss.mp4), project page).*

<video controls width="100%" src="https://diffdagger.github.io/static/videos/push_dloss.mp4"></video>

*Video 2: Pushing with diffusion loss ([source](https://diffdagger.github.io/static/videos/push_dloss.mp4), project page).*

#### RQ2: Does the Collected Data Make a Better Policy?

Here every method gets the **same number of expert demonstrations**. We then train a final policy on the collected data and measure success rate. Simulation uses 100 evaluation episodes. Real-world tasks use 20.

| | Stack (sim) | | Push (1 exp.) | | Push (2 exp.) | | Plug | | Stack (real) | Load (real) |
|---|---|---|---|---|---|---|---|---|---|---|
| | State | Image | State | Image | State | Image | State | Image | Image | Image |
| BC | 0.89 | 0.85 | 0.89 | 0.72 | 0.74 | 0.71 | 0.75 | 0.49 | 0.60 | 0.80 |
| Ensemble-DA | 0.72 | 0.83 | 0.87 | 0.81 | 0.70 | 0.59 | 0.69 | 0.37 | – | – |
| Thrifty-DA | 0.79 | 0.86 | 0.85 | 0.81 | 0.69 | 0.68 | 0.65 | 0.47 | – | – |
| **Ours** | **0.94** | **0.94** | **0.96** | **0.87** | **0.87** | **0.79** | **0.92** | **0.84** | **0.80** | **0.95** |

*Table 5 (paper Table III): Success rate after the final DAgger iteration. Ensemble and Thrifty were not run on the real robot.*

Three things stand out:

- **Diff-DAgger wins every column.** On average it gains 13.7% over the second-best method in state-based tasks. In visuomotor tasks, including real-world ones, the average gain is 25.2%. (The introduction reports 20.6% overall.)
- **The baselines can lose to plain behavior cloning.** Stacking from state: BC gets 0.89, Ensemble-DAgger 0.72. Plugging from images: BC 0.49, Ensemble-DAgger 0.37. Interactive data was worse than offline data.
- **It works on real hardware.** Real stacking goes from 0.60 (BC) to 0.80. Loading goes from 0.80 to 0.95.

The authors' real-world demonstration video is on the project page:

<video controls width="100%" src="https://diffdagger.github.io/static/videos/real_world.mp4"></video>

*Video 3: Real-world demonstration ([source](https://diffdagger.github.io/static/videos/real_world.mp4), project page).*

#### RQ3: Is It Faster?

Yes. The Ensemble approach pays three costs that Diff-DAgger avoids:

1. **Training.** Five policies must be trained. Diff-DAgger trains one.
2. **Threshold setting.** Ensemble methods run full ensemble inference on the entire training set. Diff-DAgger passes a single batch per data point.
3. **Inference.** Five forward passes versus one.

{% include figure.liquid
   path="assets/img/2026-11-16-human-in-the-loop-systems/fig4_wallclock.png"
   class="img-fluid rounded z-depth-1"
   caption="Figure 7 (paper Fig. 4): Average duration of training, threshold setting, inference, and total iteration. Measured on the final DAgger iteration of the visuomotor pushing task, on one A100 40 GB GPU."
%}

| Speedup of Diff-DAgger over | Training | Threshold setting | Inference |
|---|---|---|---|
| Ensemble-DAgger | 5.1× | 48× | 3.8× |
| ThriftyDAgger | 6.4× | 53× | 3.9× |

Overall, the policy update time drops by a factor of **7.8** versus Ensemble-DAgger. A full update takes only about **0.54 hours**. ThriftyDAgger is slower still, because it collects extra data and trains a separate risk estimator.


#### False Positives vs. False Negatives

The two kinds of query mistakes do not cost the same.

- A **false positive** means the robot asks for help it did not need. You lose some of the benefit of interactive learning. In the worst case, you are just doing offline imitation.
- A **false negative** is worse. The robot is overconfident and fails. It then drifts into states far from the useful data.

Ensemble-DAgger and ThriftyDAgger both make many false negatives, especially with two pushing experts. The authors' guess: the training data already has high action variance, because the two experts disagree in the same state. That makes the ensemble numb to *real* out-of-distribution states. The authors also suspect this is why those baselines sometimes fall below behavior cloning. (This is a speculation in the paper, not a tested claim.)

Diff-DAgger has far fewer false negatives across tasks. That is the better trade for interactive learning with messy, multimodal data.

#### Seeing the Difference: The Plugging Task

{% include figure.liquid
   path="assets/img/2026-11-16-human-in-the-loop-systems/fig5_query_modes.png"
   class="img-fluid rounded z-depth-1"
   caption="Figure 8 (paper Fig. 5): Query behavior on plugging. Black = robot rollout, red = robot queries, green = expert intervenes, magenta = automatic timeout intervention."
%}

- **Diff-DAgger (top).** The charger starts to slip, and the robot asks for help at 9.1 s. The expert steps in and the episode succeeds.
- **ThriftyDAgger (middle).** The robot keeps trying after the slip. It only asks at 14.7 s, after dropping the charger.
- **Ensemble-DAgger (bottom).** The robot makes no progress and never asks. It times out at 25.0 s, and the system forces an expert in.

The authors' plugging video is here:

<video controls width="100%" src="https://diffdagger.github.io/static/videos/plug_desc.mp4"></video>

*Video 4: Plugging ([source](https://diffdagger.github.io/static/videos/plug_desc.mp4), project page).*

#### Choosing the Prediction Target

Diffusion models can be trained to predict different targets: the noise $\epsilon$, the clean sample, or a "velocity" (V-objective). The paper notes that this choice affects both convergence speed and final performance. They found that **sample** and **V-objective** prediction converge faster than epsilon. They use V-objective everywhere, since it balances convergence and performance.

#### An Unexpected Bonus: Recovery Behavior

{% include figure.liquid
   path="assets/img/2026-11-16-human-in-the-loop-systems/fig6_recovery.png"
   class="img-fluid rounded z-depth-1"
   caption="Figure 9 (paper Fig. 6): A policy trained on Diff-DAgger data. After a failed first attempt, it tries again and stacks the cube. Top: simulation. Bottom: real robot."
%}

Policies trained on Diff-DAgger data sometimes recover from their own mistakes. The robot misaligns the cube, fails, tries again, and succeeds. This kind of behavior usually shows up only if someone carefully plans recoveries into offline data. Here it appears naturally. The authors link it to the policy-performance gain over offline training. The mechanism makes sense: the expert steps in right at the moments of trouble, so the dataset ends up full of "how to fix it" examples.


### Limitations

**Stated by the authors:**

1. **Diffusion policies are still slow to train.** Diff-DAgger removes the ensemble cost, but a diffusion policy takes longer to train than a simpler policy class. The authors hope faster computing will ease this.
2. **Speed was bought with trade-offs.** Many choices that speed up training (such as a one-step observation horizon) cost some policy performance. The authors plan to look at this in future work.

**Our reading of what remains open** (not claims from the paper):

1. **Thresholds are set by hand.** $\alpha$ and $K$ differ by task (for example, 0.9 for two-expert pushing and 0.99 elsewhere). Picking them well may need tuning.
2. **Strong baselines in the real world are missing.** Ensemble-DAgger and ThriftyDAgger were only run in simulation (Table 5), so real-world comparisons are against BC alone.

### Takeaways


Diff-DAgger is a robot-gated DAgger method that reuses a diffusion policy's own training loss as an uncertainty signal. In the paper's experiments:

- It predicts task failure better than Ensemble-DAgger and ThriftyDAgger.
- It handles multimodal demonstrations, where ensemble disagreement gets confused.
- Its collected data trains better policies, in simulation and on a real Franka robot.
- It cuts wall-clock time per policy update by 7.8× versus Ensemble-DAgger.

The core idea is simple. A model trained to denoise expert actions is naturally surprised by actions unlike the expert data. That surprise is a cheap and well-behaved query signal.

---

## KnowNo: Robots That Ask For Help <d-cite key="ren_robots_2023"></d-cite> {#knowno}

Say you ask a kitchen robot to "put the bowl in the microwave." There are two bowls on the counter, one plastic and one metal. The robot's language model planner reads the instruction, picks the metal bowl with total confidence, and puts it in the microwave. Now there are sparks coming out of the microwave, and the robot is waiting for its next instruction.

As far as the planner could tell, the plan was fine. The problem is that the instruction was ambiguous, and nothing in the planner was built to notice that. Large language models (LLMs) are good at turning instructions into step-by-step plans, but they also hallucinate: they give answers that sound reasonable, are wrong, and come with high confidence. Instructions from people make this worse, since we leave out details all the time, sometimes on purpose and sometimes without realizing it.

<d-cite key="ren_robots_2023"></d-cite> propose KnowNo ("Know When You Don't Know"), a way for an LLM planner to figure out when it should stop and ask a person for help. Like the robot-gated DAgger methods above, the robot is the one deciding when a human gets involved. What changes is the kind of help. In Diff-DAgger, the expert takes over the low-level controls. In KnowNo, the human answers a question about the plan ("Which bowl did you mean?") and the robot keeps going on its own.

The authors call their goal *uncertainty alignment*, which has two parts:

1. **Calibrated confidence.** The user picks a success rate $1 - \epsilon$, and the robot asks for enough help to reach it, with a statistical guarantee.
2. **Minimal help.** Out of all the ways to reach that success rate, the robot should ask as few questions as possible.

A robot that asks about everything meets the first goal and misses the second. A robot that never asks misses the first. KnowNo uses conformal prediction to balance the two.

### Related Work

| Area | What prior work does | Where KnowNo differs |
|---|---|---|
| **LLM planners** | SayCan <d-cite key="ahn_saycan_2022"></d-cite> and similar systems turn language instructions into multi-step robot plans. | These planners have no way to ask for clarification. When the instruction is ambiguous, they guess. |
| **Asking through prompting** | Inner Monologue <d-cite key="huang_inner_2022"></d-cite> feeds scene descriptions and human feedback back into the prompt, which lets the LLM ask questions. | Getting the robot to ask the right amount takes a lot of prompt engineering, and there is no way to know ahead of time how often the robot will succeed. |
| **Uncertainty estimates for LLMs** | Many papers try to calibrate an LLM's confidence in a single answer. | A calibrated score for one answer still doesn't tell the robot which other answers to consider. Conformal methods return a set of answers with a coverage guarantee instead. |
| **Conformal prediction for LLMs** | <d-cite key="kumar_conformal_2023"></d-cite> apply conformal prediction to multiple-choice question answering. | That work assumes the answer options are given. KnowNo has the LLM write its own options, so it works for open-ended planning. |
| **Conformal prediction in robotics** | Earlier robotics work uses conformal prediction for fault detection, trajectory prediction, and planning around moving obstacles. | Those guarantees either hold on average at each step or are very conservative. KnowNo calibrates over entire plans, so the guarantee covers the whole task even though each action changes what the robot sees next. |

### How KnowNo Works

#### Step 1: Planning as Multiple Choice

The obvious way to measure an LLM's confidence in a plan step $y$ is the probability it assigns to the whole sentence, $p(y)=\prod_{i=1}^{k}p(\sigma_i \mid \sigma_1,\dots,\sigma_{i-1})$, where $\sigma_1, \dots, \sigma_k$ are the tokens in the step. The trouble is that this product shrinks as the sentence gets longer. "Put the plastic bowl in the microwave" scores lower than "put bowl in microwave" for reasons that have nothing to do with which one is correct.

KnowNo gets around this with a multiple-choice trick. KnowNo prompts the LLM to generate candidate next steps, formatted as multiple-choice question answering (MCQA) with options A–D and an added "E: an option not listed here" choice. The prompt includes a text description of the scene (from an object detector, or given directly), the user's instruction, and a few example options from other scenarios. The LLM is then asked which option is correct. The LLM's next-token probability for each letter serves as its confidence in that option. We write this score as $\hat{f}(x)_y$, the confidence in label $y \in \mathcal{Y} = \{A, B, C, D, E\}$ given the prompt $x$. Every option is now a single token, so length stops mattering.

| Option | Plan | Score $\hat{f}(x)_y$ |
|---|---|---|
| A | Put the plastic bowl in the microwave | 0.46 |
| B | Put the metal bowl in the microwave | 0.41 |
| C | Put the plastic bowl in the sink | 0.08 |
| D | Put the metal bowl in the sink | 0.03 |
| E | An option not listed here | 0.02 |

*Table 6: MCQA scores for "put the bowl in the microwave" (made-up numbers for illustration, not from the paper).*

The LLM leans toward A, but only slightly. Picking the top option here is close to a coin flip between the two bowls. The next step decides whether a lean like that is good enough to act on.

#### Step 2: Conformal Prediction

Using a calibration set of example scenarios, conformal prediction finds a confidence threshold that achieves a user-specified success level $1 - \epsilon$. In more detail:

1. Collect $N$ example scenarios and label the correct option for each one. The paper uses $N = 400$.
2. For each scenario, compute the nonconformity score $\kappa_i = 1 - \hat{f}(x_i)_{y_i}$. This is one minus the LLM's confidence in the *correct* answer, so a high $\kappa_i$ means the right answer surprised the LLM.
3. Set $\hat{q}$ to the $\frac{\lceil (N+1)(1-\epsilon) \rceil}{N}$ empirical quantile of $\kappa_1, \dots, \kappa_N$.

At test time, every option scoring above the threshold goes into a *prediction set*:

$$
C(x_{\text{test}}) = \{\, y \in \mathcal{Y} : \hat{f}(x_{\text{test}})_y \geq 1 - \hat{q} \,\}
$$

Conformal prediction guarantees that the correct option lands in this set with probability at least $1 - \epsilon$:

$$
\mathbb{P}\big(y_{\text{test}} \in C(x_{\text{test}})\big) \geq 1 - \epsilon
$$

The intuition is that $\hat{q}$ is picked so that in roughly $1 - \epsilon$ of the calibration scenarios, the correct answer scored at least $1 - \hat{q}$. A new scenario comes from the same distribution, so its correct answer should clear the same bar just as often. This holds whether or not the LLM's scores are any good. A poorly calibrated LLM just ends up with bigger sets, which means more questions. If the scores do match the true probabilities, the sets are also the smallest possible for that success level, which is where the minimal help goal comes in.

One detail: the guarantee above is an average over random draws of the calibration set. The paper uses a stronger version that holds for the specific calibration set you collected (with probability $1 - \delta = 0.99$), so the robot doesn't need a fresh calibration set for every new task.

Back to the example: if calibration gave a threshold of $1 - \hat{q} = 0.30$, the prediction set would be {A, B}. The sink options are ruled out, but the robot still can't tell which bowl you meant.

#### Step 3: Asking for Help

- If the set contains exactly one option, the robot executes it.
- If the set contains more than one option, the robot asks the human to choose. The options are turned into a plain-language question, like "Should I put the plastic bowl or the metal bowl in the microwave?" If the right answer is E, the human describes the correct action instead.

Assuming the human always answers correctly, the robot can only fail when the correct option is missing from the set. That happens at most $\epsilon$ of the time, so the task success rate is at least $1 - \epsilon$. In other words, the coverage guarantee from conformal prediction turns into a task success guarantee.

#### Multi-Step Plans

Most tasks take more than one step, which breaks a key assumption of conformal prediction: that every data point is independent. In a multi-step plan, what the robot sees at step $t$ depends on what it did at earlier steps.

KnowNo handles this by calibrating on whole plans instead of single steps. A plan with steps $t = 0, \dots, T-1$ gets the score of its weakest step:

$$
\hat{f}(\bar{x})_{\bar{y}} = \min_{t} \hat{f}(x^t)_{y^t}
$$

where $\bar{x}$ and $\bar{y}$ are the sequences of prompts and correct labels. Calibration runs on these sequence scores and produces a single threshold $\hat{q}$. At test time the robot can't see future steps, so it builds a prediction set at each step using that same $\hat{q}$. The authors prove that these step-by-step sets contain exactly the same plans as a set built from the whole sequence at once. So if the robot asks for help whenever any step's set has more than one option, the $1 - \epsilon$ guarantee covers the entire task rather than each step separately.

The paper also extends this to cases with more than one acceptable option, like "bring me a soda" when there is both a Coke and a Sprite on the table. In that setting, the guarantee is that the set contains at least one acceptable option.

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

### KnowNo Results

1. **Reliable task success guarantees.** In simulation, with a target success level of 85% ($\epsilon = 0.15$), KnowNo had the smallest deviation from the target across all three ambiguity settings. The uncalibrated baselines either missed the target or gave the user no way to set one.
2. **Minimizing human help rate.** For the same success level, KnowNo asked for help less often than the baselines. The gap grew as instructions became more vague. Ensemble Set also performed well, but needed more inference time.
3. **Quantitative hardware performance.** On the UR5, KnowNo reduced human help by 14% per step and 8% per trial compared to Simple Set. On the mobile manipulator, it reduced help by 14%. Overall, KnowNo reduced human help by 10–24% compared to the baselines.
4. **Ablation on scoring functions.** Replacing the MCQA letter scores with sentence-level scores (raw sentence probability or perplexity) made performance worse, especially when candidate options varied a lot in length. Scoring a single letter avoids this length bias.

### Conclusions

Conformal prediction paired with MCQA allows operators to tune a statistical guarantee on task completion, while keeping human help to a minimum. KnowNo works with off-the-shelf LLMs without fine-tuning.

The key limitation is that the guarantee currently assumes:

1. The environment's scene state is accurately parsed into text.
2. Low-level robotic controllers execute the selected primitive without failure.
3. The human always answers correctly. The method does not account for human error, so a wrong answer from the human breaks the guarantee.

In addition, calibration requires a labeled dataset of scenarios. The authors used 400 examples, which took about 4 hours to label in the multi-step setting, and this must be repeated for each new environment.

---

## Student Q&A {#student-q-a}

### KnowNo {#qa-knowno}

**Q: How can robots learn from past clarifications to reduce repeated questions and adapt to individual users?**

A: Maybe we can create a memory bank of human clarifications. The next time a similar situation is faced, we can retrieve the past record from the memory bank, to reduce the frequency of asking the same question in the future.
<br><br>

**Q: What role could another LLM play in helping robots resolve uncertainty, and when would human clarification still be necessary?**

A: Perhaps another LLM could be used to help resolve uncertainty. The 2 models can discuss with each other, and if an agreement can't be reached, they can ask a human for help. This may make the decision process slower, and may ultimately end up just following the decision of the stronger LLM. If the LLMs are about equal, then the additional LLM does not really help.
<br><br>

**Q: How can KnowNo be adapted when the calibration set differs in distribution from the deployment environment?**

A: Several studies mention how we can dynamically change the calibration set for the application.
<br><br>

### Diff-DAgger {#qa-diff-dagger}

**Q: If diffusion loss measures how well an action matches the learned distribution rather than whether the action will succeed, is diffusion loss alone sufficient for reliable failure prediction?**

A: No, because diffusion models just measure how an action matches the distribution of the dataset. It measures the uncertainty of the robot. However, if an action is not in the distribution, that does not necessarily mean it failed the task.
<br><br>

**Q: What types of expert interventions would be most valuable for improving a robot's policy?**

A: Motion planner interventions may be more optimal than human expert interventions, but human expert interventions may be more diverse. There's a tradeoff between the two. A motion planner can help you find the shortest trajectory, but a human expert can help reach out-of-distribution states (helping the policy to be more generalized). This paper has humans intervene when the policy is uncertain (the action hasn't been executed yet), but it is also important to learn how to recover from failure. If demonstration data only includes successful trajectories, the robot doesn't learn how to recover from failure. However, failing on purpose can also cause cost and safety problems. It is an interesting problem to consider whether to intervene before or after failure.
<br><br>

### Model Uncertainty

**Q: What does it mean for a robot to be uncertain? How is uncertainty different from the risk of task failure?**

A: Uncertainty can mean a lot of things, such as unsafe or unwanted actions. The definition of uncertainty depends on the situation or environment the robot is in.
<br><br>

### Sources

- Lee, Kang, Kuo. *Diff-DAgger: Uncertainty Estimation with Diffusion Policy for Robotic Manipulation.* ICRA 2025. [arXiv:2410.14868](https://arxiv.org/abs/2410.14868)
- Project page and videos: [diffdagger.github.io](https://diffdagger.github.io)
- Ren et al. *Robots That Ask For Help: Uncertainty Alignment for Large Language Model Planners.* CoRL 2023. [arXiv:2307.01928](https://arxiv.org/abs/2307.01928)
- Project page: [robot-help.github.io](https://robot-help.github.io)
