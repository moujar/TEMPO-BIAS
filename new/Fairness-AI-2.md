# 3.2 Bias Analysis Over Time

This section introduces our longitudinal evaluation framework, whose
objective is to determine whether political bias in Large Language
Models (LLMs), previously measured at a single time point, evolves
across successive model releases and alignment stages. Instead of
treating models as static artifacts, we view them as temporal systems
whose ideological neutrality may decay or improve as training
strategies, scaling laws, and alignment pipelines evolve over time. The
goal of this analysis is to quantify whether Prediction Inconsistency
(IC) remains stable, increases, or decreases across model generations.\
Our temporal evaluation is conducted on the same set of political
entities, sentence templates, and Target-Oriented Sentiment
Classification (TSC) setup described in Section 3.1. This reuse ensures
that the only changing variable in our experiment is the model version,
allowing a controlled assessment of temporal bias dynamics.\
A visual overview of the experimental pipeline is presented in Figure X.

![A diagram of a diagram AI-generated content may be
incorrect.](media/image1.png){width="3.57922353455818in"
height="5.26115157480315in"}

## 3.2.1 Model Reuse and Temporal Extension

Rather than modifying the workload or introducing a new sentiment
evaluation procedure, we build directly on the previously established
TSC pipeline. All inference procedures, prompt structures, entity
substitutions, and IC computations follow the methodology defined in
Section 3.1, with no alterations to data or prompting design. This
ensures comparability between static and temporal evaluations.

The key extension is the introduction of model lineages. For each model
family F (e.g., LLaMA, Qwen, Mistral), we collect a chronologically
ordered sequence of versions {V₁, V₂, ..., Vₙ}. For each version Vᵢ, we
compute an IC score using the IC metric defined in Section 3.1.3,
producing a temporal sequence:

> S_F = { IC(V₁), IC(V₂), ..., IC(Vₙ) }

Each IC value in the sequence reflects the static bias of a single
version, while the sequence as a whole reflects the evolution of bias
across releases.

## 3.2.2 Longitudinal Metrics

To characterize how IC evolves over time, we introduce three metrics:
Bias Velocity, Alignment Delta, and Cross-Family Convergence.

### (A) Bias Velocity

Bias Velocity quantifies the rate of change in political bias across
consecutive versions. For two successive releases Vᵢ₋₁ and Vᵢ, we
compute:

> β(Vᵢ) = ( IC(Vᵢ) − IC(Vᵢ₋₁) ) / Δtᵢ

where Δtᵢ denotes the time interval between releases. A negative β
indicates bias decay (improved neutrality), while a positive β indicates
bias accretion (worsened neutrality).

### (B) Alignment Delta

LLMs are often released in both Base and Chat variants. To isolate the
effect of alignment tuning, we compute:

> Δₐₗₙ(Vᵢ) = IC(Vᵢ\^{chat}) − IC(Vᵢ\^{base})

A negative Δₐₗₙ indicates that alignment mitigates political bias, while
a positive Δₐₗₙ indicates that alignment introduces or amplifies bias.

### (C) Cross-Family Convergence

To assess ecosystem-wide dynamics, we measure whether model families
converge toward similar neutrality levels. At time index s, we compute:

> C(s) = Var( IC(F₁,s), ..., IC(Fₖ,s) )

A decreasing C(s) implies convergence toward shared neutrality norms,
whereas an increasing C(s) implies persistent geopolitical or
training-driven divergence.

## 3.2.3 Experimental Fit to Metrics

The outputs of our temporal experiment directly populate the metrics
above:\
1. IC sequences support computation of Bias Velocity (Eq. 5)\
2. Base/Chat IC pairs support computation of Alignment Delta (Eq. 6)\
3. Cross-family IC values support computation of Convergence (Eq. 7)\
\
Together, these components enable temporal inference about political
bias trajectories across the LLM ecosystem.
