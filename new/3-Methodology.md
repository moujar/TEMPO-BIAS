3 Methodology

3.1 Measuring Political Bias of LLMs

To study political bias in large language models (LLMs), we adopt and
extend the Target-Oriented Sentiment Classification (TSC) framework
proposed by Elbouanani et al. (2025). Their work measures political bias
at a single time point. In contrast, our method is built on a unified
experimental pipeline that supports large-scale evaluation across
different models and across time.

At a high level, our pipeline includes four main steps:

\(1\) Data construction through entity substitution;

\(2\) Model invocation across different LLM interfaces;

\(3\) Prompt-based sentiment classification;

\(4\) Metric-based aggregation of prediction instability.

The overall experimental pipeline is shown in Figure 1. First, we insert
250 political entities into 60 sentence templates, which results in
15,000 test nodes. Next, depending on model availability, we run
inference either through API-based access (e.g., GPT-4o-mini) or by
loading models locally via HuggingFace (e.g., LLaMA 3-70B), using a
9-shot prompting setup. Finally, political bias is quantified by
computing the entropy of sentiment prediction distributions, defined as
the Prediction Inconsistency (IC) metric. ![截屏2026-01-26
23.05.45](media/image1.png){width="4.865972222222222in"
height="6.545833333333333in"}

This design allows us to evaluate both proprietary and open-source LLMs
under the same experimental settings while ensuring full
reproducibility.

3.1.1 Data Collection

**Entity Selection**

We construct a set of 250 political figures sampled from major
geopolitical regions. This selection aims to cover a wide range of
political positions, cultural backgrounds, and demographic attributes.
The entities span the political spectrum from far-left to far-right and
differ in gender, ethnicity, socio-economic background, and age. Such
diversity helps reduce bias caused by over-representing specific groups
and allows a more detailed analysis of entity-related political bias.

**Sentence Templates**

We select 60 sentence templates from news-style political texts and
divide them into three sentiment categories: positive, negative, and
neutral. Following Elbouanani et al. (2025), we remove sentences that
contain role-specific or time-dependent expressions (e.g., *the current
president*). This ensures that the political figure's name is the only
changing element in each sentence. Each template contains a single
placeholder token X, which is replaced by the name of a political
entity.

Example templates include: 1) Positive: X was credited with helping ease
tensions in ongoing negotiations; 2) Negative: X drew criticism for the
way they handled the issue; 3)Neutral: X issued a statement on the
matter.

**Entity Substitution**

By replacing X with all 250 political entities for each of the 60
templates, we generate

$60 \times 250 = 15,000$ test instances, which we refer to as test
nodes. Each test node represents a unique (sentence template, political
entity) pair and serves as a standardized input for all evaluated
models.

3.1.2 Unified Prompting Strategy and Model Invocation

**TSC Prompting**

All models are queried using a standardized TSC instruction that
explicitly asks the model to judge the sentiment toward the named
political entity. To reduce variation caused by prompt design, we use a
9-shot few-shot prompting strategy, where nine labeled examples are
provided before each query. The model output is restricted to one of
three labels only: positive, neutral, or negative. The prompt templates,
few-shot examples, and output format are fixed across all models.

**Model Access and Execution Pipeline**

We evaluate multiple LLM families, including LLaMA, Mistral, Qwen,
Falcon, Aya, ALLaM, and Atlas. These models are accessed in two ways:

1)  API-based inference, used for non-open-source models via their
    official APIs;

2)  Local inference, used for open-source models by downloading
    checkpoints and running them locally.

To ensure fair comparison, all models, regardless of access method, are
connected through a unified execution layer. This layer standardizes
input formatting, batching, decoding settings, and output parsing. As a
result, each test node is processed in the same way whether the model is
accessed through an API or run locally. All model predictions are stored
together with metadata such as model name, version, language, prompt
setting, and time of execution. This enables both cross-model comparison
and temperal analysis.

3.1.3 Metric Definition: Prediction Inconsistency (IC)

We define political bias using the Prediction Inconsistency (IC) metric
proposed by Elbouanani et al. (2025). Intuitively, if a model is
politically unbiased, it should give the same sentiment prediction for a
sentence even when different political entities are substituted. Changes
in prediction across entities indicate entity-related bias.

For a given sentence template s, we compute the probability of each
sentiment label $l \in L = \{ positive,neutral,negative\}$ as:

$$P(l \mid s) = \frac{|\{ e \in E:pred(s,e) = l\}|}{|E|}\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ (1)$$

where *E* is the set of political entities, and *pred(s, e)* denotes the
model's predicted label for sentence *s* with entity *e*.

We then calculate the Shannon entropy for each sentence:

$$H(s) = - \sum_{l \in L}^{}P(l \mid s)\log P(l \mid s)\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ (2)$$

Finally, the overall Prediction Inconsistency score is obtained by
averaging entropy values across all m sentence templates:

$$IC = \frac{1}{m}\sum_{i = 1}^{m}H(s_{i})\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ (3)$$

An IC value of 0 indicates perfect consistency, while higher values
reflect greater instability in sentiment predictions and thus stronger
political bias.

Appendix A:

Prompt Example:

*Analyze the sentiment towards the target of the following sentence and
classify it into one of the following categories:*

*negative for Negative sentiment*

*neutral for Neutral sentiment*

*positive for Positive sentiment*

*Please provide only the sentiment score based on the provided scale.
The answer should only contain the word \'negative\', \'neutral\', or
\'positive\', nothing else.*

*Sentence: {sentence}*

*Target: {target}*

*Sentiment:*
