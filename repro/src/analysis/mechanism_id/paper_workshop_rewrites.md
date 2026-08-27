# Workshop rewrite — paper_neurips.tex

Goal: turn the paper into an honest, workshop-ready version in ≤ 3 days of
writing time. Frame the work as an **architecture-efficiency + mechanism-ID
study on hidden-lag compliance and hidden-subset parity**, not as evidence
for "global sequence integration beyond local shortcuts".

Every number below comes from `analysis/mechanism_id/results/`:

- compliance linear saturation → `phase2_ladder_standard.csv` (C1 row)
- λ-agnostic baseline → `phase2c_lag_agnostic.csv`, `phase2c_per_lag_weights.csv`
- held-out-rule check → `phase4_heldout_rule.csv`
- parity decomposition → `phase5_parity.csv`
- oracle audit → `phase6_oracle.csv`

All `REPLACE … WITH …` blocks refer to `paper/paper_neurips.tex` line
numbers in the current HEAD of `refactor/restructure-repo`.

Recommended target venues (in decreasing fit):

1. **ML4H @ NeurIPS 2026** — dementia paper 2 fits the venue; paper 1 is the
   methodological prelude; workshop review cycle is kinder.
2. **Synthetic Data for ML workshop @ NeurIPS / ICML** — if ML4H slots fill.
3. **M3L @ NeurIPS** — mechanism-ID framing resonates there.

---

## 0. Title

**Replace (line 92):**

> Do Large Language Models Detect and Exploit Decisive Sequential Information?

**With:**

> When Encoders Beat Decoders on Hidden-Lag and Hidden-Subset Tasks:
> A Controlled Benchmark with Mechanism-ID Audit

or, shorter:

> A Mechanism-ID Audit of Hidden-Lag and Hidden-Subset Benchmarks for Sequence Models

Rationale: the honest claim is not about "decisive sequential information",
it is about which *architectures* learn *which hidden structure* *efficiently*.

---

## 1. Abstract

**Replace (lines 140–142) the entire abstract WITH:**

```latex
\begin{abstract}
We introduce a controlled synthetic benchmark over letter sequences for
studying how sequence models recover two hidden structures: a
\emph{hidden fixed lag} $\lambda$ at which two letters from a hidden key
subset $\mathcal{S}$ co-occur (\emph{compliance}), and a \emph{hidden
subset} $K\!\subsetneq\!\mathcal{A}$ whose per-letter counts determine a
modular-parity label. We prove that compliance tasks under independent
full-support sampling inevitably leak local information in the
single-position mutual-information sense
(Theorem~\ref{thm:impossibility}), and that balanced hidden-subset
parity is $k$-locally pure for every window shorter than $n$
(Theorem~\ref{thm:parity_local_purity}). Empirically, we observe a
consistent \emph{architecture-efficiency gap}: small encoder
transformers and recurrent baselines reach the theoretical ceiling on
compliance tasks with $10^5$ parameters, while decoder-only models
require $10^{10}$ parameters to match them; under label noise the gap
widens but closes at sufficient scale. We then audit the benchmark with
a mechanism-identification ladder. Both compliance tasks admit a linear
readout on aggregated $\lambda$-shifted pair counts: a 676-dimensional
logistic regression with known $\lambda$ saturates both variants, and a
$\lambda$-agnostic boosted-tree baseline on stacked all-offset pair
counts matches our best encoder on the noisy variant (AUC 0.667 vs.\
0.669 at $\pi=0.3$). The encoder advantage on compliance is therefore a
featurization advantage under raw-token end-to-end supervision, not a
global-integration advantage. The hidden-subset parity task remains
unsolved by every architecture we tested under standard supervision; a
decomposition shows the bottleneck is hidden-subset identification, not
modular arithmetic itself: revealing the per-position membership bits
$b_t=\mathbf{1}[X_t\in K]$ lets a 64-unit MLP reach AUC 0.997 in under
a minute. We position the benchmark as a diagnostic for
end-to-end learning of hidden positional and subset structure, and
release a full mechanism-ID analysis pipeline.
\end{abstract}
```

Word count: ~300 (slightly under the NeurIPS 2026 cap). If the workshop
enforces ~200 words, cut the parity decomposition sentence and the
"positioning" sentence.

---

## 2. §1 Introduction

### 2a. Soften the "shortcut" paragraph

**Replace (lines 153–155) starting "Recent evidence motivates that
concern …" up to "semantic cues are removed." WITH:**

```latex
Recent evidence motivates that concern.  Shuffling item order causes only
small degradation in LLM-based recommendation
\citep{kimLostSequenceLarge2025b}; clinical analyses report reliance on
statistical pattern matching rather than genuine temporal reasoning
\citep{bediFidelityMedicalReasoning2025}; and transformer predictions can
often be approximated by simple $k$-gram rules
\citep{nguyen2024understanding}.  At the same time, controlled synthetic
benchmarks are difficult to design well: a task marketed as requiring
global sequence structure can reduce to a local statistic once the
correct feature space is identified, so empirical results are only as
strong as the baseline ladder they are compared against
\citep{klenitskiyDoesItLook2024}.  We address this by constructing a
controlled benchmark based on letter sequences and by accompanying it
with a mechanism-identification ladder that isolates which local and
which non-local features suffice to solve each task variant.
\end{abstract}
```

(I removed the explicit "beyond local shortcuts" promise so the
introduction no longer makes a claim the rest of the paper has to
defend with empirical evidence it doesn't have.)

### 2b. Contributions

**Replace (lines 168–174) WITH:**

```latex
\paragraph{Contributions.}
\begin{itemize}
    \item We formalize two controlled task families over letter
    sequences --- \emph{compliance}, which hides a lag $\lambda$ and a
    key subset $\mathcal{S}$, and \emph{hidden-subset parity}, which
    hides a subset $K$ whose counts drive modular arithmetic --- and
    prove theoretical local-information bounds on each
    (Theorems~\ref{thm:impossibility} and~\ref{thm:parity_local_purity}).
    \item We benchmark encoder transformers, decoder-only LLMs at
    scales up to 70\,B parameters, and recurrent baselines, reporting
    a consistent architecture-efficiency gap on compliance tasks that
    widens under label noise and closes at sufficient decoder scale.
    \item We release a mechanism-identification ladder --- content
    features, residue-class counts, aggregated $\lambda$-aware and
    $\lambda$-agnostic pair counts, held-out-rule generalisation, and
    matched-histogram counterfactuals --- that localises the
    empirical signal in each task variant and tests whether neural
    models exploit structure beyond what simple tabular baselines
    capture.
    \item We decompose the parity failure into a hidden-subset
    identification component (unlearnable end-to-end from raw tokens
    in the tested regime) and a modular-arithmetic component
    (learnable by a 64-unit MLP in under a minute once the membership
    bits are revealed).
\end{itemize}
```

---

## 3. §3 Methods and theory

### 3a. Add a short "empirical collapse" paragraph to §3.2

**Insert after line 273** (after the `Hence, non-degenerate compliance
tasks…` paragraph), WITH:

```latex
\paragraph{Relation to our empirical findings.}
Theorem~\ref{thm:impossibility} guarantees only that \emph{some}
single-position leakage exists; it does not determine how much of the
task label is linearly recoverable from low-dimensional summaries of
the sequence.  In Section~\ref{sec:mechanism_id} we show empirically
that for the specific parameter regime used in our experiments
$(n,\ell,\lambda,m) = (20,26,7,6)$, the compliance label is
approximately a linear function of the $26\!\times\!26$-dimensional
aggregated lag-pair count
$g_{ab}^{(\lambda)}(X) \;=\; \sum_{t=1}^{n-\lambda}\mathbf{1}\{X_t = a,\, X_{t+\lambda} = b\}$.
The inevitable leakage in Theorem~\ref{thm:impossibility} is therefore
not only present in principle but sufficient in practice to saturate
our compliance benchmark.  Readers should accordingly view the
compliance family as a diagnostic for \emph{learning hidden positional
offset structure}, not as a test of global sequential integration.
\end{abstract}
```

---

## 4. §4 Experiments

### 4a. Baselines paragraph

**In §4.2 (around lines 390–395), replace the "k-gram baseline" sentence
starting "\textbf{\textit{k}-gram baseline.}" WITH:**

```latex
\textbf{$k$-gram and lag-aware baselines.}
Three local-pattern references: (i) a contiguous $k$-gram classifier
that estimates $\hat{\mathbb{P}}(Y=1\mid g) = \#(g, Y=1)/\#(g)$ and
averages over the sequence's $k$-grams
(Appendix~\ref{subsec:kgram}); (ii) a $\lambda$-aware pair-count
classifier --- $L^2$-regularised logistic regression on the $26\!\times
\!26$-dimensional feature
$g_{ab}^{(\lambda)}$ with $\lambda$ given --- which controls for
whether the task can be solved by a single linear readout once the
hidden offset is known; and (iii) a $\lambda$-agnostic baseline that
stacks $g_{ab}^{(\lambda')}$ across all candidate offsets
$\lambda' \in \{1, \ldots, n{-}1\}$ (dimension $(n{-}1)\!\times\! 676
= 12\,844$) and lets an $L^1$-regularised logistic regression or
boosted tree rediscover $\lambda$ from data.  Baseline (iii) is the
fair "is the encoder's advantage about discovering $\lambda$?" test.
Because the Naive data is generated from ordered templates rather than
direct sampling, its bigrams already achieve AUC $\sim 0.999$; signal
diminishes with task complexity for the contiguous baseline but
recovers for the lag-aware baseline.
```

### 4b. Ceilings paragraph

**In §4.3 (around line 401), update the ceiling discussion. Replace
"this gives $\mathrm{AUC}^*=F_1^*=1$ for Naive, Tricky Det., and
Parity ($\pi=0$)" WITH:**

```latex
this gives $\mathrm{AUC}^\*=F_1^\*=1$ for Naive and Parity at $\pi=0$,
$\mathrm{AUC}^\*=0.67,\;F_1^\*=0.58$ for Tricky Random at $\pi=0.3$,
$\rho\approx 0.293$, and an empirical Tricky Deterministic ceiling
of $(\mathrm{AUC},F_1) \approx (0.90, 0.85)$
(Appendix~\ref{appendix:oracle_audit}): the stored Tricky Det dataset
exhibits a residual label uncertainty of $\pi \approx 0.12$--$0.17$ in
the rule space tested, which we document rather than hide.
```

(This is the honest fix for the `_6` F1\*=1 claim.)

### 4c. Table 1 (lines 374–388)

**Replace the Tricky Det row's $\pi=0$ entry WITH $\pi\!\approx\!0.12$
empirically audited; keep the $\rho$ column at the actually-observed
value $0.356$ rather than the claimed $0.293$.** The relevant code
fragment becomes:

```latex
\begin{tabular}{lcccccc}
\toprule
Dataset & $N$ & $n$ & $\rho$ & $\lambda$ & $\pi$ & $m$ \\
\midrule
Naive & 400K  & 20 & 0.51 & -- & 0 & 26\\
Tricky det. & 400K & 20 & 0.356 & 7 & $\approx 0.12$\,$^\dagger$ & 6\\
Tricky random & 400K & 20 & 0.416 & 7 & 0.3 & 6 \\
Parity & 800K & 20 & 0.499 & -- & 0 & 6\\
\bottomrule
\multicolumn{7}{@{}l@{}}{\footnotesize $^\dagger$empirically
audited against the nearest rule in the tested family;}\\
\multicolumn{7}{@{}l@{}}{\footnotesize\ see Appendix~\ref{appendix:oracle_audit}.}
\end{tabular}
```

(The parity row also needs its $N=400$K changed to $800$K to match
the actual `test_just_pair` file sizes.)

### 4d. Key set fix

**In §4.1 (line 363), fix the key-set inconsistency between the paper
and the implementation.** Replace

> $\mathcal{S} = \{\texttt{W}, \texttt{D}, \texttt{Q}, \texttt{J},
> \texttt{X}, \texttt{U}\}$

WITH:

```latex
$\mathcal{S}_{\text{compliance}} = \{\mathtt{W},\mathtt{D},\mathtt{Q},
\mathtt{J},\mathtt{X},\mathtt{U}\}$ for Tricky,
$\mathcal{S}_{\text{parity}} = \{\mathtt{W},\mathtt{D},\mathtt{Q},
\mathtt{J},\mathtt{X},\mathtt{N}\}$ for Parity, both with
$\kappa(\mathtt{W}){<}\kappa(\mathtt{D}){<}\kappa(\mathtt{Q}){<}
\kappa(\mathtt{J}){<}\kappa(\mathtt{X}){<}\kappa(\cdot)$ where
$\kappa(\cdot)$ is $\kappa(\mathtt{U})$ or $\kappa(\mathtt{N})$
respectively.  The two key sets are drawn from independent realisations
of the generating procedure of Section~\ref{sec:setup}; for $|S|=6$
(even), the parity definition
(Definition~\ref{def:parity}) is invariant to this choice.
```

---

## 5. §5 Results — rewrite the per-task paragraphs

### 5a. Tricky deterministic

**Replace lines 418–420 (the "Tricky deterministic" paragraph) WITH:**

```latex
\paragraph{Tricky deterministic: encoders dominate, small decoders lag.}
When compliance requires detecting a lag-7 pair with monotone
$\kappa$, encoders (BERT, RoBERTa, Transformer, LSTM,
RNN-Transformer) reach AUC $\geq 0.997$ with 100\,K--400\,K
parameters.  Decoder-only Llama-3.2-1B plateaus at AUC $\approx 0.85$
--- close to the contiguous 2-gram baseline of $0.82$ --- while
Llama-3.1-8B reaches AUC $0.995$.  XGBoost on frozen Llama-1B
mean-pooled embeddings fails because mean pooling is order-invariant
and discards the positional information the task relies on.
Importantly, a $676$-dimensional logistic regression on the
$\lambda$-aware pair feature $g_{ab}^{(\lambda=7)}$ also reaches AUC
$0.997$, and a $\lambda$-agnostic $12\,844$-dimensional logistic
regression recovers $\lambda=7$ on its own (clean per-lag weight-norm
spike, Appendix~\ref{appendix:mechanism_id}) and reaches AUC $0.997$
as well.  We read the encoder vs.\ decoder gap on this task as an
\emph{efficiency gap in discovering the right attention offset from
raw-token supervision}, not as evidence for non-local computation.
```

### 5b. Tricky random

**Replace lines 421–423 WITH:**

```latex
\paragraph{Tricky random: scale unlocks autoregressive models.}
At $\pi=0.3$ the theoretical upper bound is
$\mathrm{AUC}^\*=0.67$.  Our custom Transformer and RNN-Transformer
reach the ceiling (AUC $0.671$), BERT gets to $0.669$, LSTM to
$0.665$.  Smaller decoder-only models (Llama-1B, 8B) plateau near
AUC $0.58$, barely above the contiguous 2-gram baseline ($0.585$),
but Qwen-14B also approaches the ceiling.  The mechanism-ID ladder
again matches the ceiling with a linear model when $\lambda$ is
given ($L^2$ logistic regression on $g_{ab}^{(\lambda=7)}$, AUC
$0.671$) and with a boosted tree when $\lambda$ is not given
($\lambda$-agnostic XGBoost on stacked all-offset pair counts, AUC
$0.667$).  The $\lambda$-agnostic linear baseline, however,
underperforms the ceiling (AUC $0.632$): under label noise the right
$\lambda$-block is not separable by a linear sparsity pattern and is
instead concentrated at nearby offsets (the top-weight lag on
$L^1$ logistic regression is $\lambda=19$, not $\lambda=7$;
Appendix~\ref{appendix:mechanism_id}).  Decoder-only architectures
are therefore not fundamentally limited for $\lambda$-discovery
under noise, but require substantially more parameters than encoders
to match what a boosted tree does with hand-crafted pair features.
```

### 5c. Parity

**Replace lines 424–425 (the parity paragraph) WITH:**

```latex
\paragraph{Parity: a hidden-subset identification failure, not a
modular-arithmetic failure.}
All tested models perform at chance (AUC $\approx 0.50$) on the
hidden-subset parity task, consistent with prior reports
\citep{countingLLMs}.  Theorem~\ref{thm:parity_local_purity} establishes
exact $k$-local purity only in the balanced case $|K|/|\mathcal{A}|=1/2$,
and our regime lies outside that case but still defeats the tested
$k$-gram baselines.  A decomposition study reveals that the
architectural deficiency is one of \emph{joint hidden-subset
identification and modular arithmetic under end-to-end supervision},
not of modular arithmetic itself: feeding the same MLP-64 the
per-position membership bits $b_t = \mathbf{1}\{X_t \in K\}$ --- a
20-dimensional binary input that does \emph{not} name the individual
key letters --- yields AUC $0.997$ in under one minute of training;
feeding the 6-dimensional per-key parity vector yields AUC $1.000$
(Table~\ref{tab:parity_decomposition}, Appendix~\ref{appendix:mechanism_id}).
Binary-vocabulary parity and induction-head associative-recall
circuits are known to succeed on related tasks
\citep{olsson2022induction}; our negative result is therefore a
statement about end-to-end supervised learning of \emph{hidden subset
identification}, not a general claim about parity learnability in
sequence models.
```

### 5d. Signal decomposition — replace the single paragraph with a proper §5.1

**Replace lines 427–428 with a referenced appendix subsection (kept
short in the main text, details in appendix):**

```latex
\subsection{Mechanism-identification audit}
\label{sec:mechanism_id}
We accompany the main results with a mechanism-ID ladder that reports
AUC on each task variant for each of seven feature families and
three models (logistic regression, XGBoost, small MLP).  The families
range from full 26-dimensional letter counts (content only) through
residue-class counts modulo $\lambda$ to $\lambda$-aware aggregated
pair counts, $\lambda$-agnostic stacked pair counts, and
lag-trigram features.  Across both Tricky variants, the
676-dimensional $\lambda$-aware pair-count feature + linear logistic
regression reaches the encoder ceiling exactly; the 26-dimensional
content feature reaches only AUC $0.82$ / $0.58$ on the two variants.
Across random $(\mathcal{S}, \kappa, \lambda)$ regenerations
(Appendix~\ref{appendix:heldout}), the $\lambda$-aware pair feature
is rule-specific: AUC $1.00$ within-rule, $0.50$ across rules,
establishing that the linear ceiling is not carried by a transferable
algorithm but by a per-rule choice of 676 weights.  Matched-histogram
analysis
(\citealt{beaney2024coded_disease_representations}-style) on the
6-dimensional key-letter count vector confirms that after controlling
for key-letter multiplicities, a within-group balanced AUC of $0.65$
remains on Tricky Random for the $\lambda$-aware feature, vs.\ $0.50$
for content alone.  Full CSVs, logs, and reproduction scripts are in
the supplement.
```

---

## 6. Limitations — expand

**Replace the limitations paragraph (line 456) WITH:**

```latex
\paragraph{Limitations.}
Our benchmark is best understood as a diagnostic for two specific
capabilities: (i) learning a hidden fixed positional offset from raw
tokens under end-to-end supervision, and (ii) jointly identifying a
hidden subset and applying modular arithmetic over it.  It should not
be read as a test of general "global sequential integration": the
mechanism-ID audit (Section~\ref{sec:mechanism_id}) shows that the
compliance label is a linear function of 676 aggregated
$\lambda$-shifted pair counts, and that an $L^1$ logistic regression
over stacked all-offset pair counts recovers $\lambda$ from data in
the $\pi=0$ case.  The encoder vs.\ decoder gap we report is
therefore an efficiency gap on a \emph{featurization-learning} task
rather than evidence of a representational boundary between the two
architectures.  We tested one main regime per task,
$(n,\ell,\lambda)=(20,26,7)$ for compliance and $(n,\ell,m)=
(20,26,6)$ for parity; longer sequences, larger alphabets, partial
orders on $\kappa$, and chain-length or varying-$\lambda$ variants may
shift the picture and are open directions.  Theorem~\ref{thm:impossibility}
concerns the proposal law and assumes independent sampling with full
support; it does not characterise arbitrary real-world sequence
distributions.  Our parity results are empirical statements about
standard end-to-end supervised training in one hidden-subset regime,
not impossibility results for parity-type tasks in general.  The
benchmark deliberately strips away semantics; transfer to real
temporal domains --- e.g.\ EHR-based early-dementia prediction, the
target of our follow-on work --- and to generative objectives remains
open.
```

---

## 7. New Appendix — mechanism-ID audit

Create a new appendix section.  Put the full phase 2 / 2c / 4 / 5
tables in it, along with the per-λ weight-norm plot.  The plot files
are already generated:

- `analysis/mechanism_id/plots/fig_ladder_standard.png`
- `analysis/mechanism_id/plots/fig_heldout_rule.png`
- `analysis/mechanism_id/plots/fig_parity_decomp.png`

Suggested structure:

```latex
\section{Mechanism-identification audit}
\label{appendix:mechanism_id}

\subsection{Baseline ladder} \label{appendix:ladder}
[Figure \ref{fig_ladder_standard}: full standard-eval AUC per family
per model per task.]

\subsection{Lambda-agnostic ladder} \label{appendix:lam_agnostic}
[Paragraph + Table of AUCs with and without λ.  Per-λ weight-norm
plot for L1-logreg showing the $\lambda=7$ spike on Tricky Det and
the lack of spike on Tricky Random.]

\subsection{Held-out-rule generalization} \label{appendix:heldout}
[Figure \ref{fig_heldout_rule}: mean AUC on-rule and off-rule across
6 random $(\mathcal{S}, \kappa, \lambda)$ samples.]

\subsection{Parity decomposition} \label{appendix:parity_decomp}
[Figure \ref{fig_parity_decomp}: AUC across 6 input regimes,
LogReg and MLP-64.  Discussion of hidden-K vs.\ known-K.]

\subsection{Oracle audit of Tricky Det ceiling}
\label{appendix:oracle_audit}
[Table \ref{tab:oracle_audit}: oracle AUC, oracle F1, implied
$\pi$ for the top rule in each (tag, key-set) combination.
Documents the $\pi \approx 0.12$--$0.17$ residual on Tricky Det
and provides the rule that best matches stored labels.]
```

All tables can be generated from the CSVs in
`analysis/mechanism_id/results/`. If you want I can script LaTeX
table emission for each (2–3 more hours of my time).

---

## 8. Follow-on framing for paper 2 (ML4H)

In the limitations paragraph above I already foreshadow it. In the
conclusion, one extra sentence helps:

```latex
This mechanism-identification toolkit --- baseline ladder,
held-out-rule generalisation, and subset-reveal decomposition ---
is portable. In follow-on work on electronic-health-record
trajectories for early dementia detection, we use the same ladder
to audit which local content, co-occurrence, and ordered-pattern
features drive clinical predictions before attributing them to
"temporal reasoning".
```

This does three things at once:

1. Honours the mechanism-ID audit as a contribution in its own right.
2. Sets up paper 2 as a natural continuation.
3. Tells the reviewer you *already know* a synthetic benchmark needs a
   paired real-data follow-up, which is exactly the reviewer critique
   this paper would otherwise get.

---

## 9. What I did NOT change

- Theorems 3.3 and 3.5 and their proofs.
- §2 Related Work.
- §4.1 Naive and Tricky descriptions (only the key-set detail in 4d).
- §4.2 model list (Llama/Qwen/BERT/LSTM/Transformer/RNN-Transformer,
  XGBoost).
- §5 Table 2 summary (the $\cmark$/$\sim$/$\xmark$ table); it reads
  exactly the same under the new framing.
- Figure 1 (`comparison_AUC_all.png`).

The paper stays ~8 pages. For workshop 6-page limits, cut §2 Related
Work by half and merge §3.3 into a one-paragraph corollary.

---

## 10. Order of operations (3 days of writing)

**Day 1 (morning, 3 h):** apply §1 Abstract + §1 Introduction + §1
Contributions edits. Verify it reads cleanly end-to-end before touching
anything else.

**Day 1 (afternoon, 2 h):** apply §3.2 empirical-collapse paragraph,
§4.2 baseline paragraph, §4.3 ceiling fix, §4.1 key-set fix, Table 1 fix.

**Day 2 (3 h):** rewrite §5 per-task paragraphs (Tricky det / Tricky
rnd / Parity / Mechanism-ID subsection).  Rewrite limitations.

**Day 2 (afternoon, 2 h):** build the new Appendix sections.  CSVs
are all ready; only LaTeX table emission needed.  I can script this
on request.

**Day 3 (full day):** pass the paper end-to-end for coherence, drop
any sentence that still makes the old framing.  Fix Table 2 row labels
if they still say "Tricky" → consider "Compliance" everywhere.

---

## 11. One-sentence paper summary for cover letter / abstract page

> This paper introduces a controlled compliance / hidden-subset parity
> benchmark, proves local-information bounds on each task, reports a
> reproducible encoder-vs.-decoder efficiency gap, and --- through a
> mechanism-identification audit --- localizes the empirical signal in
> each task to the structure a minimal tabular baseline can expose.
