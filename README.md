# CABA Experiment Guide
This document introduces the core concept of CABA (Cross‑modal Alignment Backdoor Attack), together with the complete execution workflow of the code repository from dataset preparation to experimental evaluation.

- **Paper**: *CABA: A Stealthy Backdoor Attack Based on Cross‑Modal Alignment Against Cross‑Modal Retrieval Models*
- **Entry script**: `run.sh` — unified wrapper for five execution modes, with logging and optional automatic shutdown

---

## 1. What is CABA
CABA is a **data‑poisoning backdoor attack** targeting **cross‑modal retrieval models**. The adversary only poisons training data and has no access to the victim’s model structure or training procedure.

### 1.1 Core Insight
Cross‑modal retrieval models (image‑to‑text / text‑to‑image) map images and texts into a shared semantic space $\mathcal{Z}$, and perform retrieval via similarity ranking. Different image regions contribute very unequally to cross‑modal matching — only a small subset of regions dominate image‑text similarity.

We name these semantically meaningful regions that the model genuinely attends to **Cross‑Modal Alignment Regions**.
The core idea:
> **Embed the backdoor trigger inside cross‑modal alignment regions instead of applying perturbations across the whole image.**

This brings two key benefits:
- **Effectiveness**: Perturbations lie along pre‑established semantic pathways, making it easier to learn the trigger‑to‑target‑label mapping and yielding higher ASR.
- **Stealthiness**: Only local pixels are modified, preserving overall texture and yielding higher PSNR / SSIM (pixel‑level stealth). Grad‑CAM heatmaps resemble those of clean models without isolated high‑activation peaks (attention‑level stealth). This allows evasion of activation‑based defenses such as Fine‑Pruning and Februus.

### 1.2 Attack Setting (Threat Model)
- The adversary acts as a malicious data provider or public dataset poisoner.
- The adversary **can only access and modify training data**, and cannot interfere with model training, architecture, or hyper‑parameters.
- Auxiliary models: open‑source **CLIP** (image‑text similarity) + **Grounding DINO** (open‑vocabulary text‑conditioned detection).
  **Key**: These two models are independent of the victim encoders. Poisoned regions are reused across different architectures rather than relearned, which explains cross‑architecture generalisation.
- This work focuses on **image‑side trigger** under the **image‑to‑text (img2txt)** retrieval scenario.

### 1.3 Three Core Components
| Stage | Name | Function | Corresponding Formula |
|---|---|---|---|
| ① | Cross‑modal alignment region mining | Text keyword mining + image region localisation | $S_j$, $S_k$ |
| ② | Target image selection | Score samples by retrieval ranking gain to select semantic anchors | $\mathrm{Score}=S_{pos}+\lambda S_{neg}-\mu P$ |
| ③ | Poisoned image generation | Region‑constrained PGD, perturbations applied only within alignment regions | $\mathcal{L}_{emb}-\lambda_{neg}\mathcal{L}_{neg}+\lambda_{tv}\mathcal{L}_{tv}$ |

---

## 2. Mathematical Formulation of Three Core Components
### 2.1 Stage ①: Cross‑Modal Alignment Region Mining
**Why not use vanilla object detectors?**
Take the caption *"a dog running on grass"* as an example. Generic detectors often return large background boxes that contribute little to image‑text matching and cannot reliably carry trigger signals. We therefore use pre‑-trained vision‑language models to evaluate contributions from both text and image perspectives.

#### (a) Text‑side: Cross‑modal alignment keyword mining
For each candidate word $w_j$ in the text, apply two masking strategies (**direct deletion** and **[MASK] replacement**). Two contribution metrics are computed:

- **Cross‑modal semantic contribution** (degree of degradation in image‑text matching):
$$\Delta_j^{cm}=\left|\cos(f^v,f^t)-\cos\left(f^v,f^{t,\setminus j}\right)\right|$$

- **Intra‑text semantic contribution** (semantic stability of the sentence):
$$\Delta_j^{text}=1-\cos\left(f^t,f^{t,\setminus j}\right)$$

Min‑max normalise both metrics and weight them (default weight $\omega=0.5$). Take the maximum value from the two masking schemes as the final score $S_j=\max(I_j,I'_j)$. Select top‑$K^t$ words to construct the keyword set $\mathcal{W}_{key}$, where $K^t$ cannot exceed 30% of the total text length.

#### (b) Image‑side: Cross‑modal alignment region localisation
Concatenate keywords with commas as prompt input to **Grounding DINO** for open‑vocabulary detection. Apply three filtering rules on candidate boxes:
- Discard boxes with confidence below threshold
- Remove boxes with excessively small area (filter local noise)
- Reject boxes with excessively large area (avoid full‑object coverage and preserve stealthiness)

Fill each candidate box $r_k$ with pixel value 1 to produce mask $\hat{s}_k^v$. Compute:

- **Cross‑modal semantic contribution**: $\Delta_k^{cm}=\left|\cos(f^v,f^t)-\cos(f_k^v,f^t)\right|$
- **Intra‑image semantic contribution**: $\Delta_k^{img}=1-\cos(f^v,f_k^v)$

Normalise and weight to obtain $S_k$.
**Critical step**: Under the constraint that total masked area ≤ 30% of the full image, use **dynamic programming** to solve for the region combination that maximises visual alignment score. This yields the final cross‑modal alignment region $\mathcal{M}(s^v)$. $K^v$ is adaptive and varies per input image.

> Both dimensions are required. Using only cross‑modal change may misclassify regions with high accidental semantic contribution but low visual capacity as core alignment regions.

### 2.2 Stage ②: Target Image Selection
**Challenge**: Multi‑label datasets contain many images satisfying target labels. Their positions in retrieval embedding space and associated extra semantics differ. Random selection causes unstable optimisation and perturbation leakage towards unintended semantics.

We design a scoring function based on retrieval ranking structure with three terms:
$$\mathrm{Score}(s_i^v)=S_{pos}(s_i^v)+\lambda\,S_{neg}(s_i^v)-\mu\,P(s_i^v)$$

- **$S_{pos}$: Positive‑sample ranking gain** — Collect positive text samples for each target label. Apply label‑frequency balancing weight $\pi_l=\frac{1/\sqrt{f_l}}{\sum_{l'\in\mathcal{T}}1/\sqrt{f_{l'}}}$ together with duplication correction factor $\rho_j$ (avoid duplicate counting for texts containing multiple target labels). Use negative average rank; higher rank in retrieval gives larger $S_{pos}$.
- **$S_{neg}$: Hard‑negative suppression** — For each target label, collect top‑$M$ high‑co‑occurrence non‑target labels. Gather texts containing these high‑co‑occurrence labels but without the target label. $S_{neg}$ uses raw average rank (no negative sign). Larger $S_{neg}$ indicates stronger suppression of confusing samples.
- **$P$: Redundant‑label penalty** — $P=\max(\sum_c r_{i,c}-|\mathcal{T}|,0)$. Penalise excessive auxiliary labels carried by candidate images to guarantee semantic purity of target samples.

Select $\arg\max$ as target image $s_t^v$. Default hyper‑parameters: $\lambda=0.5$, $\mu=0.1$.

### 2.3 Stage ③: Region‑Constrained PGD for Poisoned Image Generation
Given alignment mask $\mathcal{M}(s^v)$ and target image $s_t^v$, optimise adversarial perturbations with three loss terms:

- **$L_{emb}$**: Embedding alignment loss $1-\cos(\mathcal{F}^v(\hat{s}^v),\mathcal{F}^v(s_t^v))$, pulling poisoned embedding towards target‑image embedding.
- **$L_{neg}$**: Negative‑contrast loss $1-\cos(\mathcal{F}^v(\hat{s}^v),\mathcal{F}^v(s^v))$, pushing poisoned embedding away from clean‑source embedding.
- **$L_{tv}$**: Total‑variation regularisation accumulating horizontal / vertical gradients **only inside alignment mask**, suppressing high‑frequency noise artefacts.

$$\mathcal{L}_{total}=\mathcal{L}_{emb}-\lambda_{neg}\mathcal{L}_{neg}+\lambda_{tv}\mathcal{L}_{tv}$$

**Precise definition of projected gradient descent**: Outer loop: gradient update → projection onto feasible set $\mathcal{S}$ → hard reset pixels outside mask to clean original values. Inner gradient update uses **AdamW** ($\beta_1{=}0.9,\beta_2{=}0.999$, weight‑decay 0.01) with **cosine‑annealing learning rate** ($T_{max}$ equals total iteration count, $\eta_{min}=\mathrm{lr}\cdot0.01$). This is different from classic sign‑gradient fixed‑step PGD.

Feasible‑set constraints for projection $\mathcal{S}$: $\ell_\infty\le\varepsilon$ inside mask; pixels outside mask strictly equal clean source values, all pixels clipped into [0,1]. Gradients outside mask are forced to zero, preventing trigger leakage beyond alignment regions.

---

## 3. Code Structure & File Responsibilities
| File | Role | Corresponding Stage |
|---|---|---|
| `caba/text_keyword_extractor.py` | Text keyword extractor (CLIP + dual‑mask strategy) | ①‑a |
| **Official Grounding DINO service** (see link below) | Generate candidate detection boxes for every image. Output is then consumed by `caba/image_region_extractor.py` as a `data/{ds}/bbox/*.json` file | ①‑b prerequisite |
| `caba/image_region_extractor.py` | Region scoring, NMS, region selection; output binary mask PNGs | ①‑b |
| `caba/extract_target_image.py` | Target‑image selection (rank‑gain + hard‑negative scoring) | ② |
| `caba/poison_image_regions.py` | Poisoned‑image generation (region‑constrained PGD) | ③ |
| `caba/evaluate_target.py` | Auxiliary evaluation & filtering for candidate target images | Auxiliary |
| `main.py` + `victims/dscmr.py` | Victim model training and evaluation (CDA / ASR) | Evaluation |
| `backdoors/caba.py` | Poisoned dataset wrapper, replace clean samples with poisoned images in training set | Bridge module |
| `run.sh` | Unified experiment scheduler | Scheduling |
| `config/dscmr.yaml` | Training hyper‑parameter configuration | Configuration |

---

## 4. Complete Experiment Workflow
> **Path convention**: Modify line 10 inside `run.sh`: `PROJECT_DIR="/root/CABA-main"` to match your actual project directory.

### Step 0: Environment Setup and Dataset Preparation
```bash
pip install -r requirements.txt
```

Dataset directory layout (replicate for each of the three datasets; images, texts and labels must be prepared in advance):
```
data/
├── NUS‑WIDE/
│   ├── images/
│   ├── cm_train_imgs.txt / cm_train_labels.txt / cm_train_txts.txt
│   ├── cm_test_imgs.txt  / cm_test_labels.txt  / cm_test_txts.txt
│   └── cm_database_imgs.txt / cm_database_labels.txt / cm_database_txts.txt
├── MS‑COCO/
└── IAPR‑TC/
```

**MS‑COCO download**:
> [https://pjreddie.com/projects/coco](https://pjreddie.com/projects/coco‑mirror/)

**NUS‑WIDE download**:
> [https://www.kaggle.com/datasets/twerwweqweq/nuswide](https://www.kaggle.com/datasets/twerwweqweq/nuswide)

**IAPR‑TC download**:
> http://www‑i6.informatik.rwth‑aachen.de/imageclef/resources/iaprtc12.tgz

### Step 1: Text Keyword Extraction (Stage ①‑a)
```bash
bash run.sh txt
# underlying command: python -m caba.text_keyword_extractor
```

Default arguments: `--dataset NUS‑WIDE`, `--split train`, `--text_mask_method merge` (hybrid masking), `--cross_modal_weight 0.5`, `--words_thred 10`, `--max_text_len 60`, `--ratio 0.01`.

**Output artifacts**:
- `log/.../NUS‑WIDE_train_text_words.pkl` (main output)
- `data/NUS‑WIDE/keywords/NUS‑WIDE_train_text_words_0.0100.json` — keyword mapping, required for subsequent Step 2
- `data/NUS‑WIDE/badcm_train_mask_text.npy` (text‑mask matrix)

### Step 2: Generate Candidate Detection Boxes (Stage ①‑b prerequisite, Grounding DINO)
> This step is **not wrapped inside `run.sh` but mandatory in workflow**. `image_region_extractor.py` no longer instantiates Grounding DINO locally; it reads pre‑computed boxes from `data/{dataset}/bbox/{dataset}_{split}_bbox_r={ratio}.json`. The boxes are produced by calling the **officialGrounding DINO service**, which is the source of truth for candidate region detection in this pipeline.

For the request format, authentication, supported prompt syntax, batch limits and output schema, please consult the official API documentation:

> [https://cloud.deepdataspace.com/zh/docs](https://cloud.deepdataspace.com/zh/docs)

Please follow the documentation to:

1. Upload each image to the service.
2. Submit a Grounding DINO detection task whose prompt is the per-image keyword string produced in **Step 1** (`data/{dataset}/keywords/{dataset}_{split}_text_words_{ratio}.json`).
3. Collect the returned bounding boxes, scores and categories.
4. Aggregate per-image results into a single JSON file at `data/{dataset}/bbox/{dataset}_{split}_bbox_r={ratio}.json`, keyed by image filename with the same schema consumed by `caba/image_region_extractor.py` (each value is a list of objects containing `bbox`, `score`, `category`).

**Output**: `data/{dataset}/bbox/{dataset}_{split}_bbox_r={ratio}.json`

### Step 3: Image‑Side Alignment Region Extraction (Stage ①‑b)
```bash
bash run.sh img
# underlying command: python -m caba.image_region_extractor
```

Default arguments hard‑coded inside script: `--dataset MS‑COCO`, `--split test`, `--box_threshold 0.3`, `--ratio 1`.

> `run.sh` does not forward custom CLI parameters. Change default values in `parse_args()` inside source file, or invoke python directly with explicit arguments.
> **Dataset / split / ratio must exactly match previous steps**, otherwise box JSON cannot be loaded.

**Output artifacts**:
- `log/grounding_dino_regions/{dataset}_{split}_image_regions_r={ratio}.pkl` + corresponding JSON file
- Binary mask PNGs stored under `data/{dataset}/masks_image/{split}_{ratio}/` — directly consumed in poisoning generation stage.

### Step 4: Target‑Image Selection (Stage ②)
```bash
python -m caba.extract_target_image \
  --dataset NUS‑WIDE \
  --split train \
  --target_labels 9,19 \
  --lambda_neg 0.5 \
  --mu_purity 0.1
```

**Output**: Top‑K candidate target‑image paths printed to stdout. Use the highest‑score image path as target image $s_t^v$.

### Step 5: Generate Poisoned Images (Stage ③)
```bash
python -m caba.poison_image_regions \
  --dataset NUS‑WIDE \
  --split train \
  --ratio 0.05 \
  --target_path ./data/NUS‑WIDE/images/xxx.jpg \
  --eps 0.0157 \
  --neg_contrast_weight 0.02 \
  --iter_attack 2000 \
  --lr_attack 1.5 \
  --tv_weight 0.000004 \
  --mask_mode default
```

**Key‑parameter explanation**:
- `--mask_mode`: ablation switch corresponding to paper experiments:
  - `default`: use original cross‑modal alignment mask (main results in paper)
  - `all`: full‑image perturbation (global baseline)
  - `random`: same‑area random spatial mask
  - `fixed`: same‑area fixed‑position patch
  These correspond to rows global / random / fixed / CABA in region‑ablation table.
- `--eps 4/255`: $\varepsilon=4/255$ used for main paper results.
- `--lr_attack` is internally divided by 255 when passed to optimiser; value `1.5` yields effective learning rate ≈0.0059.

**Output**: Poisoned images saved under `data/{dataset}/poisoned_image/{split}_{ratio}_{poison_data_last_name}/`, preserving relative file names of clean sources to keep non‑poisoned samples unchanged. PSNR / SSIM metrics are printed in runtime logs.

### Step 6: Train Victim Model and Evaluate
```bash
python main.py --config_name dscmr.yaml \
  --dataset NUS‑WIDE \
  --attack BadCM \
  --percentage 0.05
```

Configuration file `config/dscmr.yaml`:
- `module: victims.dscmr` (DSCMR victim implementation)
- `backbones: ['VGG16', 'TextCNN']` (image / text encoders)
- `target: [0,2]` (target label indices, **must match `--target_labels` used in poisoning stage**)
- `percentage: 0.01` (poisoning ratio, can be overridden by command‑line argument)
- `test_percentage: 1` (100 % poisoned test set for ASR measurement)
- `poison_data_last_name: '...'` — **must match `--poison_data_last_name` given in Step 5**.

---

## 5. Evaluation Metrics
### Attack Effectiveness (Corpus‑Level)
- **CDA (Clean‑Data Accuracy)**: MAP computed on clean test queries, measuring normal retrieval performance retention after poisoning.
- **ASR (Attack‑Success Rate)**: t‑MAP under targeted attack scenario, measuring how effectively retrieval outputs shift toward attacker‑specified target labels.

Both metrics operate at **corpus level**, averaging ranking quality across all queries. This differs from query‑level success metrics widely adopted in classification backdoor literature. All baseline methods are re‑implemented under identical evaluation protocol for fair comparison.

**Relevance criterion**: All three datasets are multi‑label; we adopt **any‑label criterion**: a retrieved sample is treated as relevant if it shares at least one label with query (label vector inner‑product > 0). CDA uses original ground‑truth labels for queries. For ASR evaluation, attack query label vectors are replaced with one‑hot vectors activated only on target labels while retrieval database retains original annotations. Under multi‑target setting, hitting any target label counts as relevant, making ASR criterion relatively more permissive than single‑label scenarios.

**AP@5000**: All datasets compute average precision with ranking truncated at top‑5000 retrieved items.

$$AP@K(q_i)=\frac{1}{|R_{i,K}|}\sum_{k=1}^{|R_{i,K}|}\frac{k}{\mathrm{rank}(k)}$$

Where $R_{i,K}$ denotes the set of relevant retrieved items within top‑$K$. Queries with $|R_{i,K}|=0$ contribute zero AP.

### Attack Stealthiness
- **PSNR** ↑, **SSIM** ↑, **LPIPS** ↓ (AlexNet feature‑space metric). They characterise visual stealthiness from pixel, structural and deep perceptual perspectives respectively.

---

## 6. Quick‑Reference Command List
```bash
# Modify project root inside run.sh line 10
# PROJECT_DIR="/root/CABA-main"   →   set to your actual project folder

# 0. Environment installation
pip install -r requirements.txt

# 1. Text‑side keyword mining (Stage ①‑a)
bash run.sh txt

# 2. Generate object‑detection candidate boxes via the official Grounding DINO service
#    See Step 2 above and https://cloud.deepdataspace.com/zh/docs for the request format.
#    Save the aggregated result to data/{dataset}/bbox/{dataset}_{split}_bbox_r={ratio}.json

# 3. Image‑side alignment‑region extraction (Stage ①‑b)
bash run.sh img

# 4. Target‑image selection (Stage ②, no wrapper in run.sh)
python -m caba.extract_target_image --dataset NUS‑WIDE --split train --target_labels 9,19

# 5. Generate poisoned images (Stage ③)
python -m caba.poison_image_regions --dataset NUS‑WIDE --split train --ratio 0.05 \
  --target_path ./data/NUS‑WIDE/images/0049_515476197.jpg --eps 0.0157 --mask_mode default

# 6. Victim‑model training & evaluation
python main.py --config_name dscmr.yaml --dataset NUS‑WIDE --attack BadCM --percentage 0.05

# 7. Auxiliary target‑sample evaluation
python -m caba.evaluate_target --target_dir ./data/NUS‑WIDE/target/person_9_train_set
```