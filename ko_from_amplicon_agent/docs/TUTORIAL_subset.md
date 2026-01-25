# intelligrate.subset Tutorial

## What this does (short)
`intelligrate.subset` helps select a representative subset of samples for shotgun sequencing by balancing:
- **diversity** in feature space
- **geographic spread** (optional)
- **metadata coverage** (categories you care about)

It does this using:
- Distance matrices from feature tables
- k selection diagnostics
- k‑medoids clustering
- A genetic algorithm (GA) that balances diversity and metadata representation

This tutorial is beginner‑friendly and runnable end‑to‑end using the provided example data.

## Why subset?
Shotgun sequencing is expensive. Subsetting lets you select a smaller number of samples that still capture
the diversity and metadata structure of the full dataset, improving downstream generalization while keeping
costs manageable.

## Install the package
Recommended: use a clean Python >= 3.10 environment (venv or conda).

```
pip install intelligrate
```

For development from the repo:
```
pip install -e .
```

## Example data
We ship small example files:
- `data/feature_table_rel.tsv`
- `data/metadata.tsv`

## Python API (recommended, with explanations)
The API is the primary interface; the CLI simply wraps these functions.

### Step 1 — Compute a distance matrix
We compute a sample–sample distance matrix so all later steps (k diagnostics, k‑medoids, GA) use the same
notion of “distance” between samples.

```
from intelligrate.subset import compute_distance_matrix
import pandas as pd

ft = pd.read_csv('data/feature_table_rel.tsv', sep='\t', index_col=0)
D = compute_distance_matrix(ft, metric='bray', assume_relative=True)
```

### Step 2 — Suggest k (diagnostics)
We evaluate k values so you can **inspect diagnostics** and pick k. We do **not** auto‑pick k.

```
from intelligrate.subset import suggest_k

metrics = suggest_k(D, ft, k_range=range(2, 5), gap_B=2, random_state=42)
```

### Step 3 — Fit k‑medoids
We cluster samples into k medoids so the GA can preserve cluster‑level representation while optimizing
metadata balance and diversity.

```
from intelligrate.subset import fit_kmedoids

km = fit_kmedoids(D, k=3, random_state=42)
```

### Step 4 — GA subset selection
We run a genetic algorithm to select a subset that balances diversity (distance + geography) and metadata
representation, while honoring any fixed‑include constraints.

```
from intelligrate.subset import ga_subset
import pandas as pd

md = pd.read_csv('data/metadata.tsv', sep='\t', index_col=0)
selected, best_scores, fitness = ga_subset(
    km['cluster_df'],
    md,
    total_samples=30,
    balance_vars=['r_samp_country', 'r_samp_source'],
    coord_vars=('latitude', 'longitude'),
    population_size=10,
    generations=3,
    random_state=42,
    fixed_include=[
        'vubh091','vubh077','vubh075','vubh076','ubzh106','ubzh035','ubzh037',
        'ubzh051','ibbh058','ibbh028','4c44e','2cd6e','5eadf','ubzh054b',
        'ubzh006','ubzh033','vubh081'
    ],
)
```

## CLI examples (same steps)
If you prefer the CLI, use these example configs.

### Step 1 — Compute a distance matrix
Config (`configs/subset_distance.yaml`):
```
mode: distance
feature_table: data/feature_table_rel.tsv
metric: bray
assume_relative: true
output_dir: results/subset
```
Run:
```
python -m intelligrate.subset.cli --config configs/subset_distance.yaml
```
Outputs:
- `results/subset/distance.tsv`
- `results/subset/distance_meta.json`

### Step 2 — Suggest k
Config (`configs/subset_k.yaml`):
```
mode: suggest_k
feature_table: data/feature_table_rel.tsv
distance_matrix: results/subset/distance.tsv
k_min: 2
k_max: 4
gap_B: 2
seed: 42
plot: true
output_dir: results/subset
```
Run:
```
python -m intelligrate.subset.cli --config configs/subset_k.yaml
```
Outputs:
- `results/subset/k_diagnostics.tsv`
- `results/subset/k_diagnostics.png`
- `results/subset/k_suggest_meta.json`

### Step 3 — Fit k‑medoids
Config (`configs/subset_kmedoids.yaml`):
```
mode: kmedoids
distance_matrix: results/subset/distance.tsv
k: 3
seed: 42
output_dir: results/subset
```
Run:
```
python -m intelligrate.subset.cli --config configs/subset_kmedoids.yaml
```
Outputs:
- `results/subset/kmedoids_clusters.tsv`
- `results/subset/kmedoids_cluster_counts.tsv`

### Step 4 — GA subset selection
We include example fixed IDs (from the notebook) in `configs/fixed_include.tsv`.
If an ID is not present in the example dataset it is ignored automatically.

Config (`configs/subset_ga.yaml`):
```
mode: ga
cluster_table: results/subset/kmedoids_clusters.tsv
metadata_table: data/metadata.tsv
output_dir: results/subset

total_samples: 30

balance_vars:
  - r_samp_country
  - r_samp_source

metadata_weights:
  r_samp_country: 1.0
  r_samp_source: 0.5

coord_vars: [latitude, longitude]

population_size: 10
generations: 3
seed: 42

min_category_n: 2
min_per_category: 2

grid_weight: 3.0
distance_weight: 2.0
balance_weight: 1.0
balance_scale: 1000.0
hard_penalty_weight: 100.0

fixed_include: configs/fixed_include.tsv
```
Run:
```
python -m intelligrate.subset.cli --config configs/subset_ga.yaml
```
Outputs:
- `results/subset/ga_selected_samples.tsv`
- `results/subset/ga_best_scores.tsv`
- `results/subset/ga_fitness_array.tsv`
- `results/subset/ga_meta.json`

## Parameter reference (subset configs)
Below is a concise reference for parameters used in the example configs.

### distance
- `feature_table`: TSV with samples x features
- `metric`: `bray` | `jaccard` | `aitchison`
- `assume_relative`: set true if rows already sum to 1
- `pseudocount`: used for CLR in `aitchison`
- `output_dir`: where outputs are written

### suggest_k
- `distance_matrix`: path to precomputed distance matrix
- `k_min`, `k_max`: range of k values to test
- `gap_B`: number of reference datasets for gap statistic
- `seed`: random seed
- `plot`: whether to save a diagnostic plot

### kmedoids
- `k`: chosen number of clusters
- `seed`: random seed

### ga
- `cluster_table`: k‑medoids cluster assignments
- `metadata_table`: sample metadata
- `total_samples`: number of samples to select (rounded down to divisible by k)
- `balance_vars`: metadata fields to balance
- `metadata_weights`: optional weights per metadata field
- `coord_vars`: [latitude, longitude] field names
- `population_size`, `generations`: GA runtime settings
- `min_category_n`: categories must have at least this many samples globally
- `min_per_category`: each included category must have at least this many samples in the subset
- `grid_weight`, `distance_weight`, `balance_weight`: objective weights
- `balance_scale`: scales metadata balance term
- `hard_penalty_weight`: penalty for underrepresented categories
- `fixed_include`: TSV with one sample ID per line to always include
- `fixed_exclude`: TSV with one sample ID per line to always exclude

## Notes
- The module does **not** align tables; it only reports overlap counts.
- GA runtime depends on population size and generations; start small and scale up.
