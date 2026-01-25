from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np
import pandas as pd
from deap import base, creator, tools
from scipy.spatial.distance import pdist


def compute_grid_cell(lat: float, lon: float, grid_size: float = 1.0) -> tuple[int, int]:
    return (round(lat / grid_size), round(lon / grid_size))


def score_global_sample_set_with_distance(
    subset_df: pd.DataFrame,
    coord_vars: tuple[str, str],
    category_targets: dict,
    *,
    min_per_category: int = 5,
    grid_size: float = 1.0,
    grid_weight: float = 3.0,
    distance_weight: float = 2.0,
    balance_weight: float = 1.0,
    balance_scale: float = 1000.0,
    hard_penalty_weight: float = 100.0,
    metadata_weights: dict | None = None,
) -> float:
    if metadata_weights is None:
        metadata_weights = {}

    grid_cells = {
        compute_grid_cell(lat, lon, grid_size)
        for lat, lon in subset_df[list(coord_vars)].dropna().values
    }
    grid_score = len(grid_cells)

    coords = subset_df[list(coord_vars)].dropna().values
    mean_distance = float(np.mean(pdist(coords))) if len(coords) > 1 else 0.0

    hard_penalty = 0.0
    for var, target_counts in category_targets.items():
        counts = subset_df[var].value_counts().to_dict()
        for cat in target_counts:
            actual = counts.get(cat, 0)
            if actual < min_per_category:
                shortfall = min_per_category - actual
                hard_penalty += shortfall * float(hard_penalty_weight)

    penalty = 0.0
    max_penalty = 0.0
    for var, target_counts in category_targets.items():
        actual_counts = subset_df[var].value_counts().to_dict()
        max_target = max(target_counts.values()) if target_counts else 1
        var_weight = float(metadata_weights.get(var, 1.0))
        for cat, target in target_counts.items():
            actual = actual_counts.get(cat, 0)
            deviation = abs(actual - target)
            rarity_weight = math.log1p((max_target / (target + 1e-6)) ** 2)
            penalty += var_weight * rarity_weight * deviation
            max_penalty += var_weight * math.log1p(max_target / (target + 1e-6)) * target

    balance_score = 1.0 - penalty / max_penalty if max_penalty > 0 else 0.0

    final_score = (
        float(grid_weight) * grid_score
        + float(distance_weight) * mean_distance
        + float(balance_weight) * balance_score * float(balance_scale)
    )
    return float(final_score - hard_penalty)


def smart_mutate(
    individual,
    *,
    cluster_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    cluster_groups: dict,
    samples_per_cluster: int,
    coord_vars: tuple[str, str],
    category_targets: dict,
    min_per_category: int = 5,
    grid_size: float = 1.0,
    locked_samples: list[str] | None = None,
):
    if locked_samples is None:
        locked_samples = []

    individual = [sid for sid in individual if sid is not None and sid in metadata_df.index and sid in cluster_df.index]
    df = metadata_df.loc[individual]

    cluster_sample_map = {cid: [] for cid in cluster_groups}
    for sid in individual:
        cid = cluster_df.loc[sid, "Cluster"]
        cluster_sample_map[cid].append(sid)

    new_individual = []
    for cid, original in cluster_sample_map.items():
        penalty_contributions = {}
        for sid in original:
            penalty = 0.0
            sample_row = df.loc[sid]
            for var, target_counts in category_targets.items():
                total_counts = metadata_df.loc[individual, var].value_counts()
                value = sample_row[var]
                if isinstance(value, pd.Series):
                    # If duplicate sample IDs exist, pick the first value (matches notebook behavior).
                    value = value.iloc[0] if not value.empty else np.nan
                elif isinstance(value, pd.DataFrame):
                    # If duplicate columns exist, pick the first value.
                    value = value.iloc[0, 0] if not value.empty else np.nan
                elif isinstance(value, (np.ndarray, list, tuple)):
                    # Fallback for unexpected container values.
                    value = value[0] if len(value) else np.nan
                if pd.isna(value):
                    continue
                if value in target_counts:
                    penalty += abs(total_counts.get(value, 0) - target_counts[value])
            penalty_contributions[sid] = penalty

        worst_samples = [
            s
            for s in sorted(penalty_contributions, key=penalty_contributions.get, reverse=True)
            if s not in locked_samples
        ]
        if len(original) < 25:
            to_replace = worst_samples[:1]
        else:
            to_replace = worst_samples[: random.randint(1, 4)]

        keep = list(set(original) - set(to_replace))
        pool = list(set(cluster_groups[cid]) - set(original))
        if len(pool) < len(to_replace):
            pool = cluster_groups[cid]
        random.shuffle(pool)
        replacements = pool[: len(to_replace)]
        new_cluster_samples = keep + replacements
        new_individual.extend(new_cluster_samples)

    new_individual = [sid for sid in new_individual if sid in metadata_df.index and sid in cluster_df.index]
    unique_individual = list(dict.fromkeys(new_individual))

    expected = samples_per_cluster * len(cluster_groups)
    if len(unique_individual) < expected:
        missing = expected - len(unique_individual)
        already_used = set(unique_individual)
        refill_pool = []
        for cid, sids in cluster_groups.items():
            refill_pool.extend([s for s in sids if s not in already_used and s not in locked_samples])
        random.shuffle(refill_pool)
        if len(refill_pool) >= missing:
            unique_individual.extend(refill_pool[:missing])
            unique_individual = list(dict.fromkeys(unique_individual))
        else:
            return creator.Individual(individual),

    if len(unique_individual) != expected:
        return creator.Individual(individual),

    return creator.Individual(unique_individual),


def ga_subset(
    cluster_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    *,
    total_samples: int,
    balance_vars: list[str] | None = None,
    coord_vars: tuple[str, str] = ("latitude", "longitude"),
    min_category_n: int = 5,
    min_per_category: int = 5,
    grid_size: float = 1.0,
    population_size: int = 50,
    generations: int = 50,
    random_state: int = 42,
    fixed_include: list[str] | None = None,
    fixed_exclude: list[str] | None = None,
    metadata_weights: dict | None = None,
    grid_weight: float = 3.0,
    distance_weight: float = 2.0,
    balance_weight: float = 1.0,
    balance_scale: float = 1000.0,
    hard_penalty_weight: float = 100.0,
):
    requested_total = int(total_samples)
    if balance_vars is None:
        balance_vars = []
    if fixed_include is None:
        fixed_include = []
    if fixed_exclude is None:
        fixed_exclude = []

    # Ensure unique sample IDs to avoid ambiguous Series/DataFrame lookups.
    if metadata_df.index.has_duplicates:
        metadata_df = metadata_df.loc[~metadata_df.index.duplicated(keep="first")]
    if cluster_df.index.has_duplicates:
        cluster_df = cluster_df.loc[~cluster_df.index.duplicated(keep="first")]

    random.seed(int(random_state))
    np.random.seed(int(random_state))

    valid_samples = cluster_df.index.intersection(metadata_df.index)
    cluster_df = cluster_df.loc[valid_samples]
    metadata_df = metadata_df.loc[valid_samples]

    fixed_include = [s for s in fixed_include if s in metadata_df.index and s in cluster_df.index]
    fixed_exclude = [s for s in fixed_exclude if s in metadata_df.index and s in cluster_df.index]

    if len(fixed_include) > total_samples:
        raise ValueError("Number of fixed_include samples exceeds total_samples.")

    if balance_vars:
        allowed_categories = {
            var: metadata_df[var].value_counts()[lambda x: x >= min_category_n].index.tolist()
            for var in balance_vars
        }
        for var, allowed in allowed_categories.items():
            metadata_df = metadata_df[metadata_df[var].isin(allowed)]
    else:
        allowed_categories = {}

    valid_samples = cluster_df.index.intersection(metadata_df.index)
    cluster_df = cluster_df.loc[valid_samples]
    metadata_df = metadata_df.loc[valid_samples]

    fixed_include = [s for s in fixed_include if s in metadata_df.index and s in cluster_df.index]
    fixed_exclude = [s for s in fixed_exclude if s in metadata_df.index and s in cluster_df.index]

    category_targets = {
        var: {cat: total_samples // len(allowed) for cat in allowed}
        for var, allowed in allowed_categories.items()
        if allowed
    }

    fixed_df = metadata_df.loc[fixed_include] if fixed_include else metadata_df.iloc[0:0]
    for var, targets in category_targets.items():
        fixed_counts = fixed_df[var].value_counts().to_dict()
        for cat in targets:
            targets[cat] -= fixed_counts.get(cat, 0)
            if targets[cat] < 0:
                targets[cat] = 0

    k = int(cluster_df["Cluster"].nunique())
    if k <= 0:
        raise ValueError("No clusters provided in cluster_df.")

    if total_samples % k != 0:
        # keep same behavior as notebook: floor via integer division
        total_samples = int((total_samples // k) * k)

    samples_per_cluster = total_samples // k

    cluster_groups = {
        cid: [
            sid
            for sid in cluster_df[cluster_df["Cluster"] == cid].index
            if sid in metadata_df.index and sid not in fixed_exclude
        ]
        for cid in cluster_df["Cluster"].unique()
    }

    if "FitnessMax" not in creator.__dict__:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    def generate_individual():
        individual = fixed_include.copy()
        fixed_cluster_counts = cluster_df.loc[fixed_include, "Cluster"].value_counts().to_dict()

        for cluster_id, samples in cluster_groups.items():
            n_fixed = fixed_cluster_counts.get(cluster_id, 0)
            n_to_add = samples_per_cluster - n_fixed
            if n_to_add < 0:
                raise ValueError(f"Cluster {cluster_id} is overrepresented by fixed samples.")

            pool = list(set(samples) - set(fixed_include) - set(individual))
            if len(pool) < n_to_add:
                raise ValueError(f"Not enough samples in cluster {cluster_id} after exclusions.")
            individual.extend(random.sample(pool, n_to_add))

        return creator.Individual(individual)

    def evaluate(individual):
        cleaned = [i for i in individual if i is not None and i in metadata_df.index]
        df = metadata_df.loc[cleaned]
        return (
            score_global_sample_set_with_distance(
                df,
                coord_vars,
                category_targets,
                min_per_category=min_per_category,
                grid_size=grid_size,
                grid_weight=grid_weight,
                distance_weight=distance_weight,
                balance_weight=balance_weight,
                balance_scale=balance_scale,
                hard_penalty_weight=hard_penalty_weight,
                metadata_weights=metadata_weights,
            ),
        )

    def mutate(individual):
        return smart_mutate(
            individual,
            cluster_df=cluster_df,
            metadata_df=metadata_df,
            cluster_groups=cluster_groups,
            samples_per_cluster=samples_per_cluster,
            coord_vars=coord_vars,
            category_targets=category_targets,
            min_per_category=min_per_category,
            grid_size=grid_size,
            locked_samples=fixed_include,
        )

    toolbox.register("individual", generate_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=int(population_size))

    best_ind = None
    best_score = -np.inf
    best_fitness_over_gens = []
    all_fitnesses_over_gens = []

    for _gen in range(int(generations)):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.7:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring

        gen_fitnesses = [ind.fitness.values[0] for ind in population]
        all_fitnesses_over_gens.append(gen_fitnesses)
        best_gen = max(population, key=lambda ind: ind.fitness.values[0])
        best_fitness_over_gens.append(best_gen.fitness.values[0])

        if best_gen.fitness.values[0] > best_score:
            best_score = best_gen.fitness.values[0]
            best_ind = best_gen

    valid_best_ind = [
        i for i in best_ind if i is not None and i in metadata_df.index and i in cluster_df.index
    ]
    valid_best_ind = list(dict.fromkeys(valid_best_ind))

    expected = samples_per_cluster * len(cluster_groups)
    if len(valid_best_ind) < expected:
        cluster_counts = Counter(cluster_df.loc[valid_best_ind, "Cluster"])
        already_used = set(valid_best_ind)
        for cid, sids in cluster_groups.items():
            current = cluster_counts.get(cid, 0)
            missing = samples_per_cluster - current
            if missing <= 0:
                continue
            candidates = [s for s in sids if s not in already_used and s not in fixed_include]
            if len(candidates) < missing:
                raise ValueError(f"Not enough candidates to refill cluster {cid}")
            base_df = metadata_df.loc[valid_best_ind]
            scored_candidates = []
            for s in candidates:
                test_df = pd.concat([base_df, metadata_df.loc[[s]]])
                score = score_global_sample_set_with_distance(
                    test_df,
                    coord_vars,
                    category_targets,
                    min_per_category=min_per_category,
                    grid_size=grid_size,
                    grid_weight=grid_weight,
                    distance_weight=distance_weight,
                    balance_weight=balance_weight,
                    balance_scale=balance_scale,
                    hard_penalty_weight=hard_penalty_weight,
                    metadata_weights=metadata_weights,
                )
                scored_candidates.append((s, score))
            best_additions = [s for s, _ in sorted(scored_candidates, key=lambda x: x[1], reverse=True)[:missing]]
            valid_best_ind.extend(best_additions)
            already_used.update(best_additions)
        valid_best_ind = list(dict.fromkeys(valid_best_ind))

    # If the requested total was not divisible by k, top up with best candidates.
    if len(valid_best_ind) < requested_total:
        missing = requested_total - len(valid_best_ind)
        already_used = set(valid_best_ind)
        candidates = [
            s
            for s in metadata_df.index
            if s not in already_used and s in cluster_df.index and s not in fixed_exclude
        ]
        if len(candidates) < missing:
            raise ValueError("Not enough candidates to reach requested total_samples.")
        base_df = metadata_df.loc[valid_best_ind]
        scored_candidates = []
        for s in candidates:
            test_df = pd.concat([base_df, metadata_df.loc[[s]]])
            score = score_global_sample_set_with_distance(
                test_df,
                coord_vars,
                category_targets,
                min_per_category=min_per_category,
                grid_size=grid_size,
                grid_weight=grid_weight,
                distance_weight=distance_weight,
                balance_weight=balance_weight,
                balance_scale=balance_scale,
                hard_penalty_weight=hard_penalty_weight,
                metadata_weights=metadata_weights,
            )
            scored_candidates.append((s, score))
        best_additions = [s for s, _ in sorted(scored_candidates, key=lambda x: x[1], reverse=True)[:missing]]
        valid_best_ind.extend(best_additions)
        valid_best_ind = list(dict.fromkeys(valid_best_ind))

    result_df = metadata_df.loc[valid_best_ind].copy()
    result_df["Cluster"] = cluster_df.loc[valid_best_ind, "Cluster"].values

    return result_df, best_fitness_over_gens, all_fitnesses_over_gens
