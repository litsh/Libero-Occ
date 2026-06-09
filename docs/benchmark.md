# Benchmark Assets

LIBERO-Occ releases final benchmark assets only. The benchmark-generation code is intentionally not included.

## Released Suites

| Suite | BDDL directory | Init directory |
| --- | --- | --- |
| Spatial | `benchmark_assets/bddl_files/libero_spatial_occluded` | `benchmark_assets/init_files/libero_spatial_occluded` |
| Goal | `benchmark_assets/bddl_files/libero_goal_occluded` | `benchmark_assets/init_files/libero_goal_occluded` |
| Object | `benchmark_assets/bddl_files/libero_object_occluded` | `benchmark_assets/init_files/libero_object_occluded` |
| LIBERO-10 | `benchmark_assets/bddl_files/libero_10_occluded` | `benchmark_assets/init_files/libero_10_occluded` |

The BDDL files define occluded task variants. The init files contain the corresponding initial states used by LIBERO.

## Usage

Install the assets into an upstream LIBERO checkout:

```bash
export LIBERO_ROOT=/path/to/LIBERO
bash scripts/setup/install_libero_occ_assets.sh
```

Then use the suite names above in your LIBERO evaluation code.

