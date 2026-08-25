# Graph-local Pet preflight

This is an engineering preflight for the graph-local continual-update idea. It
does not replace the eventual VOC/COCO incremental-object-detection study.

Before invoking the Pet detector run, verify the six module paths without a
compiled detector or downloaded dataset:

```powershell
C:\Users\23642\miniconda3\envs\tree-detr\python.exe tools\graph_local\run_preflight.py `
    --output exps\pet_graph_local\synthetic_preflight.json
```

The report is deliberately limited to implementation feasibility: it verifies
non-zero graph selection, balanced replay, pseudo-label filtering, local-margin
gradients, off-neighborhood projection, and LoRA merge equivalence. It is not a
positive result for the locality hypothesis.

## Optional trainable GNN estimator

The default runner uses the transparent gradient-cosine estimator. The
trainable GNN is an optional stage-level side-car: it consumes compressed class
gradient sketches from the frozen detector and predicts directed harm scores;
it is not inserted into the detector backbone or image-level forward pass.

Every increment run writes `gnn_stage.pt`. The artifact contains compressed
class sketches, the empirical one-step harm row for the probed new class, and a
validity mask so unmeasured source rows are not treated as zero-harm labels.
Train a GNN only from **prior** stage artifacts:

```powershell
C:\Users\23642\miniconda3\envs\tree-detr\python.exe `
  tools\graph_local\train_gnn.py `
  --stages exps\pet_graph_local\increment_seed42\gnn_stage.pt `
           exps\pet_graph_local\increment_seed43\gnn_stage.pt `
  --output exps\pet_graph_local\class_interference_gnn.pt
```

Use the learned estimator for a later increment by passing its checkpoint:

```powershell
C:\Users\23642\miniconda3\envs\tree-detr\python.exe `
  tools\graph_local\run_increment.py `
  --graph-estimator gnn `
  --gnn-checkpoint exps\pet_graph_local\class_interference_gnn.pt `
  ...
```

The runner rejects GNN mode without a checkpoint. This preserves the causal
protocol: the current stage's probe harm is saved for future GNN training, but
does not train or select the neighborhood used by that same stage.

## What the script runs

`tools/windows/run_graph_local_pet.ps1` uses a fixed Oxford-IIIT Pet split:

- 29 known classes train the base detector;
- 20 labeled `Abyssinian` images form the first new-class increment;
- the increment validation set has all 29 old classes plus that new class;
- a graph arm, a matched random-neighborhood arm, and a global-replay LoRA arm
  are available after the quality gate passes.

The base detector must reach `AP50 >= 0.40` on the known-only validation split.
If it does not, the Python runner writes `summary.json` with an `inconclusive`
verdict and does not train the incremental arms. This is intentional: a low
quality detector cannot test whether forgetting is local.

## Run

Before any incremental run, use the diagnostic runner. It first requires an
overfit pass on identical train/validation images, then evaluates a balanced
six-known-class baseline. It does not run incremental adaptation.

```powershell
.\tools\windows\run_pet_baseline_diagnostics.ps1
```

The runner stops unless overfit `AP50 >= 0.90` and small-task validation
`AP50 >= 0.40`. This prevents the graph-local experiment from inheriting an
unreliable detector.

If that fresh baseline overfits but does not generalize, use a generic COCO
detector checkpoint, not a previous Pet checkpoint. The expected architecture
is the original single-stage Deformable-DETR configuration: ResNet-50, six
encoder layers, six decoder layers, 1024-dim FFN, 300 queries, no box-refine
and no two-stage head. Then run:

```powershell
.\tools\windows\run_pet_coco_pretrained_baseline.ps1 `
    -CocoCheckpoint C:\programlearning\Tree-DETR\pretrained\r50_deformable_detr.pth
```

The loader retains only equal-shape tensors, so the COCO classifier is discarded
and the remapped Pet classifier is trained at a higher learning rate.

From PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
Set-Location C:\programlearning\Tree-DETR
.\tools\windows\run_graph_local_pet.ps1 -BaselineEpochs 60 -Seed 42
```

To evaluate a baseline checkpoint already trained elsewhere:

```powershell
.\tools\windows\run_graph_local_pet.ps1 `
  -BaselineCheckpoint .\exps\pet_feasibility\baseline_pretrain\checkpoint.pth `
  -Seed 42
```

`-RunDespiteQualityGate` exists only for implementation smoke tests. Its output
is explicitly marked `engineering-only` and must not be used as evidence for
the hypothesis.

For a bounded exploratory ablation, add `-ModuleAblations`. Alongside the
graph, random, and global replay arms, it runs graph-local variants without
pseudo labels, the local margin, the off-neighborhood projection, or replay.
Use `-OutputTag` to keep exploratory results separate from the standard run.
These comparisons are diagnostic only when the base quality gate fails.

## Three-seed protocol

Keep `-DataSeed 42` fixed, then run independent detector and increment seeds:

```powershell
.\tools\windows\run_graph_local_pet.ps1 -BaselineEpochs 60 -Seed 42
.\tools\windows\run_graph_local_pet.ps1 -BaselineEpochs 60 -Seed 43
.\tools\windows\run_graph_local_pet.ps1 -BaselineEpochs 60 -Seed 44
```

Each run writes `exps\pet_graph_local\increment_seed<seed>\summary.json`.
Do not promote the idea unless the quality gate passes and graph-local replay
beats the matched random-neighborhood arm with the pre-registered criteria in
`Ideas/graph-local-adapter-falsification-plan.md`.
