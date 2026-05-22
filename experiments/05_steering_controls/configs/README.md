# Steering Repair Configs

Place steering repair configs here. Configs should record:

- model and model dtype;
- layer and seed;
- task set and template set;
- calibration/evaluation examples per class;
- shared-basis source, rank, and max states;
- beta/lambda/random-control grids;
- candidate calibration settings;
- output directory.

Suggested filename:

```text
model=<model_slug>__layer=<layer>__tasks=<task_slug>__seed=<seed>.yaml
```
