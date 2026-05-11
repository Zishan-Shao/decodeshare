# Troubleshooting

## Missing Model or Dataset Access

Check that Hugging Face authentication is configured outside the repository and
that gated model access has been accepted for the account running the job.

## GPU or Node Issues

For the current camera-ready pass, use only `Node0` and `Node1`. If a script
records a different node target, update the command before rerunning.

## Out-of-Memory Runs

Record the model, layer, rank, batch size, and GPU type in the run log before
retrying. Prefer lowering batch size before changing the experiment definition.

## Artifact Not Found

Check `MANIFEST.md` first. Large raw artifacts are expected to live outside git,
while compact tables, figures, and summaries should live under
`paper_artifacts/`.
