# Team workflow

Use your own GitHub account and Git author identity. Collaborator access must be granted by the repository owner after confirming account names.

1. Pick an issue and make a short-lived task branch from current `main`, e.g. `simul/adrec-training`, `anusha/gru4rec-training`, or `mahalakshmi/sasrec-training`.
2. Make small, focused commits. Do not upload datasets, weights, credentials, or notebook secrets.
3. Run a small validation appropriate to the change and record the outcome.
4. Push your branch and open a pull request linking the issue with `Closes #NUMBER`.
5. Ask another member to review it. Address feedback before merging.
6. Update your local `main` and delete the merged task branch.

Keep `main` runnable once implementation exists. Do not merge failing code. One teammate approval is a team rule; it is not currently enforced by repository protection settings. Coordinate before editing shared data/evaluation code.

## Pull request evidence

Include what changed, the exact command or notebook used, environment, and a short result. Full training is not required for documentation-only changes. For model changes, include input/output shape checks and a small run; for metrics, check a tiny example with a known ranking.

## Colab/Kaggle

Use a recorded repository commit. Save checkpoints outside Git and log their location. Save the configuration, random seed, package versions, hardware, and small results in Git. Clear unnecessary notebook output before committing.

## Completion rule

A task is complete when its deliverable is reproducible, reviewed by another member, and merged. Contribution quality is measured through working changes, experiments, and reviews, not commit counts.
