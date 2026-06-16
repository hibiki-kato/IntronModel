# History Rewrite Playbook

This playbook removes historical large files under `data/**` from Git history.
Use only after team alignment.

## Preconditions

- Confirm all collaborators are ready to re-clone.
- Ensure a clean working tree.
- Ensure `git filter-repo` is installed.

## Steps

1. Create freeze tag:

```bash
git tag pre-data-history-rewrite-20260219
```

2. Rewrite history:

```bash
git filter-repo --path data --invert-paths
```

3. Expire reflog and garbage-collect:

```bash
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

4. Verify large data objects are absent:

```bash
git rev-list --objects --all | rg '^.+\s+data/'
```

5. Push rewritten history:

```bash
git push --force --all
git push --force --tags
```

## Collaborator Recovery

Recommended recovery after rewrite:

```bash
git fetch --all --prune
cd ..
rm -rf IntronModel
git clone <remote-url> IntronModel
```

## Safety Notes

- History rewrite changes commit hashes for all rewritten commits.
- Never run this process while unresolved local work exists.
