# GitHub Push Guide

This project is initialized as a local `main` branch repository. It has no
remote and Codex did not create a commit or push anything.

## Review and commit locally

```bash
cd /Users/paxton/Project/PythonProject/02_CL_Papers/UIE3_workspace/UIE3_VN

git status
git diff --check
git add .
git commit -m "feat: initialize v1 NAFNet-small experiment project"
```

## Create an empty GitHub repository

Create the repository in GitHub first. Leave all initialization options off:

- do not add a README;
- do not add a license;
- do not add a `.gitignore`.

Do not substitute a guessed account or repository. Replace both placeholders
below with the values you created.

### SSH

```bash
git remote add origin git@github.com:<USERNAME>/<REPOSITORY>.git
git push -u origin main
```

### HTTPS

```bash
git remote add origin https://github.com/<USERNAME>/<REPOSITORY>.git
git push -u origin main
```

Use one remote form, not both.

## Verify

```bash
git remote -v
git status
git log --oneline -1
```
