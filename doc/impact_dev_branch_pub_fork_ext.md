# Creating the `impact_dev` Branch with Public Fork of External Repos as Submodules

Create impact_dev branch locally then push to TRACTOR remote repo

```
git clone https://github.com/genesys-neu/TRACTOR/ && cd TRACTOR
git branch impact_dev
git checkout impact_dev
git push origin impact_dev
```

Verify `impact_dev` appears in the branch list of `https://github.com/genesys-neu/TRACTOR/`.

On GitHub, create a public fork `colosseum-scope-e2`. The fork is public because the external repo is public.

Add the fork of `colosseum-scope-e2` as a submodule to `TRACTOR`. This needs to be done on the command line because
some IDEs do not support the `git submodule add` as a built-in feature.

```
git submodule add https://github.com/<owner-account>/colosseum-scope-e2
git add .gitmodules
git add colosseum-scope-e2
git commit -m "<commit-message>"
git push origin impact_dev
```

Verify the file `.gitmodules` and the folder `colosseum-scope-e2@<latest-commit>` appears in the `impact_dev` branch.

`TRACTOR` is now the parent repo of the submodule `colosseum-scope-e2`.

The parent repo tracks changes to
the reference `@<latest-commit.` pointing to the revision of the submodule. When new commits are added to the
submodule, the parent can choose to update its reference to submodule revision. The parent does not need to update
its reference as often as the submodule is updated.

**NOTE:**

- Changes to the reference appear in the commit history of the parent.
- Changes to the submodule appear in the commit history of the submodule.

**NOTE:**

When using some IDEs such as PyCharm and CLion, you need to add the path to the submodule to the `directory mapping`
in the IDE settings. Otherwise, only changes to the reference will be the tracked by the Git GUI of the IDE, but not
changes to the submodule. This step is important if you wish to develop for both the parent and submodule form the
parent IDE project, as opposed to developing for each repo as individual IDE projects

**TODO**: **_more instructions on this step._**

Add a new feature to the submodule.

```
cd <root>/TRACTOR/colosseum-scope-e2
touch <new_source_file>
git add <new_source_file>
git commit -m "<commit-message>"
git push origin main
```

**NOTE:**
You are running the above Git commands inside the submodule directory. Therefore, push target `origin main` is the
`main`
branch of the submodule, **_not_** the `main` branch of the parent. Recall, the parent is still on the `impact_dev`
branch.

Update the parent's reference to the submodule to the latest commit

```
cd <root>/TRACTOR
git add colosseum-scope-e2
git commit -m "<commit-message>"
git push origin impact_dev
```

**NOTE:**
You are running the above Git commands inside the parent directory. Therefore, push target is the
`impact_dev` branch.

When using IDE to add features to the submodule, two changes appears in the commit staging list of the IDE:

1. the parent's reference to
   the submodule
2. the edited file in the submodule.

You can commit both changes together or individually.
Since each commit targets a different repo and its respective branch, if you commit both changes together, the same
commit message appears in the history of both targets. Therefore, provide a message that makes sense for both
repos such as _"add feature
x to
submodule y and update parent's reference".

**NOTE:** 
In JetBrains IDEs (i.e. PyCharm, Clion), if you are working on the main branch of the submodule, and an experimental 
branch of the parent, then the IDE Git GUI indicates that there is a commit to be pushed on the experimental branch 
and the main branch. This GUI is very confusing, because the main branch of the parent and the main branch of the 
submodule are superimposed on the same graphic. Therefore, it might seem like you are about to the push the commit 
to the main branch of the parent, which would be **incorrect!** However, with a closer look at the GUI, you can 
confirm that the commit is in fact targeting the main branch of the submodule. **TODO: provide some screenshots to 
show this problem**

Cherry pick the slice reassignment feature modification intended for the `colosseum-scope-e2` repo from 
`refactor_scope_as_submodule` branch to `impact_dev` branch.

```
cd <root>/TRACTOR
git checkout impact_dev
git cherry-pick c0a81303
# Resolve the conflict by accepting "their" changes, which belongs to branch you are cherry-picking from. 
git cherry-pick continue
```
At this point, do not commit the changes of the cherry-pick to the `impact_dev` branch. Instead, copy these changes 
to the submodule

```
cp colosseum/radio_code/colosseum-scope-e2/src/du_app/srs_connector.c colosseum-scope-e2/src/du_app/srs_connector.c
cp colosseum/radio_code/colosseum-scope-e2/src/du_app/srs_connector.h colosseum-scope-e2/src/du_app/srs_connector.h
git commit -m "<commit-message>"
git push origin impact_dev
```
At this point, you can delete the changes of the cherry-pick from `impact_dev` branch on your local repo.

Create a public fork of `colosseum-scope` and add as a submodule to `TRACTOR` by following the same instructions for
adding `colosseum-scope-e2` as a submodule

Cherry pick the slice reassignment feature modification intended for the `colosseum-scope` repo from
`refactor_scope_as_submodule` branch to `impact_dev` branch.

```
cd <root>/TRACTOR
git checkout impact_dev
git cherry-pick 94993572
# Resolve the conflict by accepting "their" changes, which belongs to branch you are cherry-picking from. 
git cherry-pick continue
```
At this point, do not commit the changes of the cherry-pick to the `impact_dev` branch. Instead, copy these changes
to the submodule. Follow the same steps as the before. 

