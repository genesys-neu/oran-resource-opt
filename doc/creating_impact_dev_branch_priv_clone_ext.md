# Creating the `impact_dev` Branch with Private Clone of External Repo as Submodule

Create impact_dev branch locally then push to remote repo

```
git clone https://github.com/genesys-neu/TRACTOR/ && cd TRACTOR
git branch impact_dev
git checkout impact_dev
git push origin impact_dev
```
Verify `impact_dev` appears in the branch list of `https://github.com/genesys-neu/TRACTOR/`.

On GitHub, create a new private repo called `colosseum-scope-e2-dev`, and leave it empty, do not import a repository.

Push the external (public) repo from `colosseum-scope-e2` to the newly created private repo `colosseum-scope-e2-dev`.

```
git clone https://github.com/wineslab/colosseum-scope-e2 && cd colosseum-scope-e2
git remote rename origin upstream
git remote add origin https://github.com/<owner-account>/colosseum-scope-e2-dev
git push origin main
```
Verify the contents and commit history of the `colosseum-scope-e2` appears in `https://github.
com/<owner-account>/colosseum-scope-e2-dev`.

Add the private repo `colosseum-scope-e2-dev` as a submodule to `TRACTOR`
```
git submodule add https://github.com/<owner-account>/colosseum-scope-e2-dev
git status
```

Verify the file `.gitmodules` and the folder `colosseum-scope-e2-dev` are added to `TRACTOR`.

Create then push a commit for refactoring `TRACTOR\colosseum\radio_code\colosseum-scope-e2` into 
`TRACTOR\colosseum-scope-e2-dev`

```shell

```



