# NLP_Toolkit
Toolkits to pre-process text to be trained using NLP

**1) Installing conda environment**

```
    conda env create --name nlpToolkit --file requirements.yaml 
```

or

```
    pip install -r requirements.txt
```

**2) Updating conda environment**

```
    conda env export > requirements.yaml
```

or

```
    pip list --format=freeze > requirements.txt
```

### Add submodule

```
git submodule add -b main https://github.com/fmorenovr/nlpToolkit.git path_to_install/nlpToolkit
git submodule update --remote
```

### Delete submodule

* First, remove local directories:
```
git rm --cached path_to_submodule
rm -rf path_to_submodule
```

* Then, remove it in files `.gitsubmodules` and `.git/config`.

* Finally, remove it from git:

```
rm -rf .git/modules/path_to_submodule
```
