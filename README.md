# Transformers-HuggingFace-with-docs
Transformers-HuggingFace-with-docs

## STEPS -

### STEP 01- Create a repository by using template repository

### STEP 02- Clone the new repository

### STEP 03- Create a conda environment after opening the repository in VSCODE

```bash
conda create --prefix ./env python=3.8 -y
```

```bash
conda activate ./env
```
OR
```bash
source activate ./env
```

### STEP 04- install the requirements
```bash
pip install -r requirements.txt
```


### STEP 05- commit and push the changes to the remote repository

### STEP 06- Run init_setup to create environment
```bash
bash init_setup.sh
```

### STEP 07- Create your mkdocs site
```bash
mkdocs new .
```

### STEP 08- Run mkdocs server
```bash
mkdocs serve
```

https://squidfunk.github.io/mkdocs-material/creating-your-site/

```bash
pip install ipykernel
```
