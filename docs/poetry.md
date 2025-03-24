# Package managemement: Poetry (.toml)

This method is not recommended for using with editable submodules imported in jupyter Notebooks.

Poetry (pyproject.toml)

### Kick start for Mac users wit homebrew
```bash
brew install poetry
```

Inside the project folder:
```bash
poetry init
```
- this will require manual inputs.

if there are requirements for pip:
```bash
poetry add $(cat requirements.txt)
```

If you want to use Poetry only for dependency management but not for packaging, 
you can disable package mode by setting `package-mode = false` in your pyproject.toml file.

create venv:
```bash
poetry install; \
poetry env use python3
```

VS Code needs to know about the Poetry envs from PATH:
```bash
poetry env info --path
```

VS Code command pallete -> Open User Settings (JSON) -> copy the output from the previous command to add line:
```
"python.venvPath": "/Users/user_name/Library/Caches/pypoetry/virtualenvs"
```

### Regular usage

To activate virtual environment:
```bash
poetry env activate
```

Add library, for example:
```bash
poetry add toml
```

### Maintenance

```bash
poetry update
```

TODO: make a shell script
```bash
#!/bin/bash
poetry update; \
rm -rf requirements.txt; \
poetry run python -c "import toml; print('\n'.join([f'{p[\"name\"]}=={p[\"version\"]}' for p in toml.load(open('poetry.lock'))['package']]))" > requirements.txt; \
git add pyproject.toml poetry.lock requirements.txt; \
git commit -m "Updated dependencies"; \
git push origin main
```


