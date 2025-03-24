# Draft setup guide

## Disclaimer

- Use pip for projects where this package is integrated as a submodule.
- Use poatry for development and maintainance of the original code.
- Rationel: developing the submodule on the go wile using it in Jupyter notebooks is not fully supported by Poetry yet. 

TODO:
- [ ] Follow up on this problem in Python Developers Community on Dicogs

## How to install submodule

To use as a submodule in project with Jupyter Notebooks:
```bash
# update this line to the downstream project:
git clone https://github.com/username/project_name.git; \
git submodule add https://github.com/Funnear/data_ravers_utils.git src/data_ravers_utils; \
git submodule update --init --recursive; \
python3 -m venv venv; \
source venv/bin/activate; \
pip install --upgrade pip; \
# requirements must include ipykernel
pip install -r requirements.txt; \ 
pip install -e ./src/data_ravers_utils; \
python -m ipykernel install --user --name=venv --display-name "Jupyter (venv)"
```

## technical guidelines
- [Poetry](./poetry.md)
- [Unit tests](./pytest.md)

