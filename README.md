# data_ravers_utils
A library of code snippets for Data Science projects.


## Setup

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

