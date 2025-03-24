# Run existing tests

Activate virtual environment:
- pip-way: `source venv/bin/activate`
- poetry-way: `poetry env activate`

`pytest` must be already installed with the main requirements. If not:
- pip-way: `pip install pytest`
- poetry-way: `poetry add pytest`

Run tests:
``bash
pytest tests/
```

``bash
pytest tests/test_probability.py
```

# Add new tests

