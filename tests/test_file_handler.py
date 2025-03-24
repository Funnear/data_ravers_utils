import os
import pickle
import pytest
from unittest.mock import patch, mock_open, MagicMock
from data_ravers_utils.file_handler import save_model_pickle, MODELS_PATH, GITIGNORE_PATH, hash_object

# Tests for file_handler.py
@pytest.fixture
def mock_model():
    return {"key": "value"}

def test_save_model_pickle_creates_directory(mock_model, tmp_path, monkeypatch):
    mock_models_path = tmp_path / "models"
    monkeypatch.setattr("data_ravers_utils.file_handler.MODELS_PATH", str(mock_models_path))
    with patch("pickle.dump"):
        save_model_pickle(mock_model, "test_model")
        assert mock_models_path.exists()

def test_save_model_pickle_saves_model(mock_model, tmp_path, monkeypatch):
    mock_models_path = tmp_path / "models"
    monkeypatch.setattr("data_ravers_utils.file_handler.MODELS_PATH", str(mock_models_path))
    with patch("pickle.dump") as mock_pickle_dump:
        save_model_pickle(mock_model, "test_model")
        saved_file = mock_models_path / "test_model.pkl"
        assert saved_file.exists()
        mock_pickle_dump.assert_called_once()

def test_save_model_pickle_skips_save_if_no_changes(mock_model, tmp_path, monkeypatch):
    mock_models_path = tmp_path / "models"
    monkeypatch.setattr("data_ravers_utils.file_handler.MODELS_PATH", str(mock_models_path))
    saved_file = mock_models_path / "test_model.pkl"
    saved_file.parent.mkdir(parents=True, exist_ok=True)
    with saved_file.open("wb") as f:
        pickle.dump(mock_model, f)

    with patch("pickle.dump") as mock_pickle_dump:
        save_model_pickle(mock_model, "test_model")
        mock_pickle_dump.assert_not_called()

def test_save_model_pickle_updates_gitignore_if_file_exceeds_limit(mock_model, tmp_path, monkeypatch):
    mock_models_path = tmp_path / "models"
    mock_gitignore_path = tmp_path / ".gitignore"
    monkeypatch.setattr("data_ravers_utils.file_handler.MODELS_PATH", str(mock_models_path))
    monkeypatch.setattr("data_ravers_utils.file_handler.GITIGNORE_PATH", str(mock_gitignore_path))

    large_file_size = 200 * 1024 * 1024  # 200 MB
    with patch("os.path.getsize", return_value=large_file_size):
        save_model_pickle(mock_model, "test_model")
        assert mock_gitignore_path.exists()
        with mock_gitignore_path.open() as f:
            assert "models/test_model.pkl" in f.read()

def test_save_model_pickle_logs_warning_on_load_failure(mock_model, tmp_path, monkeypatch):
    mock_models_path = tmp_path / "models"
    monkeypatch.setattr("data_ravers_utils.file_handler.MODELS_PATH", str(mock_models_path))
    with patch("pickle.load", side_effect=Exception("Load error")), patch("logging.warning") as mock_warning:
        save_model_pickle(mock_model, "test_model")
        mock_warning.assert_any_call("Could not load existing model for comparison: Load error")