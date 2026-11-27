import os
import tempfile

_tmp = tempfile.mkdtemp(prefix="hooklab-test-")
os.environ["DATA_DIR"] = _tmp
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}/test.db"
os.environ.pop("ANTHROPIC_API_KEY", None)

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def weak_script():
    return open("samples/romance_weak_opening.txt").read()


@pytest.fixture()
def strong_script():
    return open("samples/romance_strong_opening.txt").read()
