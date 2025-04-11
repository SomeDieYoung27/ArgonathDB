"""RAGLite config."""
import contextlib
import os
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Literal

from platformdirs import user_data_dir
from sqlalchemy.engine import URL
