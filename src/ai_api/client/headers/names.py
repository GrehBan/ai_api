"""Header names and values used by the library."""

import sys
from typing import Final

from multidict import istr

from ai_api.__meta__ import __version__

USER_AGENT: Final[istr] = istr("User-Agent")
SERVER_SOFTWARE: Final[str] = (
    f"Python/{sys.version_info[0]}.{sys.version_info[1]} ai_api/{__version__}"
)
"""Value of the ``User-Agent`` header sent by the library."""
CONTENT_TYPE: Final[istr] = istr("Content-Type")
AUTHORIZATION: Final[istr] = istr("Authorization")
LOCATION: Final[istr] = istr("Location")
