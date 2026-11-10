"""bugnav: external Bug-algorithm obstacle exploration for a Deebot T80s.

The robot firmware is closed, so this logic runs on an external host and
talks to the robot through the (unofficial) deebot-client cloud API.
The core in this package is transport-agnostic: it consumes position
samples and emits high-level commands.  Adapters (simulator, real robot)
live outside the core.
"""

from .controller import BugNavController, ControllerConfig
from .model import Command, CommandKind, Sample

__all__ = ["BugNavController", "ControllerConfig", "Command", "CommandKind", "Sample"]
