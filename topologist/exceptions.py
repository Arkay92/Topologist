class TopologistError(Exception):
    """Base error for Topologist."""


class NodeNotFoundError(TopologistError):
    """Raised when a required node is missing."""


class PersistenceError(TopologistError):
    """Raised when save/load fails."""


class ValidationError(TopologistError):
    """Raised when graph or rule data is invalid."""
