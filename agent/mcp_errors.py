"""Permanent MCP boundary errors (never matched by the timeout retry policy)."""


class MCPCallError(ValueError):
    """A safe, actionable failure without server payloads or transport secrets."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)
