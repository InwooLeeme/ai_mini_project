"""시간 초과 재시도 정책의 대상이 아닌 MCP 경계의 오류를 정의한다."""


class MCPCallError(ValueError):
    """서버 응답 원문이나 통신 비밀값 대신 오류 코드와 안전한 메시지를 전달한다."""

    def __init__(self, code: str, message: str):
        """호출자가 오류를 구분할 코드와 공개 가능한 메시지를 보관한다."""
        self.code = code
        super().__init__(message)
