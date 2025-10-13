from typing import Callable, Any, Dict
from pydantic import BaseModel, ValidationError
from core.observability import get_logger

logger = get_logger("tool_registry")


class ToolError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class ToolSpec:
    def __init__(self, name: str, arg_model: BaseModel, result_model: BaseModel, executor: Callable[[dict], dict]):
        self.name = name
        self.arg_model = arg_model
        self.result_model = result_model
        self.executor = executor


class ToolRegistry:
    def __init__(self):
        self.tools = {}
        # register default tools
        from tools.web_search_stub import WebSearch
        from tools.math_executor import Calculator

        self.register("web_search", WebSearch.spec())
        self.register("calculator", Calculator.spec())

    def register(self, name: str, spec: ToolSpec):
        self.tools[name] = spec

    def call(self, name: str, args: dict) -> dict:
        spec = self.tools.get(name)
        if not spec:
            raise ToolError("not_found", f"Tool {name} not found")
        # validate args
        try:
            validated = spec.arg_model(**args)
        except ValidationError as e:
            logger.warning("tool arg validation failed", extra={"error": str(e)})
            raise ToolError("invalid_args", str(e))
        res = spec.executor(validated.dict())
        # validate result
        try:
            validated_res = spec.result_model(**res)
        except ValidationError as e:
            logger.warning("tool result validation failed", extra={"error": str(e)})
            raise ToolError("invalid_result", str(e))
        return validated_res.dict()
