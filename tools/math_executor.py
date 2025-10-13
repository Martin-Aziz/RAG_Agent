from pydantic import BaseModel
from typing import Dict, Any


class CalcArgs(BaseModel):
    expression: str


class CalcResult(BaseModel):
    expression: str
    value: float
    units: str = None


class Calculator:
    @staticmethod
    def spec():
        return type("ToolSpec", (), {"name": "calculator", "arg_model": CalcArgs, "result_model": CalcResult, "executor": Calculator.execute})

    @staticmethod
    def execute(args: Dict[str, Any]) -> Dict[str, Any]:
        expr = args.get("expression")
        import ast
        import operator

        ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.Mod: operator.mod,
            ast.FloorDiv: operator.floordiv,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }

        def _eval(node):
            if isinstance(node, ast.Expression):
                return _eval(node.body)
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    return node.value
                raise ValueError("Unsupported constant type")
            if isinstance(node, ast.BinOp):
                left = _eval(node.left)
                right = _eval(node.right)
                op_type = type(node.op)
                if op_type in ops:
                    return ops[op_type](left, right)
                raise ValueError("Unsupported binary operator")
            if isinstance(node, ast.UnaryOp):
                operand = _eval(node.operand)
                op_type = type(node.op)
                if op_type in ops:
                    return ops[op_type](operand)
                raise ValueError("Unsupported unary operator")
            raise ValueError(f"Unsupported expression: {type(node)}")

        try:
            node = ast.parse(expr, mode="eval")
            value = _eval(node)
            return {"expression": expr, "value": float(value), "units": None}
        except Exception:
            return {"expression": expr, "value": 0.0, "units": None}
