import ast
import typing


def eqsign_kvpairs(string: str) -> dict[str, typing.Any]:
    tree = ast.parse(f"func({string})", mode="eval")
    match tree.body:
        case ast.Call() as call:
            if call.args:
                raise ValueError("Only keyword args allowed")
            return {str(kw.arg): ast.literal_eval(kw.value) for kw in call.keywords}
    raise ValueError()
