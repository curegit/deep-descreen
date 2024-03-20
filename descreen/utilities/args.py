import ast
import typing


def nonempty(string):
    if string:
        return string
    else:
        raise ValueError()


def fileinput(string):
    # stdin (-) を None で返す
    if string == "-":
        return None
    return nonempty(string)

def eqsign_kvpairs(string: str) -> dict[str, typing.Any]:
    tree = ast.parse(f"func({string})", mode="eval")
    match tree.body:
        case ast.Call() as call:
            if call.args:
                raise ValueError("Only keyword args allowed")
            return {str(kw.arg): ast.literal_eval(kw.value) for kw in call.keywords}
    raise ValueError()
