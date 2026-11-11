"""Privately extend only the reviewed runner's actual GPU domain to0..5."""
import ast
import inspect
from pathlib import Path
from types import FunctionType

from experiments.cat_batch_gather_cmcc_20261104 import search as batched
from experiments.cat_batch_gather_cmcc_20261104.fit import _validate_identity
from experiments.cat_auto_protocol_cmcc_20261104.contracts import require
from experiments.cat_auto_protocol_cmcc_20261104.adapter import file_sha


class _Cards(ast.NodeTransformer):
    def __init__(self): self.count = 0

    def visit_Compare(self, node):
        if (isinstance(node.left, ast.Subscript) and isinstance(node.left.value, ast.Name)
                and node.left.value.id == 'request' and isinstance(node.left.slice, ast.Constant)
                and node.left.slice.value == 'physical_GPU' and len(node.ops) == 1 and isinstance(node.ops[0], ast.In)):
            require(ast.literal_eval(node.comparators[0]) == (1, 3, 4, 5), 'Reviewed actual host GPU policy changed.')
            node.comparators[0] = ast.Tuple(elts=[ast.Constant(value=value) for value in range(6)], ctx=ast.Load())
            self.count += 1
        return node


def _private_validate():
    native = batched.native
    require(file_sha(native.__file__) == batched.NATIVE_SEARCH_SHA256, 'Review changed native runner before extending its host domain.')
    tree = ast.parse(inspect.getsource(native.validate_request)); hook = _Cards(); tree = hook.visit(tree)
    require(hook.count == 1, 'Replace exactly one actual physical_GPU tuple; never rewrite request device identity.')
    ast.fix_missing_locations(tree); namespace = dict(native.validate_request.__globals__)
    exec(compile(tree, str(Path(__file__).resolve())+':GPU0_5_validator', 'exec'), namespace)
    return namespace['validate_request']


def validate_request(request):
    inventory, identity = _private_validate()(request); _validate_identity(identity)
    required = {str(path.resolve()) for folder in (Path(__file__).parent, Path(batched.__file__).parent) for path in folder.glob('*.py')}
    require(required <= set(request['source_sha256']), 'Pin every actual new queue/router and reviewed batched-fit source.')
    return inventory, identity


def execute_candidates(request, *, registry_path=None):
    original = batched._private_execute(); namespace = dict(original.__globals__); namespace['validate_request'] = validate_request
    entry = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
    entry.__kwdefaults__ = original.__kwdefaults__
    return entry(request, registry_path=registry_path)


def main():
    native = batched.native; namespace = dict(native.main.__globals__)
    namespace.update(__package__=__package__, validate_request=validate_request, execute_candidates=execute_candidates)
    entry = FunctionType(native.main.__code__, namespace, native.main.__name__, native.main.__defaults__, native.main.__closure__)
    entry.__kwdefaults__ = native.main.__kwdefaults__; return entry()


if __name__ == '__main__': main()
