"""Finite native model runner routed to the source-bound batched fit."""
import ast
import inspect
from pathlib import Path
from types import FunctionType

from experiments.cat_auto_protocol_cmcc_20261104 import search as native
from experiments.cat_auto_protocol_cmcc_20261104.adapter import file_sha
from experiments.cat_auto_protocol_cmcc_20261104.contracts import require
from .fit import _validate_identity

NATIVE_SEARCH_SHA256 = '3a6699fff10ecbe1d92e3087c23d262f51dbea8068c97775e5da259bf20688b4'


def validate_request(request):
    inventory, identity = native.validate_request(request)
    _validate_identity(identity)
    actual = {str(path.resolve()) for path in Path(__file__).parent.glob('*.py')}
    require(actual <= set(request['source_sha256']), 'Pin every actual batched-runner source.')
    return inventory, identity


class _FitImport(ast.NodeTransformer):
    def __init__(self):
        self.count = 0

    def visit_ImportFrom(self, node):
        if node.level == 1 and node.module == 'fit_adapter':
            require([(item.name, item.asname) for item in node.names] ==
                [('fit_category', None), ('install_category_bank', None)],
                'The actual native fit import contract changed.')
            self.count += 1
            return [
                ast.copy_location(ast.ImportFrom(module=__package__ + '.fit',
                    names=[ast.alias(name='fit_category')], level=0), node),
                ast.copy_location(ast.ImportFrom(module='fit_adapter',
                    names=[ast.alias(name='install_category_bank')], level=1), node),
            ]
        return node


def _private_execute():
    require(file_sha(native.__file__) == NATIVE_SEARCH_SHA256,
        'The native runner changed; review the actual fit call-site before reuse.')
    tree = ast.parse(inspect.getsource(native.execute_candidates))
    hook = _FitImport()
    tree = hook.visit(tree)
    require(hook.count == 1, 'Expected exactly one actual native fit import.')
    ast.fix_missing_locations(tree)
    namespace = dict(native.execute_candidates.__globals__)
    namespace['validate_request'] = validate_request
    exec(compile(tree, str(Path(__file__).resolve()) + ':native_execute', 'exec'), namespace)
    return namespace['execute_candidates']


def execute_candidates(request, *, registry_path=None):
    return _private_execute()(request, registry_path=registry_path)


def main():
    # Reuse the original finite process-group timeout/cleanup and logging. Its
    # child module follows this package, so the child also uses the new fit.
    namespace = dict(native.main.__globals__)
    namespace.update(__package__=__package__, validate_request=validate_request,
        execute_candidates=execute_candidates)
    entry = FunctionType(native.main.__code__, namespace, native.main.__name__,
        native.main.__defaults__, native.main.__closure__)
    entry.__kwdefaults__ = native.main.__kwdefaults__
    return entry()


if __name__ == '__main__':
    main()
