import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location('docs_tool', Path(__file__).resolve().parents[1] / 'tools/docs.py')
docs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(docs)


class DocsCategoriesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name, body in {
            'docs/experience/lessons/compression/fitting.md': '# 拟合\n\n学习率 COMPRESSION\n',
            'docs/experience/lessons/category_recovery/optimization.md': '# 恢复优化\n\n学习率 RECOVERY\n',
            'docs/experience/lessons/category_recovery/scope.md': '# 恢复范围\n\n学习率 SCOPE\n',
            'docs/experience/records/example.md': '# 记录\n\n学习率 RECORD\n',
            'docs/guide.md': '# 指南\n\n学习率 GUIDE\n',
            'docs/experience/lessons/legacy.md': '# 旧入口\n\n<!-- lesson-redirect -->\n学习率 REDIRECT\n<!-- /lesson-redirect -->\n\n学习率 CONCURRENT\n',
        }.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body, encoding='utf-8')

    def run_cli(self, *args):
        output = io.StringIO()
        with patch.object(docs, 'ROOT', self.root), patch('sys.argv', ['docs.py', *args]):
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                try:
                    code = docs.main()
                except SystemExit as exc:
                    code = exc.code
        return code, output.getvalue()

    def test_category_filters_other_stages_and_records(self):
        code, output = self.run_cli('search', '学习率', '--category', 'category_recovery')
        self.assertEqual(code, 0)
        self.assertIn('RECOVERY', output)
        self.assertIn('SCOPE', output)
        self.assertNotIn('COMPRESSION', output)
        self.assertNotIn('RECORD', output)

    def test_chinese_category_and_topic(self):
        code, output = self.run_cli('search', '学习率', '--category', '逐类别恢复', '--topic', 'optimization')
        self.assertEqual(code, 0)
        self.assertIn('optimization.md:3:', output)
        self.assertNotIn('SCOPE', output)

    def test_original_default_and_record_scope(self):
        code, output = self.run_cli('search', '学习率')
        self.assertEqual(code, 0)
        self.assertIn('COMPRESSION', output)
        self.assertIn('GUIDE', output)
        self.assertNotIn('RECORD', output)
        code, output = self.run_cli('search', '学习率', '--scope', 'records')
        self.assertEqual(code, 0)
        self.assertIn('RECORD', output)
        self.assertNotIn('RECOVERY', output)

    def test_redirect_does_not_hide_concurrent_append(self):
        code, output = self.run_cli('search', '学习率', '--scope', 'lessons')
        self.assertEqual(code, 0)
        self.assertNotIn('REDIRECT', output)
        self.assertIn('legacy.md:7: 学习率 CONCURRENT', output)

    def test_categories_list_actual_topics(self):
        code, output = self.run_cli('categories', '--category', 'category_recovery')
        self.assertEqual(code, 0)
        self.assertIn('2 个专题', output)
        self.assertIn('optimization — 恢复优化', output)
        self.assertNotIn('fitting', output)

    def test_redirect_does_not_hide_concurrent_prefix(self):
        path = self.root / 'docs/experience/lessons/legacy.md'
        path.write_text('学习率 PREPENDED\n' + path.read_text(encoding='utf-8'), encoding='utf-8')
        code, output = self.run_cli('search', '学习率', '--scope', 'lessons')
        self.assertEqual(code, 0)
        self.assertIn('legacy.md:1: 学习率 PREPENDED', output)
        self.assertIn('legacy.md:8: 学习率 CONCURRENT', output)
        self.assertNotIn('REDIRECT', output)

    def test_invalid_filters_are_explicit_errors(self):
        for options in [
            ['--category', 'unknown'],
            ['--topic', 'optimization'],
            ['--category', 'category_recovery', '--topic', '../compression/fitting'],
            ['--category', 'category_recovery', '--scope', 'records'],
        ]:
            with self.subTest(options=options):
                code, output = self.run_cli('search', '学习率', *options)
                self.assertEqual(code, 2)
                self.assertIn('error:', output)

    def test_limit_and_regex_remain_available(self):
        code, output = self.run_cli('search', 'RECOVERY|SCOPE', '--category', 'category_recovery', '--limit', '1')
        self.assertEqual(code, 0)
        self.assertIn('Match limit reached', output)
        self.assertEqual(output.count('.md:'), 1)


if __name__ == '__main__':
    unittest.main()
