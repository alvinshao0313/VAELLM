"""Exercise real SCP, checksums and cleanup against an isolated local sshd."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pwd
import shutil
import socket
import subprocess
import sys
import time

import pytest


def test_scp_upload_checksum_and_cleanup_without_touching_existing_files(tmp_path):
    sshd, keygen = shutil.which('sshd'), shutil.which('ssh-keygen')
    if not sshd or not keygen:
        pytest.skip('需要已有OpenSSH客户端及sshd；测试不安装服务或修改系统配置')
    repo = Path(__file__).resolve().parents[1]
    for name in ('host_key', 'client_key'):
        subprocess.run([keygen, '-q', '-t', 'ed25519', '-N', '', '-f', str(tmp_path / name)], check=True)
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    username = pwd.getpwuid(os.getuid()).pw_name
    server_config = tmp_path / 'sshd_config'
    server_config.write_text(
        f'ListenAddress 127.0.0.1\nPort {port}\nHostKey {tmp_path / "host_key"}\n'
        f'PidFile {tmp_path / "sshd.pid"}\nAuthorizedKeysFile {tmp_path / "client_key.pub"}\n'
        f'AllowUsers {username}\nPasswordAuthentication no\nKbdInteractiveAuthentication no\n'
        'UsePAM no\nStrictModes no\nSubsystem sftp internal-sftp\n', encoding='utf-8')
    known = tmp_path / 'known_hosts'
    key_fields = (tmp_path / 'host_key.pub').read_text().split()
    known.write_text(f'[{"127.0.0.1"}]:{port} {key_fields[0]} {key_fields[1]}\n', encoding='utf-8')
    client_config = tmp_path / 'ssh_config'
    client_config.write_text(
        f'Host vaellm-loopback\n  HostName 127.0.0.1\n  User {username}\n'
        f'  IdentityFile {tmp_path / "client_key"}\n  UserKnownHostsFile {known}\n'
        '  IdentitiesOnly yes\n  BatchMode yes\n  StrictHostKeyChecking yes\n', encoding='utf-8')
    receiver = tmp_path / 'receiver'
    receiver.mkdir()
    sentinel = receiver / 'keep.txt'
    sentinel.write_text('unrelated data must stay', encoding='utf-8')
    original = tmp_path / 'original.bin'
    original.write_bytes(os.urandom(1024 * 1024))
    original_stat = original.stat()
    original_sha = hashlib.sha256(original.read_bytes()).hexdigest()
    with (tmp_path / 'sshd.log').open('w', encoding='utf-8') as server_log:
        server = subprocess.Popen([sshd, '-D', '-e', '-f', str(server_config)], stdout=server_log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 5
            while True:
                if server.poll() is not None:
                    pytest.fail('隔离sshd启动失败：' + (tmp_path / 'sshd.log').read_text())
                try:
                    with socket.create_connection(('127.0.0.1', port), timeout=0.2):
                        break
                except OSError:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(0.05)
            for label, payload, expected_bytes, rounds in [
                ('random', ['--size-mib', '16'], 16 * 1024**2, 2),
                ('existing', ['--file', str(original)], original_stat.st_size, 1),
            ]:
                output = tmp_path / label
                result = subprocess.run(
                    [sys.executable, str(repo / 'tools/scp_benchmark.py'), '--host', 'vaellm-loopback',
                     '--port', str(port), '--ssh-config', str(client_config), '--remote-dir', str(receiver),
                     '--rounds', str(rounds), '--estimate-gib', '700', '--output', str(output), *payload],
                    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
                assert result.returncode == 0, result.stdout
                report = json.loads((output / 'report.json').read_text())
                assert report['status'] == 'passed'
                assert report['direction'] == '本机 -> 新机器'
                assert len(report['rounds']) == rounds
                assert all(r['sha256_verified'] and r['seconds'] > 0 for r in report['rounds'])
                assert report['summary']['total_uploaded_bytes'] == expected_bytes * rounds
                assert report['summary']['aggregate_MiB_per_second'] > 0
                assert report['summary']['estimate']['hours'] > 0
                assert report['remote_test_files_removed'] is True
                assert report['cleanup_errors'] == []
                assert list(receiver.iterdir()) == [sentinel]
            assert sentinel.read_text() == 'unrelated data must stay'
            assert hashlib.sha256(original.read_bytes()).hexdigest() == original_sha
            assert original.stat().st_mtime_ns == original_stat.st_mtime_ns
        finally:
            server.terminate()
            server.wait(timeout=10)
