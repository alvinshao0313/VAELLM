"""Measure uploads from this machine to an SSH server using real scp and SHA256."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
import uuid


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def remote_setup(parent):
    root = '"$HOME"' if parent is None else shlex.quote(parent)
    return f'''set -eu
bench_root={root}
test -d "$bench_root"
test -w "$bench_root"
command -v sha256sum >/dev/null
bench_dir=$(mktemp -d -- "${{bench_root%/}}/vaellm_scp_bench.XXXXXXXX")
printf 'VAELLM_SCP_BENCH_DIR=%s\\n' "$bench_dir"
'''


def run_logged(argv, log, *, timeout):
    with log.open('a', encoding='utf-8', errors='backslashreplace') as stream:
        stream.write('$ ' + shlex.join(argv) + '\n')
        stream.flush()
        # A persistent SSH master can keep inherited pipes open after its
        # command exits. Regular temporary files let us wait for the command.
        with tempfile.TemporaryFile(mode='w+', encoding='utf-8', errors='backslashreplace') as stdout, tempfile.TemporaryFile(mode='w+', encoding='utf-8', errors='backslashreplace') as stderr:
            try:
                result = subprocess.run(argv, stdout=stdout, stderr=stderr, timeout=timeout)
            finally:
                stdout.seek(0)
                stderr.seek(0)
                out, error = stdout.read(), stderr.read()
                stream.write(out + error)
                stream.flush()
    if result.returncode:
        raise RuntimeError(f'命令退出{result.returncode}：{shlex.join(argv)}\n{error.strip()}')
    return out


def create_payload(args, directory):
    if args.file:
        return args.file.resolve()
    payload = Path(directory) / 'payload.bin'
    remaining = (args.size_mib or 1024) * 1024 * 1024
    with payload.open('xb') as stream:
        while remaining:
            block = os.urandom(min(remaining, 8 * 1024 * 1024))
            stream.write(block)
            remaining -= len(block)
        stream.flush()
        os.fsync(stream.fileno())
    return payload


def summarize(args, report, size):
    total_seconds = sum(row['seconds'] for row in report['rounds'])
    bytes_per_second = size * args.rounds / total_seconds
    report['summary'] = {'total_uploaded_bytes': size * args.rounds,
                         'total_transfer_seconds': total_seconds,
                         'aggregate_MiB_per_second': bytes_per_second / 1024**2,
                         'median_MiB_per_second': statistics.median(row['MiB_per_second'] for row in report['rounds']),
                         'aggregate_Mbps': bytes_per_second * 8 / 1e6}
    if args.estimate_gib:
        report['summary']['estimate'] = {'GiB': args.estimate_gib,
                                        'hours': args.estimate_gib * 1024**3 / bytes_per_second / 3600,
                                        'scope': '按本次大文件吞吐估算，不含大量小文件/校验/断点重传开销。'}


def benchmark_sftp(args, report, output):
    """Select a gateway resource, then use real scp -s without remote exec."""
    log = output / 'commands.log'
    with tempfile.TemporaryDirectory(prefix='vaellm_sftp_') as control_directory, tempfile.TemporaryDirectory(
            prefix='vaellm_sftp_payload_', dir=args.local_dir) as local_directory:
        control_path = str(Path(control_directory) / 'master')
        options = ['-o', 'Compression=no', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=15',
                   '-o', 'ServerAliveCountMax=3', '-o', 'ControlPath=' + control_path]
        if args.ssh_config:
            options += ['-F', str(args.ssh_config.resolve())]
        if args.identity:
            options += ['-i', str(args.identity.resolve())]
        ssh = ['ssh', *options]
        sftp = ['sftp', *options, '-o', 'BatchMode=yes']
        scp = ['scp', '-s', *options, '-o', 'BatchMode=yes']
        if args.port:
            ssh += ['-p', str(args.port)]
            sftp += ['-P', str(args.port)]
            scp += ['-P', str(args.port)]
        directory = 'vaellm_scp_bench.' + uuid.uuid4().hex
        remote_file = directory + '/payload.bin'
        report.update({'transport': 'SCP over gateway SFTP', 'scp_command_used': True,
                       'gateway_directory': args.sftp_root, 'remote_encoding': args.remote_encoding,
                       'remote_temporary_directory_relative_to_selected_home': directory,
                       'compression': False, 'ssh_setup_included_in_timing': False,
                       'hashing_included_in_timing': False,
                       'checksum_method': 'SFTP回读后本机SHA256，回读耗时不计入上传吞吐；额外下载同等数据量。'})
        created = False
        selected_parent = None

        def quoted(path, encoding):
            value = str(path)
            if any(c in value for c in '\r\n\x00'):
                raise ValueError('SFTP路径不能包含换行或NUL')
            return b'"' + value.encode(encoding).replace(b'\\', b'\\\\').replace(b'"', b'\\"') + b'"'

        def batch(commands):
            # GateShell keeps the resource selection on the shared SSH
            # connection. Later channels must use the selected node's path.
            if selected_parent is None:
                prefix = [b'cd ' + quoted(args.sftp_root, args.remote_encoding)]
                if args.remote_dir is not None:
                    prefix.append(b'cd ' + quoted(args.remote_dir, args.remote_encoding))
            else:
                prefix = [b'cd ' + quoted(selected_parent, args.remote_encoding)]
            batch_path = Path(control_directory) / 'commands.sftp'
            batch_path.write_bytes(b'\n'.join([*prefix, *commands]) + b'\n')
            return run_logged([*sftp, '-b', str(batch_path), args.host], log, timeout=args.timeout)

        try:
            # No shell/exec channel is requested; the gateway may only allow SFTP.
            run_logged([*ssh, '-M', '-N', '-f', '-o', 'ControlPersist=300', args.host], log, timeout=args.timeout)
            setup = batch([b'pwd', b'mkdir ' + quoted(directory, 'ascii')])
            created = True
            parents = [line.removeprefix('Remote working directory: ') for line in setup.splitlines()
                       if line.startswith('Remote working directory: ')]
            if len(parents) != 1 or not re.fullmatch(r'/[A-Za-z0-9_./-]*', parents[0]):
                raise RuntimeError('未获取到可用于SCP的实际节点ASCII绝对目录；请查看commands.log')
            selected_parent = parents[0]
            report['remote_temporary_directory'] = selected_parent.rstrip('/') + '/' + directory
            payload = create_payload(args, local_directory)
            original = payload.stat()
            digest = sha256(payload)
            report.update({'bytes_per_round': original.st_size, 'source_sha256': digest,
                           'source': str(payload) if args.file else '独立随机数据文件（结束后删除）'})
            returned = Path(local_directory) / 'returned.bin'
            for round_id in range(1, args.rounds + 1):
                print(f'SCP/SFTP UPLOAD {round_id}/{args.rounds}: {original.st_size / 1024**2:.0f} MiB -> {args.host}', flush=True)
                started = time.perf_counter()
                destination = selected_parent.rstrip('/') + '/' + remote_file
                run_logged([*scp, str(payload), args.host + ':' + destination], log, timeout=args.timeout)
                seconds = time.perf_counter() - started
                batch([b'get ' + quoted(remote_file, 'ascii') + b' ' + quoted(returned, sys.getfilesystemencoding())])
                if returned.stat().st_size != original.st_size or sha256(returned) != digest:
                    raise RuntimeError('SFTP上传后的回读SHA256不一致')
                now = payload.stat()
                if (original.st_size, original.st_mtime_ns, original.st_ino) != (now.st_size, now.st_mtime_ns, now.st_ino):
                    raise RuntimeError('测速期间源文件发生变化')
                row = {'round': round_id, 'bytes': original.st_size, 'seconds': seconds,
                       'MiB_per_second': original.st_size / seconds / 1024**2,
                       'Mbps': original.st_size * 8 / seconds / 1e6, 'sha256_verified': True}
                report['rounds'].append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
                (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            summarize(args, report, original.st_size)
        finally:
            errors = []
            if created:
                for command in [b'rm ' + quoted(remote_file, 'ascii'), b'rmdir ' + quoted(directory, 'ascii')]:
                    try:
                        batch([command])
                    except Exception as exc:
                        errors.append(str(exc))
                report['remote_test_files_removed'] = not errors
            try:
                if Path(control_path).exists():
                    run_logged([*ssh, '-O', 'exit', args.host], log, timeout=15)
            except Exception as exc:
                errors.append(str(exc))
            report['cleanup_errors'] = errors
            if errors and sys.exc_info()[0] is None:
                raise RuntimeError('SFTP临时产物或专用连接清理失败：' + '; '.join(errors))


def benchmark(args, report, output):
    log = output / 'commands.log'
    with tempfile.TemporaryDirectory(prefix='vaellm_ssh_') as control_directory:
        control_path = str(Path(control_directory) / 'master')
        options = ['-o', 'Compression=no', '-o', 'ConnectTimeout=15',
                   '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3',
                   '-o', 'ControlPath=' + control_path]
        if args.ssh_config:
            options += ['-F', str(args.ssh_config.resolve())]
        if args.identity:
            options += ['-i', str(args.identity.resolve())]
        ssh = ['ssh', *options]
        scp = ['scp', *options]
        if args.port:
            ssh += ['-p', str(args.port)]
            scp += ['-P', str(args.port)]
        remote = None
        try:
            # The first call may ask for a password through /dev/tty; subsequent
            # commands reuse our own master connection and never store passwords.
            setup = run_logged([*ssh, '-o', 'ControlMaster=auto', '-o', 'ControlPersist=300',
                                args.host, remote_setup(args.remote_dir)], log, timeout=args.timeout)
            records = [line.removeprefix('VAELLM_SCP_BENCH_DIR=') for line in setup.splitlines()
                       if line.startswith('VAELLM_SCP_BENCH_DIR=')]
            if len(records) != 1:
                raise RuntimeError('无法解析远端mktemp返回的独立目录；请查看commands.log')
            candidate = records[0]
            if not re.fullmatch(r'/[A-Za-z0-9_./-]+/vaellm_scp_bench\.[A-Za-z0-9]{8}', candidate):
                raise RuntimeError('SCP兼容OpenSSH8/9需要不含空格或shell特殊字符的绝对接收路径')
            remote = candidate
            report['remote_temporary_directory'] = remote
            destination = remote + '/payload.bin'
            batch = ['-o', 'BatchMode=yes', '-o', 'ControlMaster=no']
            with tempfile.TemporaryDirectory(prefix='vaellm_scp_payload_', dir=args.local_dir) as local_directory:
                payload = create_payload(args, local_directory)
                before = payload.stat()
                digest = sha256(payload)
                report.update({'bytes_per_round': before.st_size, 'source_sha256': digest,
                               'source': str(payload) if args.file else '独立随机数据文件（结束后删除）',
                               'compression': False, 'ssh_setup_included_in_timing': False,
                               'hashing_included_in_timing': False})
                for round_id in range(1, args.rounds + 1):
                    print(f'UPLOAD {round_id}/{args.rounds}: {before.st_size / 1024**2:.0f} MiB -> {args.host}', flush=True)
                    started = time.perf_counter()
                    run_logged([*scp, *batch, str(payload), args.host + ':' + destination], log, timeout=args.timeout)
                    seconds = time.perf_counter() - started
                    if seconds <= 0:
                        raise RuntimeError('无效的SCP计时结果')
                    actual = run_logged([*ssh, *batch, args.host, 'sha256sum -- ' + shlex.quote(destination)],
                                        log, timeout=args.timeout).split()[0]
                    if actual != digest:
                        raise RuntimeError(f'第{round_id}轮SHA256不一致；不能把该轮记录为有效吞吐')
                    now = payload.stat()
                    if (before.st_size, before.st_mtime_ns, before.st_ino) != (now.st_size, now.st_mtime_ns, now.st_ino):
                        raise RuntimeError('测速期间源文件发生变化')
                    result = {'round': round_id, 'bytes': before.st_size, 'seconds': seconds,
                              'MiB_per_second': before.st_size / seconds / 1024**2,
                              'Mbps': before.st_size * 8 / seconds / 1e6, 'sha256_verified': True}
                    report['rounds'].append(result)
                    print(json.dumps(result, ensure_ascii=False), flush=True)
                    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
                summarize(args, report, before.st_size)
        finally:
            cleanup_errors = []
            if remote is not None:
                cleanup = 'rm -f -- ' + shlex.quote(remote + '/payload.bin') + '\nrmdir -- ' + shlex.quote(remote)
                try:
                    run_logged([*ssh, '-o', 'BatchMode=yes', args.host, cleanup], log, timeout=30)
                    report['remote_test_files_removed'] = True
                except Exception as exc:
                    report['remote_test_files_removed'] = False
                    cleanup_errors.append(str(exc))
            try:
                # ControlPath is unique to this invocation; other SSH sessions
                # and the remote server's own sshd are never stopped.
                if Path(control_path).exists():
                    run_logged([*ssh, '-O', 'exit', args.host], log, timeout=15)
            except Exception as exc:
                cleanup_errors.append(str(exc))
            report['cleanup_errors'] = cleanup_errors
            if cleanup_errors and sys.exc_info()[0] is None:
                raise RuntimeError('测速已结束，但临时文件或专用SSH连接清理失败：' + '; '.join(cleanup_errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True, help='SSH别名或user@host；IPv6请配置SSH别名')
    parser.add_argument('--port', type=int, help='省略时沿用SSH配置；本次新机器为60022')
    parser.add_argument('--remote-dir', help='已存在的可写绝对目录，默认新机器登录用户的家目录；仅支持ASCII无空格路径')
    parser.add_argument('--sftp-root', help='经堡垒机资源目录使用scp -s；不使用远程exec，校验需SFTP回读')
    parser.add_argument('--remote-encoding', choices=('utf-8', 'gb18030'), default='utf-8', help='SFTP资源目录编码')
    parser.add_argument('--local-dir', type=Path, help='随机测试文件的本机临时父目录，默认系统临时目录')
    payload = parser.add_mutually_exclusive_group()
    payload.add_argument('--size-mib', type=int, help='随机测试文件MiB，默认1024')
    payload.add_argument('--file', type=Path, help='改用现有稳定大文件，只读；不复制或修改源文件')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--estimate-gib', type=float, help='需要迁移的GiB总量，用于粗估时间')
    parser.add_argument('--identity', type=Path, help='可选：SSH私钥路径，不读取或复制私钥正文')
    parser.add_argument('--ssh-config', type=Path, help='可选：独立SSH配置，支持跳板机等设置')
    parser.add_argument('--timeout', type=int, default=1800, help='每次认证/SCP/哈希的超时秒数')
    parser.add_argument('--output', type=Path, required=True, help='尚不存在的结果目录')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@-]*', args.host):
        parser.error('--host必须是SSH别名或user@host，不接受选项、空格或远程路径')
    if args.port is not None and not 1 <= args.port <= 65535:
        parser.error('--port必须为1..65535')
    if args.rounds < 1 or args.timeout < 1 or (args.size_mib is not None and args.size_mib < 1):
        parser.error('--rounds、--timeout、--size-mib必须是正整数')
    if args.estimate_gib is not None and (not math.isfinite(args.estimate_gib) or args.estimate_gib <= 0):
        parser.error('--estimate-gib必须是有限正数')
    if args.remote_dir is not None and not re.fullmatch(r'/[A-Za-z0-9_./-]*', args.remote_dir):
        parser.error('--remote-dir必须是ASCII且不含空格的绝对路径')
    if args.sftp_root is not None and (not args.sftp_root.startswith('/') or any(c in args.sftp_root for c in '\r\n\x00')):
        parser.error('--sftp-root必须是无换行或NUL的绝对资源目录')
    for name in ('file', 'identity', 'ssh_config'):
        value = getattr(args, name)
        if value is not None and (not value.is_file() or value.stat().st_size == 0):
            parser.error('--' + name.replace('_', '-') + '必须是存在且非空的文件')
    if args.local_dir is not None and not args.local_dir.is_dir():
        parser.error('--local-dir不存在')
    if not shutil.which('ssh') or not shutil.which('scp'):
        parser.error('需要本机已有OpenSSH ssh和scp')
    if args.sftp_root and not shutil.which('sftp'):
        parser.error('堡垒机模式需要已有OpenSSH sftp')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'started': datetime.now(timezone.utc).isoformat(), 'host': args.host,
              'port': args.port, 'direction': '本机 -> 新机器', 'rounds': [], 'transport': 'SCP',
              'scope': '真实SCP大文件吞吐（含协议/加密/文件IO），不是裸网络带宽；不承诺远端fsync落盘速度。'}
    try:
        if args.sftp_root:
            benchmark_sftp(args, report, output)
        else:
            benchmark(args, report, output)
        report['status'] = 'passed'
    except (Exception, KeyboardInterrupt):
        report['status'] = 'failed'
        report['traceback'] = traceback.format_exc()
        print(report['traceback'], file=sys.stderr)
    finally:
        report['finished'] = datetime.now(timezone.utc).isoformat()
        (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'report': str(output / 'report.json'),
                      'summary': report.get('summary')}, ensure_ascii=False), flush=True)
    return int(report['status'] == 'failed')


if __name__ == '__main__':
    raise SystemExit(main())
