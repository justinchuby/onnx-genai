"""Containment checks use real local children, not a compiled CLI or installed wheel."""

import os
from pathlib import Path
import signal
import select
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import python_wheel_smoke as smoke


class ContainmentTests(unittest.TestCase):
    def test_real_subprocess_output_and_failure_are_checked(self):
        output = smoke.run_command([sys.executable, "-c", "print('actual-child')"], os.environ.copy())
        self.assertEqual(output.stdout.strip(), "actual-child")
        with self.assertRaisesRegex(RuntimeError, "actual-failure"):
            smoke.run_command([sys.executable, "-c", "import sys; sys.exit('actual-failure')"], os.environ.copy())

    @unittest.skipUnless(os.name == "posix", "POSIX process-group and creation-signal guarantee")
    def test_timeout_kills_descendant_that_ignores_termination_and_reaps_parent(self):
        read_end, write_end = os.pipe()
        try:
            with tempfile.TemporaryDirectory() as directory:
                pid_file = Path(directory) / "descendant.pid"
                child = "import os,signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);open(%r,'w').write(str(os.getpid()));time.sleep(60)" % str(pid_file)
                parent = "import os,subprocess,sys,time;subprocess.Popen([sys.executable,'-c',%r],pass_fds=(%d,));os.close(%d);time.sleep(60)" % (child, write_end, write_end)
                with self.assertRaises(subprocess.TimeoutExpired):
                    with smoke.contained_process([sys.executable, "-c", parent], stdout=subprocess.PIPE, stderr=subprocess.PIPE, pass_fds=(write_end,)) as process:
                        os.close(write_end)
                        write_end = None
                        deadline = time.monotonic() + 5
                        while not pid_file.exists() and time.monotonic() < deadline:
                            time.sleep(0.01)
                        self.assertTrue(pid_file.exists(), "actual descendant must start before testing timeout cleanup")
                        process.communicate(timeout=0.05)
                self.assertIsNotNone(process.poll(), "direct parent must be reaped")
                # Only the SIGTERM-ignoring descendant retains the writer. EOF
                # proves group cleanup terminated it, without process-table access.
                ready, _, _ = select.select([read_end], [], [], 2)
                self.assertEqual(ready, [read_end], "descendant retained its pipe after cleanup")
                self.assertEqual(os.read(read_end, 1), b"", "descendant must close its inherited pipe on termination")
        finally:
            os.close(read_end)
            if write_end is not None:
                os.close(write_end)

    @unittest.skipUnless(os.name == "posix", "POSIX creation-signal guarantee")
    def test_signal_during_creation_is_delivered_after_handle_ownership(self):
        previous = signal.getsignal(signal.SIGTERM)
        popen = subprocess.Popen
        created = []

        def interrupted_creation(*args, **kwargs):
            process = popen(*args, **kwargs)
            created.append(process)
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            return process

        with patch.object(smoke.subprocess, "Popen", interrupted_creation):
            with self.assertRaisesRegex(KeyboardInterrupt, "during child creation"):
                with smoke.contained_process([sys.executable, "-c", "import time;time.sleep(60)"]):
                    self.fail("pending cancellation must precede child use")
        self.assertEqual(len(created), 1)
        self.assertIsNotNone(created[0].poll(), "owned child must be reaped")
        self.assertIs(signal.getsignal(signal.SIGTERM), previous)


if __name__ == "__main__":
    unittest.main()
