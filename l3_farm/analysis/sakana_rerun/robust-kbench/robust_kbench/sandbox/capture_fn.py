from contextlib import contextmanager
import sys
import io


@contextmanager
def capture_output():
    # Capture Python stdout/stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = stdout_capture, stderr_capture

    # Store original subprocess.Popen
    import subprocess

    old_popen = subprocess.Popen

    # Collect subprocess output
    subprocess_output = []

    def new_popen(*args, **kwargs):
        # Ensure we have pipes for stdout and stderr
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
        process = old_popen(*args, **kwargs)

        # Capture and store output
        out, err = process.communicate()
        if out:
            subprocess_output.append(out.decode())
        if err:
            subprocess_output.append(err.decode())

        return process

    subprocess.Popen = new_popen

    try:
        yield stdout_capture, stderr_capture, subprocess_output
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        subprocess.Popen = old_popen
