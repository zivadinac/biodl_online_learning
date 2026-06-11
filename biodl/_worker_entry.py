"""Standalone entry point for the NEST worker process.

Run as ``python -m biodl._worker_entry`` by ``biodl.decoder._NestWorker``. Using a
dedicated module (rather than a ``multiprocessing`` spawn target) means the child's
``__main__`` is THIS file, not the user's script -- so the worker never re-imports
and re-runs an example's module-level GUI/bracelet/LSL setup. It connects back to
the parent over a localhost socket and serves the NEST request loop.
"""

import os
import sys
from multiprocessing.connection import Client


def main() -> None:
    addr = os.environ["BIODL_WORKER_ADDR"]
    host, port = addr.rsplit(":", 1)
    authkey = os.environ["BIODL_WORKER_AUTHKEY"].encode("latin1")
    conn = Client((host, int(port)), authkey=authkey)

    from biodl.decoder import _worker_serve
    _worker_serve(conn)


if __name__ == "__main__":
    main()
