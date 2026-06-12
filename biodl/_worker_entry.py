"""Standalone entry point for the NEST worker process.

Run as ``python -m biodl._worker_entry`` by ``biodl.decoder._NestWorker``. Using a
dedicated module (rather than a ``multiprocessing`` spawn target) means the child's
``__main__`` is THIS file, not the user's script -- so the worker never re-imports
and re-runs an example's module-level GUI/bracelet/LSL setup. It connects back to
the parent over the configured socket or inherited pipe and serves the NEST
request loop.
"""

import os
from multiprocessing.connection import Client


def main() -> None:
    from biodl.decoder import _FdConnection, _worker_serve

    if "BIODL_WORKER_READ_FD" in os.environ:
        conn = _FdConnection(
            os.fdopen(int(os.environ["BIODL_WORKER_READ_FD"]), "rb", buffering=0),
            os.fdopen(int(os.environ["BIODL_WORKER_WRITE_FD"]), "wb", buffering=0),
        )
    else:
        authkey = bytes.fromhex(os.environ["BIODL_WORKER_AUTHKEY"])
        if "BIODL_WORKER_SOCKET" in os.environ:
            conn = Client(
                os.environ["BIODL_WORKER_SOCKET"], family="AF_UNIX", authkey=authkey
            )
        else:
            addr = os.environ["BIODL_WORKER_ADDR"]
            host, port = addr.rsplit(":", 1)
            conn = Client((host, int(port)), authkey=authkey)

    _worker_serve(conn)


if __name__ == "__main__":
    main()
