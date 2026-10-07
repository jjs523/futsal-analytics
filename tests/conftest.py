import os

os.environ.setdefault("FUTSAL_NO_AUTOAPP", "1")    # importing server.app.main must not start a server-wide worker
