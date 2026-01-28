import ctypes
import os
import threading
import subprocess

# Load libc
libc = ctypes.CDLL("libc.so.6", use_errno=True)

# Constants
CLONE_NEWNET = 0x40000000  # from sched.h

# Define setns() prototype
setns = libc.setns
setns.argtypes = [ctypes.c_int, ctypes.c_int]
setns.restype = ctypes.c_int

def enter_netns_and_run(namespace_name):
    ns_path = f"/var/run/netns/{namespace_name}"
    print(ns_path)
    try:
        fd = os.open(ns_path, os.O_RDONLY)
    except FileNotFoundError:
        print(f"Namespace '{namespace_name}' does not exist.")
        return

    # Perform setns
    if setns(fd, CLONE_NEWNET) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))

    os.close(fd)

    # Print network interfaces inside the namespace
    print(f"[Thread] Now in namespace '{namespace_name}':")
    subprocess.run(["ip", "addr"], check=True)

# Main thread stays in the default namespace
print("[Main] Interfaces in the default namespace:")
subprocess.run(["ip", "addr"], check=True)

# Start a new thread in a different netns
thread = threading.Thread(target=enter_netns_and_run, args=("ue1",))
thread.start()
thread = threading.Thread(target=enter_netns_and_run, args=("ue2",))
thread.start()
thread.join()
