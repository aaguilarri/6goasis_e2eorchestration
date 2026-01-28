import ctypes
import os
import subprocess
import time

# Load libc
libc = ctypes.CDLL("libc.so.6", use_errno=True)

# Constants
CLONE_NEWNET = 0x40000000

# Define setns() prototype
setns = libc.setns
setns.argtypes = [ctypes.c_int, ctypes.c_int]
setns.restype = ctypes.c_int

class NamespaceSwitcher:
    def __init__(self):
        self.current_namespace = "default"
        
    def switch_to_namespace(self, namespace_name):
        """Switch to a specific network namespace"""
        if namespace_name == "default":
            return self.switch_to_default()
            
        ns_path = f"/var/run/netns/{namespace_name}"
        print(f"Switching to namespace: {namespace_name}")
        
        try:
            fd = os.open(ns_path, os.O_RDONLY)
            
            if setns(fd, CLONE_NEWNET) != 0:
                errno = ctypes.get_errno()
                raise OSError(errno, os.strerror(errno))
            
            os.close(fd)
            self.current_namespace = namespace_name
            print(f"Successfully switched to namespace: {namespace_name}")
            return True
            
        except FileNotFoundError:
            print(f"Namespace '{namespace_name}' does not exist.")
            return False
        except Exception as e:
            print(f"Failed to switch to namespace '{namespace_name}': {e}")
            return False
    
    def switch_to_default(self):
        """Switch back to the default namespace"""
        try:
            # Open the default namespace (usually /proc/1/ns/net)
            fd = os.open("/proc/1/ns/net", os.O_RDONLY)
            
            if setns(fd, CLONE_NEWNET) != 0:
                errno = ctypes.get_errno()
                raise OSError(errno, os.strerror(errno))
            
            os.close(fd)
            self.current_namespace = "default"
            print("Successfully switched to default namespace")
            return True
            
        except Exception as e:
            print(f"Failed to switch to default namespace: {e}")
            return False
    
    def get_current_interfaces(self):
        """Get current network interfaces in the active namespace"""
        print(f"\n=== Interfaces in '{self.current_namespace}' namespace ===")
        try:
            result = subprocess.run(["ip", "addr", "show"], 
                                  capture_output=True, text=True, check=True)
            # Show only interface names for brevity
            lines = result.stdout.split('\n')
            interfaces = [line.split(':')[1].strip() for line in lines 
                         if line.startswith((' ', '\t')) == False and ':' in line]
            print("Interfaces:", ', '.join(interfaces))
        except subprocess.CalledProcessError as e:
            print(f"Failed to get interfaces: {e}")
    
    def perform_task_in_namespace(self, namespace_name, task_description):
        """Perform a task in a specific namespace"""
        print(f"\n--- Performing task: {task_description} ---")
        
        if self.switch_to_namespace(namespace_name):
            self.get_current_interfaces()
            
            # Simulate some work
            print(f"Executing task in {namespace_name}...")
            time.sleep(1)  # Simulate work
            
            # You can add your actual task logic here
            # For example: network requests, socket operations, etc.
            
            print(f"Task completed in {namespace_name}")
        else:
            print(f"Could not perform task - failed to switch to {namespace_name}")

# Example usage
def main():
    switcher = NamespaceSwitcher()
    
    # Show initial state
    print("=== Initial State ===")
    switcher.get_current_interfaces()
    
    # Perform tasks in different namespaces
    switcher.perform_task_in_namespace("ue1", "Check network connectivity in UE1")
    switcher.perform_task_in_namespace("ue2", "Configure routing in UE2")
    switcher.perform_task_in_namespace("default", "Update system routing table")
    
    # Switch between namespaces multiple times
    print(f"\n=== Dynamic Switching Demo ===")
    namespaces = ["ue1", "ue2", "default", "ue1"]
    
    for ns in namespaces:
        switcher.switch_to_namespace(ns)
        switcher.get_current_interfaces()
        time.sleep(0.5)
    
    print("\nDemo completed!")

if __name__ == "__main__":
    main()
