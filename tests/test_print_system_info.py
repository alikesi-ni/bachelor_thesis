import platform
import cpuinfo
import psutil
import subprocess

def get_installed_ram():
    try:
        output = subprocess.check_output(
            ["wmic", "MemoryChip", "get", "Capacity"],
            universal_newlines=True
        )
        lines = output.strip().split("\n")[1:]
        total_bytes = sum(int(line.strip()) for line in lines if line.strip().isdigit())
        return round(total_bytes / (1024 ** 3), 2)
    except Exception:
        return "Unknown"

def get_os_version():
    return f"{platform.system()} {platform.release()}"

def print_machine_spec():
    cpu_info = cpuinfo.get_cpu_info()
    cpu_name = cpu_info['brand_raw']
    cores = psutil.cpu_count(logical=False)
    threads = psutil.cpu_count(logical=True)
    freq = psutil.cpu_freq()
    freq_max = round(freq.max, 0)  # in MHz
    installed_ram = get_installed_ram()
    usable_ram = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    arch = platform.machine()
    os_version = get_os_version()
    python_version = platform.python_version()
    interpreter = platform.python_implementation()

    print("Machine Specifications:\n")
    print(f"CPU: {cpu_name} ({cores} cores / {threads} threads)")
    print(f"Max CPU frequency: {int(freq_max)} MHz")
    print(f"Installed RAM: {installed_ram} GB")
    print(f"Usable RAM: ~{usable_ram} GB")
    print(f"Architecture: {arch}")
    print(f"Operating System: {os_version}")
    print(f"Python Version: {python_version}")
    print(f"Interpreter: {interpreter}")

print_machine_spec()

