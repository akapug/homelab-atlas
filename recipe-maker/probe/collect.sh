#!/usr/bin/env bash
# Read-only hardware collector for probe_box.py. Prints "@@<section>" headers, each followed by raw command output.
# Runs locally or over `ssh <host> bash -s`. Installs nothing, writes nothing. A section that cannot be read prints
# "@@!<section> <reason>" instead, so the parser can tag the field UNKNOWN with that reason.
# Privileged reads only ever use `sudo -n` (never prompt); without passwordless sudo they report "no sudo".
set +e
export LC_ALL=C
unset -f grep awk sed 2>/dev/null  # a caller's exported shell functions must not replace the system tools
have() { command -v "$1" >/dev/null 2>&1; }
SUDO=""
if [ "$(id -u)" = 0 ]; then SUDO=" "; elif sudo -n true 2>/dev/null; then SUDO="sudo -n"; fi

echo "@@meta"
echo "hostname=$(hostname 2>/dev/null)"
echo "time=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "kernel=$(uname -r)"
echo "sudo=$([ -n "$SUDO" ] && echo yes || echo no)"
# the mount holding the probing user's home ("~/..." recipe paths resolve to it); the path itself is not printed
echo "home_mount=$(findmnt -n -o TARGET -T "$HOME" 2>/dev/null)"

echo "@@cpuinfo"
# topology for every CPU, model + flags of the first one only (flags repeat per CPU)
grep -E '^(processor|physical id|core id)[[:space:]]*:' /proc/cpuinfo
awk -F: '/^model name/ && !m {print "model name:" $2; m=1} /^flags/ && !f {print "flags:" $2; f=1}' /proc/cpuinfo
echo "online: $(cat /sys/devices/system/cpu/online 2>/dev/null)"
# CPUs a user process may run on (a cgroup cpuset such as a reserved user.slice shows here)
grep -E '^Cpus_allowed_list:' /proc/self/status

echo "@@nodes"
if [ -d /sys/devices/system/node/node0 ]; then
  for n in /sys/devices/system/node/node[0-9]*; do
    echo "node=${n##*node} cpus=$(cat "$n/cpulist" 2>/dev/null)"
    grep -E 'MemTotal|MemFree' "$n/meminfo" 2>/dev/null
  done
else
  echo "@@!nodes /sys/devices/system/node absent (kernel without NUMA sysfs)"
fi

echo "@@meminfo"
grep -E '^(MemTotal|MemFree|MemAvailable|SwapTotal|SwapFree):' /proc/meminfo

echo "@@sysctl"
rd() { if [ -r "$2" ]; then echo "$1=$(cat "$2" 2>/dev/null)"; else echo "$1=!unreadable $2"; fi; }
rd numa_balancing /proc/sys/kernel/numa_balancing
rd demotion_enabled /sys/kernel/mm/numa/demotion_enabled
rd thp_enabled /sys/kernel/mm/transparent_hugepage/enabled
rd thp_defrag /sys/kernel/mm/transparent_hugepage/defrag
rd swappiness /proc/sys/vm/swappiness

echo "@@swaps"
cat /proc/swaps

echo "@@dax"
# devdax / kmem devices: which NUMA node each onlines as system RAM, and the region (nd = NVDIMM, cxl = CXL) behind it
for d in /sys/bus/dax/devices/*; do
  [ -e "$d" ] || continue
  echo "dev=${d##*/} target_node=$(cat "$d/target_node" 2>/dev/null) size=$(cat "$d/size" 2>/dev/null) driver=$(basename "$(readlink "$d/driver" 2>/dev/null)" 2>/dev/null) path=$(readlink -f "$d")"
done

echo "@@nd"
for r in /sys/bus/nd/devices/region*; do
  [ -e "$r" ] || continue
  echo "region=${r##*/} devtype=$(cat "$r/devtype" 2>/dev/null) size=$(cat "$r/size" 2>/dev/null) numa_node=$(cat "$r/numa_node" 2>/dev/null) target_node=$(cat "$r/target_node" 2>/dev/null) mappings=$(cat "$r/mappings" 2>/dev/null) persistence=$(cat "$r/persistence_domain" 2>/dev/null)"
done
for m in /sys/bus/nd/devices/nmem*; do
  [ -e "$m" ] || continue
  echo "nmem=${m##*/} vendor=$(cat "$m/nfit/vendor" 2>/dev/null) device=$(cat "$m/nfit/device" 2>/dev/null)"
done
[ -d /sys/bus/cxl/devices ] && echo "cxl_devices=$(ls /sys/bus/cxl/devices 2>/dev/null | tr '\n' ' ')"

echo "@@dmidecode"
if [ -n "$SUDO" ] && have dmidecode; then
  $SUDO dmidecode -t memory 2>&1 | grep -E '^Handle|^Memory Device|^[[:space:]]+(Size|Locator|Bank Locator|Type|Speed|Configured Memory Speed|Configured Clock Speed|Memory Technology|Rank|Number Of Devices|Maximum Capacity|Form Factor):'
elif ! have dmidecode; then
  echo "@@!dmidecode dmidecode not installed"
else
  echo "@@!dmidecode needs root: no passwordless sudo (sudo -n failed)"
fi

echo "@@nvidia"
if have nvidia-smi; then
  nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.used,memory.free,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,driver_version --format=csv,noheader,nounits 2>&1
else
  echo "@@!nvidia nvidia-smi not installed"
fi

echo "@@xpu"
if have xpu-smi; then
  ids=$(timeout 20 xpu-smi discovery -j 2>/dev/null | grep -oE '"device_id": *[0-9]+' | grep -oE '[0-9]+$')
  for i in $ids; do
    echo "@@xpu_dev $i"
    timeout 20 xpu-smi discovery -d "$i" -j 2>/dev/null
  done
else
  echo "@@!xpu xpu-smi not installed"
fi

echo "@@drm"
# every DRM card: PCI path, vendor, driver, endpoint link and the root port's link (a card behind an on-board PCIe
# switch, like Intel Arc Pro, reports a virtual 2.5 GT/s x1 link at the endpoint; the slot link is at the root port)
for c in /sys/class/drm/card[0-9]*; do
  case "${c##*/}" in *-*) continue;; esac
  d=$(readlink -f "$c/device")
  [ -e "$d/vendor" ] || continue
  rp=$(echo "$d" | awk -F/ '{for (i=1;i<=NF;i++) if ($i ~ /^pci[0-9a-f]+:/) {print $(i+1); exit}}')
  echo "card=${c##*/} pci=${d##*/} path=$d vendor=$(cat "$d/vendor") device=$(cat "$d/device") class=$(cat "$d/class") driver=$(basename "$(readlink "$d/driver" 2>/dev/null)" 2>/dev/null) boot_vga=$(cat "$d/boot_vga" 2>/dev/null) vram_total=$(cat "$d/mem_info_vram_total" 2>/dev/null) vram_used=$(cat "$d/mem_info_vram_used" 2>/dev/null) vram_vendor=$(cat "$d/mem_info_vram_vendor" 2>/dev/null) ep_speed=$(cat "$d/current_link_speed" 2>/dev/null | tr ' ' '_') ep_width=$(cat "$d/current_link_width" 2>/dev/null) ep_max_speed=$(cat "$d/max_link_speed" 2>/dev/null | tr ' ' '_') ep_max_width=$(cat "$d/max_link_width" 2>/dev/null) rootport=$rp rp_speed=$(cat "/sys/bus/pci/devices/$rp/current_link_speed" 2>/dev/null | tr ' ' '_') rp_width=$(cat "/sys/bus/pci/devices/$rp/current_link_width" 2>/dev/null) rp_max_speed=$(cat "/sys/bus/pci/devices/$rp/max_link_speed" 2>/dev/null | tr ' ' '_') rp_max_width=$(cat "/sys/bus/pci/devices/$rp/max_link_width" 2>/dev/null)"
done

echo "@@mounts"
findmnt -rn -b -o TARGET,SOURCE,FSTYPE,SIZE,AVAIL,UUID 2>&1

echo "@@lsblk"
lsblk -J -b -o NAME,KNAME,PATH,SIZE,TYPE,ROTA,TRAN,MODEL,FSTYPE,UUID,MOUNTPOINT 2>&1

echo "@@btrfs"
for f in /sys/fs/btrfs/*/devices; do
  [ -d "$f" ] || continue
  u=${f%/devices}; echo "uuid=${u##*/} members=$(ls "$f" | tr '\n' ' ')"
done

echo "@@loop"
for l in /sys/block/loop*/loop/backing_file; do
  [ -r "$l" ] || continue
  b=${l%/loop/backing_file}; echo "dev=${b##*/} backing=$(cat "$l")"
done
echo "@@end"
