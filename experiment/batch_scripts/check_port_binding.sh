#!/bin/bash
# Script to check what's binding to port 8000 and related addresses

echo "=== Checking port 8000 bindings ==="
echo ""

echo "1. All processes listening on port 8000:"
sudo lsof -i :8000 2>/dev/null || lsof -i :8000 2>/dev/null || echo "  (lsof not available or no permissions)"
echo ""

echo "2. Network connections on port 8000:"
netstat -tulpn 2>/dev/null | grep :8000 || netstat -tuln 2>/dev/null | grep :8000 || echo "  (netstat not available)"
echo ""

echo "3. Socket statistics for port 8000:"
ss -tulpn 2>/dev/null | grep :8000 || ss -tuln 2>/dev/null | grep :8000 || echo "  (ss not available)"
echo ""

echo "4. All listening ports (checking for 8000):"
netstat -tuln 2>/dev/null | grep LISTEN | grep 8000 || ss -tuln 2>/dev/null | grep LISTEN | grep 8000 || echo "  (no process found on 8000)"
echo ""

echo "5. Hostname and IP addresses:"
echo "  Hostname: $(hostname)"
echo "  FQDN: $(hostname -f 2>/dev/null || hostname)"
echo "  IP addresses: $(hostname -I 2>/dev/null || hostname -i 2>/dev/null || echo 'unknown')"
echo ""

echo "6. Checking for vLLM processes:"
ps aux | grep -E "vllm|python.*serve" | grep -v grep || echo "  (no vLLM processes found)"
echo ""

echo "7. Checking for any Python processes on port 8000:"
for pid in $(lsof -ti :8000 2>/dev/null); do
  echo "  PID $pid: $(ps -p $pid -o cmd= 2>/dev/null || echo 'process not found')"
done
echo ""

echo "8. Testing if port 8000 is actually accessible:"
timeout 1 bash -c "echo > /dev/tcp/$(hostname -I | awk '{print $1}')/8000" 2>/dev/null && echo "  Port 8000 is OPEN" || echo "  Port 8000 is CLOSED or unreachable"
echo ""

echo "9. Checking socket states (TIME_WAIT, etc.):"
netstat -an 2>/dev/null | grep :8000 | head -10 || ss -an 2>/dev/null | grep :8000 | head -10 || echo "  (no sockets found)"
echo ""

