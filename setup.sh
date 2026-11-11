#!/usr/bin/env bash
# Install the Python dependencies required by FastWave.AXERA.
set -e
python3 -m pip install --upgrade numpy
# axengine (pyaxengine) is preinstalled on AXera BSP images; install it if missing:
python3 -c "import axengine" 2>/dev/null || {
  echo "axengine not found. Install the axengine wheel shipped with your AX650 BSP, e.g.:"
  echo "  pip3 install --no-index --no-deps axengine-0.1.3-py3-none-any.whl"
}
