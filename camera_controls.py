"""Identity-checked Windows UVC controls; no automatic value substitution."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

CONTROL_IDS = {
    "video.brightness": 0, "video.contrast": 1, "video.hue": 2,
    "video.saturation": 3, "video.sharpness": 4, "video.gamma": 5,
    "video.white_balance": 7, "video.backlight": 8, "video.gain": 9,
    "camera.roll": 2, "camera.zoom": 3, "camera.exposure": 4, "camera.focus": 6,
}


def control_shell() -> str | None:
    """Use the installed Windows host, independent of terminal PATH order.

    The Store-packaged pwsh host on this machine rejects the camera's
    DirectShow BindToObject call, while Windows PowerShell succeeds.
    """
    windows_shell = (Path(os.environ.get("SystemRoot", r"C:\Windows")) /
                     "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe")
    if windows_shell.is_file():
        return str(windows_shell)
    return shutil.which("powershell") or shutil.which("pwsh")


def configure_controls(index: int, expected_id: str, fixed: dict[str, int] | None = None,
                       automatic: dict[str, int] | None = None):
    if index < 0:
        raise ValueError("camera index must be nonnegative")
    shell = control_shell()
    if not shell:
        raise RuntimeError("PowerShell is required for fixed UVC controls")
    requests = []
    if set(fixed or {}) & set(automatic or {}):
        raise ValueError("a camera control cannot be both manual and automatic")
    for name, value in {**(fixed or {}), **(automatic or {})}.items():
        if name not in CONTROL_IDS or not isinstance(value, int):
            raise ValueError(f"invalid camera control {name}: {value}")
        requests.append({"Kind": name.split(".")[0], "Property": CONTROL_IDS[name],
                         "Value": value, "Flags": 1 if name in (automatic or {}) else 2})
    source = str(Path(__file__).with_name("calibration") / "DirectShowModes.cs").replace("'", "''")
    identity = expected_id.replace("'", "''")
    payload = json.dumps(requests).replace("'", "''")
    command = (
        f"Add-Type -Path '{source}'; "
        f"$settings = '{payload}' | ConvertFrom-Json; "
        "$controls = @(foreach ($setting in $settings) { "
        "$request = [CameraDiagnostics.ControlRequest]::new(); "
        "$request.Kind=$setting.Kind; $request.Property=$setting.Property; "
        "$request.Value=$setting.Value; $request.Flags=$setting.Flags; $request }); "
        f"[CameraDiagnostics.Modes]::Controls({index},'{identity}',[CameraDiagnostics.ControlRequest[]]$controls) | ConvertTo-Json -Depth 5 -Compress"
    )
    result = subprocess.run([shell, "-NoProfile", "-Command", command], capture_output=True,
                            encoding="utf-8", errors="replace", timeout=30)
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"cannot verify/fix controls for {expected_id}: {result.stderr}")
    # PowerShell can print non-terminating invocation errors with exit code zero.
    if result.stderr.strip():
        raise RuntimeError(f"camera control operation failed: {result.stderr}")
    state = json.loads(result.stdout)
    actual = {f"{item['Kind']}.{item['Name']}": item for item in state["Controls"]}
    for name, value in (fixed or {}).items():
        if name not in actual or actual[name]["Value"] != value or actual[name]["Flags"] != 2:
            raise RuntimeError(f"camera did not retain manual control {name}={value}")
    for name in (automatic or {}):
        if name not in actual or actual[name]["Flags"] != 1:
            raise RuntimeError(f"camera did not retain automatic mode for {name}")
    return state
