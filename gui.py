# gui.py
import os
import threading
import subprocess
from typing import Callable, Dict, List, Optional, Tuple

# ---------- CONFIG ----------
BUILD_DIR = r".\build\Release"
OUTPUT_FILENAME = "out.png"

# Decoder -> executable file name (without path)
DECODERS: List[Tuple[str, str]] = [
    ("abs_val",        "abs_val.exe"),
    ("cosine",         "cosine.exe"),
    ("hilbertFIR",     "hilbertFIR.exe"),
    ("hilbertFFT",     "hilbertFFT.exe"),
    ("contrast",       "contrast.exe"),
    ("falsecolour",    "falsecolour.exe"),
    ("pseudocolour1",  "pseudocolour1.exe"),
    ("pseudocolour2",  "pseudocolour2.exe"),
]

FALSECOLOUR_MODES = ["manual", "pseudo"]
FALSECOLOUR_COLORS = [
    "gray", "jet", "ocean", "hot", "bone", "winter",
    "rainbow", "autumn", "summer", "spring", "cool",
    "hsv", "pink", "parula", "turbo"
]


def get_default_output_dir() -> str:
    return os.path.join(os.getcwd(), "output")


class DecoderCore:
    """
    Pure functional core. No GUI imports here.

    Exposes:
      - build_command(...) -> List[str]
      - run_async(cmd, on_stdout, on_complete) -> None
    """

    def __init__(
        self,
        build_dir: str = BUILD_DIR,
        output_filename: str = OUTPUT_FILENAME,
        decoders: List[Tuple[str, str]] = DECODERS,
        falsecolour_modes: List[str] = FALSECOLOUR_MODES,
        falsecolour_colors: List[str] = FALSECOLOUR_COLORS,
    ):
        self.build_dir = build_dir
        self.output_filename = output_filename
        self.decoders_map: Dict[str, str] = dict(decoders)
        self.falsecolour_modes = list(falsecolour_modes)
        self.falsecolour_colors = list(falsecolour_colors)

    def resolve_exe_path(self, decoder_key: str) -> str:
        if decoder_key not in self.decoders_map:
            raise ValueError(f"Unknown decoder '{decoder_key}'")
        exe_name = self.decoders_map[decoder_key]
        exe_path = os.path.join(self.build_dir, exe_name)
        if not os.path.isfile(exe_path):
            raise FileNotFoundError(f"Decoder not found at: {exe_path}")
        return exe_path

    def build_output_path(self, output_dir: Optional[str]) -> str:
        outdir = (output_dir or get_default_output_dir()).strip()
        os.makedirs(outdir, exist_ok=True)
        return os.path.join(outdir, self.output_filename)

    def build_command(
        self,
        decoder_key: str,
        wav_path: str,
        output_dir: Optional[str],
        width: str,
        mode: Optional[str] = None,
        color: Optional[str] = None,
    ) -> Tuple[List[str], str]:
        """
        Returns (cmd_list, out_png_path)
        """
        if not wav_path:
            raise ValueError("Missing input WAV path.")
        if not width.isdigit():
            raise ValueError("Width must be a positive integer string.")

        exe_path = self.resolve_exe_path(decoder_key)
        out_png = self.build_output_path(output_dir)

        # base: exe wav out width
        cmd = [exe_path, wav_path, out_png, width]

        # Pseudocolour variants: require mode="manual" and color="pseudo"
        if decoder_key in ("pseudocolour1", "pseudocolour2"):
            cmd.extend([mode or "manual", color or "pseudo"])

        # falsecolour: mode in {manual,pseudo}, color in whitelist
        elif decoder_key == "falsecolour":
            m = mode or self.falsecolour_modes[0]
            c = color or self.falsecolour_colors[0]
            if m not in self.falsecolour_modes:
                raise ValueError(f"Invalid mode for falsecolour: {m}")
            if c not in self.falsecolour_colors:
                raise ValueError(f"Invalid color for falsecolour: {c}")
            cmd.extend([m, c])

        return cmd, out_png

    def run_async(
        self,
        cmd: List[str],
        on_stdout: Callable[[str], None],
        on_complete: Callable[[Optional[Exception]], None],
    ) -> None:
        """
        Runs the process on a background thread.
        Calls on_stdout(line) for each output line.
        Calls on_complete(error_or_none) at the end.
        """
        def _worker():
            try:
                creationflags = 0
                if hasattr(subprocess, "CREATE_NO_WINDOW"):
                    creationflags = subprocess.CREATE_NO_WINDOW

                with subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    shell=False,
                    creationflags=creationflags,
                ) as p:
                    for line in p.stdout:
                        try:
                            on_stdout(line.rstrip("\n"))
                        except Exception:
                            # Don't let UI callback errors kill the process
                            pass
                    rc = p.wait()
                    on_stdout(f"\nProcess exited with code {rc}")
                on_complete(None)
            except Exception as e:
                try:
                    on_stdout(f"Error: {e}")
                except Exception:
                    pass
                on_complete(e)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
