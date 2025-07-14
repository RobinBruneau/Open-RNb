import os
import subprocess
import shutil
from utils.misc import dump_config, parse_version


import pytorch_lightning
if parse_version(pytorch_lightning.__version__) > parse_version('1.8'):
    from pytorch_lightning.callbacks import Callback
else:
    from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only, rank_zero_warn
from pytorch_lightning.callbacks.progress import TQDMProgressBar

# --- Helper function for pattern matching (similar to .gitignore) ---
# We'll need a simple way to match paths against patterns
import fnmatch

def matches_pattern(filepath, patterns):
    """
    Checks if a filepath matches any of the given patterns.
    Patterns can include wildcards like '*' or '**'.
    Supports basic .gitignore-like patterns.
    """
    for pattern in patterns:
        if pattern.endswith('/'): # If pattern ends with '/', it matches directories
            if filepath.startswith(pattern):
                return True
        elif fnmatch.fnmatch(filepath, pattern): # Match full file paths
            return True
        elif os.path.basename(filepath) == pattern: # Match just the filename
            return True
    return False


class VersionedCallback(Callback):
    def __init__(self, save_root, version=None, use_version=True):
        self.save_root = save_root
        self._version = version
        self.use_version = use_version

    @property
    def version(self) -> int:
        """Get the experiment version.

        Returns:
            The experiment version if specified else the next version.
        """
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    def _get_next_version(self):
        existing_versions = []
        if os.path.isdir(self.save_root):
            for f in os.listdir(self.save_root):
                bn = os.path.basename(f)
                if bn.startswith("version_"):
                    dir_ver = os.path.splitext(bn)[0].split("_")[1].replace("/", "")
                    existing_versions.append(int(dir_ver))
        if len(existing_versions) == 0:
            return 0
        return max(existing_versions) + 1
    
    @property
    def savedir(self):
        if not self.use_version:
            return self.save_root
        return os.path.join(self.save_root, self.version if isinstance(self.version, str) else f"version_{self.version}")


class CodeSnapshotCallback(VersionedCallback):
    def __init__(self, save_root, version=None, use_version=True, ignore_patterns=None):
        super().__init__(save_root, version, use_version)
        # Store the patterns to ignore
        self.ignore_patterns = ignore_patterns if ignore_patterns is not None else []
    
    def get_file_list(self):
        # Get all files tracked by git AND untracked files
        git_tracked_files = [b.decode() for b in subprocess.check_output('git ls-files', shell=True).splitlines()]
        git_untracked_files = [b.decode() for b in subprocess.check_output('git ls-files --others --exclude-standard', shell=True).splitlines()]
        
        all_files = set(git_tracked_files) | set(git_untracked_files)
        
        filtered_files = []
        for f in all_files:
            # Check if the file path matches any ignore pattern
            if matches_pattern(f, self.ignore_patterns):
                # print(f"Ignoring: {f}") # For debugging
                continue
            filtered_files.append(f)
            
        return filtered_files
    
    @rank_zero_only
    def save_code_snapshot(self):
        os.makedirs(self.savedir, exist_ok=True)
        # Iterate over the filtered list of files
        for f in self.get_file_list():
            if not os.path.exists(f) or os.path.isdir(f):
                continue
            
            # Ensure parent directories exist in the target snapshot folder
            target_file_path = os.path.join(self.savedir, f)
            os.makedirs(os.path.dirname(target_file_path), exist_ok=True)
            
            # Copy the file
            shutil.copyfile(f, target_file_path)
            
    def on_fit_start(self, trainer, pl_module):
        try:
            self.save_code_snapshot()
        except Exception as e: # Catch specific exception for better debugging if needed
            rank_zero_warn(f"Code snapshot is not saved. Error: {e}. Please make sure you have git installed and are in a git repository.")


class ConfigSnapshotCallback(VersionedCallback):
    def __init__(self, config, save_root, version=None, use_version=True):
        super().__init__(save_root, version, use_version)
        self.config = config

    @rank_zero_only
    def save_config_snapshot(self):
        os.makedirs(self.savedir, exist_ok=True)
        dump_config(os.path.join(self.savedir, 'parsed.yaml'), self.config)
        shutil.copyfile(self.config.cmd_args['config'], os.path.join(self.savedir, 'raw.yaml'))

    def on_fit_start(self, trainer, pl_module):
        self.save_config_snapshot()


class CustomProgressBar(TQDMProgressBar):
    def get_metrics(self, *args, **kwargs):
        # don't show the version number
        items = super().get_metrics(*args, **kwargs)
        items.pop("v_num", None)
        return items