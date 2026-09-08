import sys


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--diagnose-runtime":
        from photo_sorter.runtime_diagnostics import write_diagnostics
        from pathlib import Path
        sys.exit(write_diagnostics(Path(sys.argv[2])))
    elif len(sys.argv) == 3 and sys.argv[1] == "--diagnose-gpu":
        from photo_sorter.gpu_diagnostics import write_diagnostics
        from pathlib import Path
        sys.exit(write_diagnostics(Path(sys.argv[2])))
    else:
        if getattr(sys, "frozen", False):
            from photo_sorter.runtime_terms import accept_runtime_terms
            from photo_sorter.app_paths import application_dir
            if not accept_runtime_terms(application_dir()):
                sys.exit(0)
        from photo_sorter.gui import main
        main()
