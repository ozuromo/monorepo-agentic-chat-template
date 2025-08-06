def main():
    import subprocess
    from pathlib import Path

    script_dir = Path(__file__).parent
    app_file = script_dir.parent.parent / "src/frontend/streamlit_app.py"
    subprocess.run(["uv", "run", "streamlit", "run", str(app_file)])


if __name__ == "__main__":
    main()
