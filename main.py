# main.py

from core.watcher import start_watching

if __name__ == "__main__":
    try:
        start_watching()
    except Exception as e:
        print(f"[❌] Watcher failed: {e}")
    # start_watching()
