import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from dotenv import load_dotenv
from utils.config_lodder import Config
from core.evaluator import evaluate_prompts

load_dotenv()
cfg = Config(os.getenv("CONFIG_PATH")).as_dict()


class ChangeHandler(FileSystemEventHandler):
    def on_any_event(self, event):
        if event.event_type in ["modified", "created"] and not event.is_directory:
            print(f"[📁] Detected change: {event.src_path}")
            evaluate_prompts()


def start_watching():
    print("[👀] Watching for changes in:", cfg["watch_dirs"])
    observer = Observer()
    handler = ChangeHandler()

    for path in cfg["watch_dirs"]:
        if os.path.exists(path):
            observer.schedule(handler, path=path, recursive=True)
        else:
            print(f"[⚠️] Watch path does not exist: {path}")

    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
