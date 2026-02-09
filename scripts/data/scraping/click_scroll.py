"""Click N points on screen, then auto-scroll down at each position in sequence. Press Esc to stop."""

import argparse
import time
import threading
import pyautogui
from pynput import mouse, keyboard

stop_event = threading.Event()
scroll_positions = []
num_positions = 4


def on_click(x, y, button, pressed):
    if pressed:
        scroll_positions.append((x, y))
        print(f"  Position {len(scroll_positions)}/{num_positions}: ({x:.0f}, {y:.0f})")
        if len(scroll_positions) >= num_positions:
            return False


def on_press(key):
    if key == keyboard.Key.esc:
        print("\nEsc pressed — stopping.")
        stop_event.set()
        return False


def scroll_loop(amount, interval):
    """Scroll down at each recorded position in round-robin."""
    print(f"Scrolling {amount} units every {interval}s across {len(scroll_positions)} positions. Press Esc to stop.")
    while not stop_event.is_set():
        for pos in scroll_positions:
            if stop_event.is_set():
                break
            pyautogui.moveTo(pos[0], pos[1])
            pyautogui.scroll(amount)
            time.sleep(interval)
    print("Scrolling stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-scroll at multiple clicked positions")
    parser.add_argument("-p", "--positions", type=int, default=4,
                        help="Number of positions to record")
    parser.add_argument("-n", "--amount", type=int, default=-100,
                        help="Scroll amount per tick (positive=down with natural scrolling)")
    parser.add_argument("-t", "--interval", type=float, default=0.5,
                        help="Time between scrolls in seconds")
    args = parser.parse_args()

    num_positions = args.positions

    kb_listener = keyboard.Listener(on_press=on_press)
    kb_listener.start()

    print(f"Click {num_positions} positions on screen...")
    with mouse.Listener(on_click=on_click) as m:
        m.join()

    if stop_event.is_set():
        kb_listener.join()
        exit(0)

    scroll_loop(amount=args.amount, interval=args.interval)
    kb_listener.join()
