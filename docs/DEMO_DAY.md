# Demo day: what to do and when (demo at 8:00)

## Tonight, before you sleep
Training is finished, so the laptop doesn't need to stay on overnight.
1. Plug in the laptop. Postpone any pending Windows restart so that updates don't run in the morning.
2. Send `docs/DEMO_SPEECH.md` to Sejal and Aashrith so they can learn their parts. These are the final numbers. Tell them
   that Sejal's part was built and trained for her, on Kaggle, so she can read through it first.
3. Agree where you are meeting and when.

## 06:30: check it worked
1. Double-click `check_demo.bat` in the project folder.
2. It must end with **READY FOR THE DEMO**.

If it doesn't, read the PROBLEM lines. `start_demo.bat` falls back to the October model only if the new model's results are
missing. The demo still works then, but its numbers won't match the speech, so use the backup screens instead.

## 06:45: rehearse once
1. Double-click `start_demo.bat`, then open `http://127.0.0.1:7860`.
2. Leave the threshold slider alone: it starts at the right value, 0.22.
3. Run the five clips from `C:\spoofdet-data\demo_clips\`, in order. For each: upload, **Analyze**, scroll to the plots, clear with **X**.
4. Read the speech aloud once with a timer. Target: under 10 minutes.

## 07:30: set up in the room
1. Plug in, connect the projector, and set the display to duplicate.
2. Silence notifications (Focus Assist on) and close other apps.
3. Have these open and ready:
   - The black `start_demo.bat` window, and the browser on the demo page
   - File Explorer at `C:\spoofdet-data\demo_clips\`
   - File Explorer at `results\figures\`
   - `docs\DEMO_SPEECH.md`
4. Run clip 1 once so that everything is loaded.

## During the demo
Follow `docs/DEMO_SPEECH.md`. Remember:
- Say the clips were hand-picked, and that clips 4 and 5 fail on purpose.
- **Never press play on clip 5** (a real person's voice).
- Don't use the microphone unless someone asks, and warn them it may call them fake.

## If something breaks
| Problem | Do this |
|---|---|
| App won't start or crashes | Show `C:\spoofdet-data\demo_clips\backup_screens\` and keep talking |
| Port 7860 busy | Close the other black window, then start again |
| Analyze is slow | Wait up to 30 seconds, then use the backup screens |
| Laptop dies | Only this laptop has the demo, so this morning copy `backup_screens`, the five figures named in the speech and the speech itself to a USB drive or the team chat |

## After the demo
Push the code and open the PR, so that `main` finally has everyone's work.
