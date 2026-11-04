# Demo day: what to do and when (demo at 8:00)

## Tonight, before you sleep
1. Plug in the laptop. **Leave the lid open and the Claude app running.** New training runs stop starting at about 04:30.
2. Don't restart, shut down or install updates. If Windows has a restart pending, postpone it.
3. Don't use the laptop for anything heavy: it slows training.
4. Send `docs/DEMO_SPEECH.md` to Sejal and Aashrith so they know their parts. Tell them the numbers will change in the
   morning, and that Sejal's part was built and run on your laptop.
5. Agree where you are meeting and when.

## 06:30: check it worked
1. Double-click `check_demo.bat` in the project folder.
2. It must end with **READY FOR THE DEMO**.
3. Ask Claude to update the speech and runbook if that hasn't happened yet, then send the final speech to your teammates.

If it says the new model isn't finished, `start_demo.bat` uses the backup model from October. The demo still works, but then
use the numbers and figures in `C:\spoofdet-data\backup_2026-10-20\`.

## 06:45: rehearse once
1. Double-click `start_demo.bat`, then open `http://127.0.0.1:7860`.
2. Leave the threshold slider alone: it starts at the right value.
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
| Laptop dies | Only this laptop has the model, so copy `backup_screens` and the figures to a USB drive or the team chat this morning |

## After the demo
Push the code and open the PR, so that `main` finally has everyone's work.
