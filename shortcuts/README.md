# Sair on iPhone (Apple Shortcuts)

Two Shortcuts for Dad's iPhone. No app to install. Both talk to the Sair hub over the tailnet only.

- **Sair** hears today's walk (mission spoken twice, text on screen).
- **Sair Bolo** records a voice note after the walk and shows the diary entry.

Placeholders used below: `<hub>.<tailnet>.ts.net` is the hub's tailnet HTTPS name, `<DAD token>` is Dad's device token (given by the owner). Never write real values into this file, a screenshot or a shared Shortcut.

Strings marked `(not in design)` are not in the design export; they are working copy for the owner to confirm.

## Before you start

1. The iPhone has Tailscale installed and connected to the family tailnet (iOS: turn on "On-Demand" so it reconnects by itself).
2. Open the Shortcuts app and make a new Shortcut for each section below.
3. Do not use iCloud "Share" on these Shortcuts: the token is stored inside them.

## Paste the token once

At the top of each Shortcut add a **Text** action and paste `<DAD token>` into it (no quotes, no "Bearer", no spaces or line break). Name it with **Set Variable** → `token`. Every request below sends the header:

```
Authorization: Bearer <token variable>
```

In "Get Contents of URL" use **Show More → Headers → Add new header**: key `Authorization`, value `Bearer ` followed by the `token` variable. Do not share these Shortcuts.

## Shortcut 1: "Sair" (before the walk)

1. **Text** → `<DAD token>`; **Set Variable** `token`.
2. **Get Contents of URL**
   - URL: `https://<hub>.<tailnet>.ts.net/mission/today`
   - Method: GET. Header `Authorization: Bearer <token>`.
3. **Get Dictionary Value** → key `mission_id` from the result → **Set Variable** `mission_id`.
4. **Get Dictionary Value** → key `text` from the step 2 result → **Set Variable** `mission_text`.
5. **Get Contents of URL**
   - URL: `https://<hub>.<tailnet>.ts.net/audio/` + `mission_id`
   - Method: GET. Same header. (The result is the spoken mission audio.)
6. **Play Sound** (the audio from step 5).
7. **Wait** 2 seconds.
8. **Play Sound** again (the same audio). The mission is spoken twice with a 2 s pause (journey J1).
9. **Show Result** with `mission_text`. Screen title: `Today's walk`. Card status line: `Reading aloud` (both from the "Sair on iPhone" frame `shortcut-siri`; the frame also shows the labels "Sair" and "Shortcut").

Then put the phone on the shelf.

Optional: if the mission audio is not ready yet, `audio_path` in the step 2 reply is empty; step 5 may answer 404. Show the text anyway (see troubleshooting).

## Shortcut 2: "Sair Bolo" (after the walk)

1. **Text** → `<DAD token>`; **Set Variable** `token`.
2. **Record Audio**
   - Quality: Normal. Start: Immediately. Finish: **On Tap**, and set a 60 second limit. Note: Shortcuts may not offer a time limit on every iOS version; if it does not, the hub still accepts up to 90 s (FR-J1) so On Tap is fine.
   - Screen text (from frame `shortcut-bolo-talking`): `Sair Bolo`, `Tell Sair what you noticed`, a timer (`0:18` in the mock), and `Tap to stop · up to a minute`. Shortcuts draws this screen itself; the title and prompt are whatever you name the Shortcut and the Record Audio prompt.
   - **Set Variable** `recording`.
3. **Get Contents of URL**
   - URL: `https://<hub>.<tailnet>.ts.net/walk`
   - Method: POST. Request Body: **Form**. Add a field of type **File**, key `audio`, value `recording`. Header `Authorization: Bearer <token>`.
4. **Get Dictionary Value** → `walk_id` → **Set Variable** `walk_id`.
5. **Show Notification** or **Show Result**: `Writing it down…` (from the app frame `writing`; the Shortcut mock does not show this step).
6. **Repeat** 15 times:
   1. **Wait** 3 seconds.
   2. **Get Contents of URL**: GET `https://<hub>.<tailnet>.ts.net/walk/` + `walk_id`, same header.
   3. **Get Dictionary Value** → `status`.
   4. **If** `status` is `done`: **Set Variable** `entry` to the `entry` value; stop repeating (add **Stop This Shortcut** after the Delete Files step below, or set a `finished` flag and check it at the top of the loop).
   5. **If** `status` is `lost`: stop repeating, go to the lost message below.

   15 rounds x 3 s is about 45 s (FR-J8).
7. If `status` is `done`: **Show Result**
   - Title: `Got it. Saved.`
   - Body: the `entry` text. (The mock also shows the subtitle `It's in the diary. Enjoy your evening.`; use it when the entry is not shown.) The frame `shortcut-bolo-saved` says Siri can read this out loud.
8. If 45 s pass and the status is still `received`: **Show Result** `Your diary entry will be ready soon` (from the PRD; not in design). The entry will be in the diary later.
9. Always, last of all (both paths, and also after the lost path): **Delete Files** on the `recording` variable (FR-J9, FR-J6).
   - Set "Delete immediately" ON, so the file does not sit in Recently Deleted.
   - The recording from **Record Audio** is a temporary file inside the Shortcut's own storage. If you skip this action, iOS may keep it in the Shortcut's temp folder until the system cleans it. The hub has its own copy and deletes it after transcription, so deleting the phone copy leaves nothing behind.
   - If **Delete Files** reports it cannot delete the item, tell the owner; do not save the recording anywhere.
10. Lost message, when `status` is `lost`: `I couldn't hear that clearly.` and `Sometimes the fan or the traffic is loud. Let's try once more.` (both from the app frame `couldnt-hear`; used here for the silent-audio case, `reason` = `audio_silent`; other reasons: not in design).

The owner's phone gets a separate ntfy message (`Mom + Dad walked · 2 things noticed · mission done`, frame `shortcut-notification`). Dad's phone does not get it.

## Siri phrases

- "Hey Siri, Sair" runs **Sair**. (frame `shortcut-siri` heading: `Hey Siri, Sair`)
- "Hey Siri, Sair Bolo" runs **Sair Bolo**. (frame heading: `Sair Bolo`)

The Siri phrase is the Shortcut's name. If Siri mishears Dad (it hears "Cyber", "Sare" and so on):

1. Open the Shortcut, tap its name at the top, **Rename**.
2. Try a word Siri hears reliably for Dad, said out loud a few times, for example a short word he picks himself.
3. Test with "Hey Siri, <new name>". Keep the old name in a note for the owner (PRD §18.7: the phrase must work for Dad; rename if misheard).
4. Alternatively add the Shortcut to the Home Screen or Lock Screen and tap it, no voice needed.

## Troubleshooting

| What you see | What it means | What to do |
|---|---|---|
| `Show Result` says nothing useful, or an error with 401 | Token is wrong or has a stray space or line break | Re-paste the token into the Text action (once, no "Bearer" inside it). Ask the owner for a fresh token if it still fails. |
| 403 | Dad's token used on an owner-only address | Check the URL path; the Shortcuts only use `/mission/today`, `/audio/...`, `/walk`, `/walk/...`. |
| 404 on `/audio/...` | Spoken mission not ready or older than 48 h | Show the text from `/mission/today` anyway; try again in a minute. |
| 404 on `/walk/<id>` | `walk_id` empty (the upload failed) | Run **Sair Bolo** again. Check Tailscale is connected. |
| No answer at all, spinning, "could not connect" | The hub computer is asleep or Tailscale is off | Wait a minute and try again. Check the Tailscale app shows Connected. Message on screen if you add one: `Sair's home computer is sleeping.` (frame `hub-sleeping`; the rest of that frame, "Your note will be saved when it wakes up", does not apply: a Shortcut cannot queue, so record again). |
| `Your diary entry will be ready soon` | The entry took more than 45 s | Nothing to do. It will be in the diary. |
