# Internet / YouTube downloads and video codecs

**Authors:**
David Martin Moncunill (david.martinm@ucjc.edu)
César Andrés Sánchez (cesar.andress@ucjc.edu)

**Affiliation:** Camilo José Cela University (UCJC), Madrid, Spain

---

## Why this matters

FightSafe AI **samples frames** from your video using **OpenCV** (`VideoCapture`). On many systems, OpenCV is built **without** a full **AV1** (or **VP9** / **HEVC**) decode path, or **hardware** acceleration fails, so the reader returns **no decodable frames** even when the file plays fine in a browser or VLC.

Downloads from the **internet** (including **`yt-dlp`**) often default to **AV1** for efficiency. That is **not** a bug in FightSafe: it is a **codec + OpenCV** limitation for local batch processing.

**Symptoms:** logs such as `Failed to get pixel format`, `Get current frame error`, or “No frames extracted” / empty `frames/` with no clear I/O error.

**Mitigation:** **re-encode** the clip to **H.264** (widely supported) before `extract-frames` or `run-pipeline`, or use **`ffmpeg`** to ask `yt-dlp` for a compatible format when possible (see your `yt-dlp` docs for `--merge-output-format` / format selection).

Respect **YouTube Terms of Service**, **copyright**, and your **local ethics** and consent policies for any source video.

---

## Re-encode to H.264 (recommended)

Requires **`ffmpeg`** on your `PATH`.

**Full file:**

```bash
ffmpeg -y -i "IN.mp4" -c:v libx264 -crf 23 -pix_fmt yuv420p -c:a aac "OUT_h264.mp4"
```

**Short segment** (faster for pipeline tests; adjust `-ss` start and `-t` duration):

```bash
ffmpeg -y -i "IN.mp4" -ss 00:01:00 -t 00:00:30 \
  -c:v libx264 -crf 23 -pix_fmt yuv420p -c:a aac "OUT_clip_h264.mp4"
```

Then point `--video` to `OUT_h264.mp4` or `OUT_clip_h264.mp4`.

---

## Optional: `yt-dlp` format hint

With **`fightsafe download`**, the merged file’s codec still depends on what YouTube and `yt-dlp` select. If you get zero frames, **re-encode** the downloaded file with the command above, or use `yt-dlp`’s **format** options to prefer an **H.264** (avc) stream when available (check current `yt-dlp` help; flags change over time).

---

## Related commands

- `fightsafe extract-frames` — first stage that fails if decoding fails.
- `fightsafe run-pipeline` — starts with the same frame extraction.

See also [`docs/mvp-demo.md`](mvp-demo.md) and the main [`README`](../README.md): **Installation** lists `ffmpeg` as recommended for cutting and video tooling.
