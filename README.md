# Bathukamma Group Dance Video Generator

Create a 35-second Bathukamma dance video from a photograph. The five people are original pixels from the supplied photograph, separated into transparent layers and animated with restrained 2D/2.5D movement. The dancers form a shallow arc around a central, stepped-triangle Bathukamma flower stack; detected neighboring wrists are linked so the group reads as hand-in-hand. This does not generate live-action footage or alter faces.

## Features

- YOLO person segmentation that looks for five people, sorts them left to right, and saves `assets/person1.png` through `assets/person5.png`.
- A manual fallback that uses five transparent PNG cutouts when automatic segmentation is unavailable or does not find exactly five people.
- Pose-estimated hand links that move with the dancers, plus an arc around a larger central flower stack.
- Librosa tempo and beat analysis saved to `generated/audio_analysis.json`, with a fixed half-second beat fallback if analysis fails.
- 30 FPS animation with synchronized and alternating sway, small whole-person rotation and scale changes, a generated Bathukamma flower stack, drifting petals, subtle lights, and a gentle camera move.
- A 1280 x 720, 30 FPS landscape H.264 MP4 with the supplied song embedded as AAC audio.
- The finished video is saved to `output/bathukamma_dance.mp4` and opened in the system's default video player.

## Requirements

- Windows, macOS, or Linux
- Python 3.10 or newer
- Internet access on the first automatic-segmentation run to download the YOLO model weights. If that is unavailable, use the manual cutout fallback below.

The project uses Pillow, NumPy, OpenCV, librosa, imageio/imageio-ffmpeg, MoviePy, and Ultralytics. FFmpeg is supplied through `imageio-ffmpeg`; a separate FFmpeg installation is not required.

## Install

From the project folder in PowerShell:

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with `source venv/bin/activate`.

## Input files

Place the source photo at:

```text
input/women.png
```

Place the song at:

```text
input/bathukamma_song.mp3
```

The supplied filename `input/batukamma_song.mp3` is also accepted. The original media files are not modified.

## Run

```powershell
python main.py
```

The program creates the needed folders if missing, extracts the five cutouts under `assets/`, and writes beat information to `generated/audio_analysis.json`. It renders the video, embeds the supplied song, saves the finished MP4 to `output/bathukamma_dance.mp4`, and opens it in the default video player. The temporary silent render is deleted afterward.

The video opens in the system's default player, where it can fit the available screen while retaining its 16:9 landscape aspect ratio. The saved MP4 can also be shared or opened later from the `output/` folder.

For a quick two-second generation and playback check:

```powershell
python main.py --test-seconds 2
```

The normal command creates a 35-second video. The shorter test option is useful for checking the pipeline.

## Person segmentation and fallback

The program first runs the pretrained `yolov8n-seg.pt` person-segmentation model. On first use, Ultralytics downloads its model weights. It then runs `yolo11n-pose.pt` to locate wrists and links each neighboring pair of selected foreground dancers. Pose matches are restricted to the segmentation boxes so background people are not linked. If pose detection fails, the flower-circle animation still renders without fabricated hand locations. If segmentation detects more than five people, the five largest/highest-confidence detections are retained and ordered left to right. Each output PNG uses original image pixels with a feathered transparent mask; the people are not face-swapped or regenerated.

If the model is unavailable or does not find exactly five people, the program looks for these manually prepared transparent PNGs:

```text
assets/person1.png
assets/person2.png
assets/person3.png
assets/person4.png
assets/person5.png
```

Keep the complete person, including head and feet where possible, inside each PNG. Name the files from left to right. If the automatic method fails and all five manual files are not present, the program prints their required names and exits without creating a broken video.

## Beat analysis and animation

Librosa loads the song, estimates its tempo, and finds beat times. These are saved in `generated/audio_analysis.json` and used to accent the group's small sways and bounces. If beat detection raises an error or returns no beats, the generator reports `Beat detection failed. Using fallback dance timing.` and uses a beat every 0.5 seconds.

The five complete cutouts move as layers: the outer dancers sway in opposing directions, the center dancer has a little extra bounce, and all dancers share occasional group movement. Rotation stays within six degrees and scale stays between 0.95 and 1.06. No face-specific edits or body warping are performed. The animation is intentionally a Python-generated mixed-media effect from a photograph, not realistic frame-by-frame choreography.

## How the MP4 is made

The renderer creates 1,050 frames at 30 FPS for the default 35-second run and streams them to imageio's bundled FFmpeg as a temporary H.264 clip. FFmpeg embeds the original MP3 as AAC audio in the final MP4; if the song is shorter than the video, it is looped to match the video duration. The silent temporary clip is deleted, and the source song is unchanged.

## Common issues

- **Image missing:** Put `women.png` in `input/` and run `python main.py` again.
- **Song missing:** Put the song in `input/bathukamma_song.mp3`. The alternate supplied spelling `batukamma_song.mp3` is accepted too.
- **YOLO model download fails:** Add the five transparent cutouts listed above. The rest of the video pipeline can then run without automatic segmentation.
- **Beat detection fails:** The video continues with the documented half-second beat timing.
- **The video does not open automatically:** Open `output/bathukamma_dance.mp4` manually in a video player that supports H.264 video and AAC audio.
- **Encoding fails:** Reinstall the bundled encoder packages with `pip install --upgrade imageio imageio-ffmpeg moviepy`, then rerun the script.