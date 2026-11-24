import { useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  Camera,
  Download,
  Maximize2,
  Pause,
  Play,
  Volume2,
  VolumeX,
} from "lucide-react";
import { trinetraApi } from "@/api/client";
import { Button } from "@/components/ui/Button";
import { useToast } from "@/components/ui/Toast";
import { formatDuration } from "@/lib/utils";
import type { VideoRecord } from "@/types/api";

const UNSUPPORTED_NATIVE_FORMATS = new Set(["avi", "mkv"]);
const UNSUPPORTED_NATIVE_CODECS = [
  "fmp4",
  "mp4v",
  "xvid",
  "divx",
  "mjpg",
  "mjpeg",
  "wmv",
];

function shouldPreferServerStream(video: VideoRecord): boolean {
  const fmt = (video.format ?? "").toLowerCase();
  if (UNSUPPORTED_NATIVE_FORMATS.has(fmt)) return true;
  const codec = (video.codec ?? "").toLowerCase();
  return UNSUPPORTED_NATIVE_CODECS.some((token) => codec.includes(token));
}

export function VideoPlayer({ video }: { video: VideoRecord }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const seekCommitTimerRef = useRef<number | null>(null);
  const { pushToast } = useToast();

  const [playbackMode, setPlaybackMode] = useState<"native" | "server">(() =>
    shouldPreferServerStream(video) ? "server" : "native",
  );
  const [playing, setPlaying] = useState(false);
  const [muted, setMuted] = useState(false);
  const [volume, setVolume] = useState(1);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(video.duration_seconds || 0);
  const [speed, setSpeed] = useState(1);
  const [loading, setLoading] = useState(true);
  const [playbackError, setPlaybackError] = useState<string | null>(null);

  const [serverFrameTime, setServerFrameTime] = useState(0);
  const [serverStreamAnchor, setServerStreamAnchor] = useState({
    start: 0,
    speed: 1,
    nonce: 0,
  });

  const streamUrl = trinetraApi.getVideoStreamUrl(video.id);

  useEffect(() => {
    if (seekCommitTimerRef.current !== null) {
      window.clearTimeout(seekCommitTimerRef.current);
      seekCommitTimerRef.current = null;
    }
    setPlaybackMode(shouldPreferServerStream(video) ? "server" : "native");
    setPlaying(false);
    setCurrentTime(0);
    setServerFrameTime(0);
    setServerStreamAnchor((prev) => ({ start: 0, speed: 1, nonce: prev.nonce + 1 }));
    setSpeed(1);
    setDuration(video.duration_seconds || 0);
    setLoading(true);
    setPlaybackError(null);
  }, [video.id, video.format, video.codec, video.duration_seconds]);

  useEffect(() => {
    return () => {
      if (seekCommitTimerRef.current !== null) {
        window.clearTimeout(seekCommitTimerRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (playbackMode !== "server" || !playing) return;
    const startedAtMs = performance.now();
    const baseTime = serverStreamAnchor.start;
    const activeSpeed = serverStreamAnchor.speed;
    const maxDuration = duration > 0 ? duration : 0;

    const timer = window.setInterval(() => {
      const elapsedSec = ((performance.now() - startedAtMs) / 1000) * activeSpeed;
      const next = baseTime + elapsedSec;
      if (maxDuration > 0 && next >= maxDuration) {
        setCurrentTime(maxDuration);
        setServerFrameTime(maxDuration);
        setPlaying(false);
        return;
      }
      setCurrentTime(next);
    }, 100);

    return () => window.clearInterval(timer);
  }, [playbackMode, playing, serverStreamAnchor, duration]);

  const togglePlay = () => {
    if (playbackError) return;

    if (playbackMode === "server") {
      if (playing) {
        setServerFrameTime(currentTime);
        setPlaying(false);
      } else {
        const restartAt =
          duration > 0 && currentTime >= Math.max(0, duration - 0.15) ? 0 : currentTime;
        setCurrentTime(restartAt);
        setLoading(true);
        setServerStreamAnchor((prev) => ({
          start: restartAt,
          speed,
          nonce: prev.nonce + 1,
        }));
        setPlaying(true);
      }
      return;
    }

    const el = videoRef.current;
    if (!el) return;
    if (el.paused) {
      void el.play().catch(() => {
        // Fallback to server-decoded stream if browser rejects play() due to codec.
        setPlaybackMode("server");
        setServerFrameTime(currentTime);
        setLoading(true);
      });
    } else {
      el.pause();
    }
  };

  const commitServerSeek = (nextTime: number) => {
    if (playing) {
      setLoading(true);
      setServerStreamAnchor((prev) => ({
        start: nextTime,
        speed,
        nonce: prev.nonce + 1,
      }));
    } else {
      setLoading(true);
      setServerFrameTime(nextTime);
    }
  };

  const handleSeek = (nextTime: number) => {
    setCurrentTime(nextTime);
    if (playbackMode === "server") {
      if (seekCommitTimerRef.current !== null) {
        window.clearTimeout(seekCommitTimerRef.current);
      }
      seekCommitTimerRef.current = window.setTimeout(() => {
        seekCommitTimerRef.current = null;
        commitServerSeek(nextTime);
      }, 90);
      return;
    }

    const el = videoRef.current;
    if (!el) return;
    el.currentTime = nextTime;
  };

  const handleSpeedChange = (nextSpeed: number) => {
    setSpeed(nextSpeed);
    if (playbackMode === "server") {
      if (playing) {
        setLoading(true);
        setServerStreamAnchor((prev) => ({
          start: currentTime,
          speed: nextSpeed,
          nonce: prev.nonce + 1,
        }));
      }
      return;
    }
    if (videoRef.current) {
      videoRef.current.playbackRate = nextSpeed;
    }
  };

  const handleVolumeChange = (nextVol: number) => {
    setVolume(nextVol);
    setMuted(nextVol === 0);
    if (videoRef.current) {
      videoRef.current.volume = nextVol;
      videoRef.current.muted = nextVol === 0;
    }
  };

  const toggleMute = () => {
    const next = !muted;
    setMuted(next);
    if (videoRef.current) {
      videoRef.current.muted = next;
    }
  };

  const handleFullscreen = () => {
    const target = containerRef.current ?? videoRef.current;
    if (!target || !target.requestFullscreen) return;
    void target.requestFullscreen();
  };

  const handleCaptureFrame = () => {
    const canvas = document.createElement("canvas");
    if (playbackMode === "server") {
      const img = imgRef.current;
      if (!img || img.naturalWidth === 0 || img.naturalHeight === 0) {
        pushToast({
          tone: "warning",
          title: "Frame capture unavailable",
          description: "Video frame has not finished loading.",
        });
        return;
      }
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    } else {
      const el = videoRef.current;
      if (!el || el.videoWidth === 0 || el.videoHeight === 0) {
        pushToast({
          tone: "warning",
          title: "Frame capture unavailable",
          description: "Video frame is not decoded in browser playback.",
        });
        return;
      }
      canvas.width = el.videoWidth;
      canvas.height = el.videoHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.drawImage(el, 0, 0, canvas.width, canvas.height);
    }

    try {
      const dataUrl = canvas.toDataURL("image/png");
      const a = document.createElement("a");
      a.href = dataUrl;
      a.download = `${video.original_filename}_frame_${currentTime.toFixed(2)}s.png`;
      a.click();
    } catch {
      pushToast({
        tone: "error",
        title: "Frame capture failed",
        description: "Could not export current video frame.",
      });
    }
  };

  const serverSrc = playing
    ? trinetraApi.getVideoMjpegUrl(
        video.id,
        serverStreamAnchor.start,
        serverStreamAnchor.speed,
        serverStreamAnchor.nonce,
      )
    : trinetraApi.getVideoFrameUrl(video.id, serverFrameTime);

  return (
    <div ref={containerRef} className="space-y-2 bg-surface">
      <div className="relative overflow-hidden rounded-lg border border-border bg-black">
        {playbackError ? (
          <div className="flex min-h-[220px] flex-col items-center justify-center gap-3 bg-surface-muted p-8 text-center text-foreground">
            <AlertTriangle className="h-6 w-6 text-status-warning" aria-hidden="true" />
            <div className="max-w-md space-y-1">
              <p className="text-sm font-medium text-foreground">
                Inline video playback unavailable for {video.format.toUpperCase()}
              </p>
              <p className="text-xs text-muted-foreground">{playbackError}</p>
            </div>
            <a href={streamUrl} download={video.original_filename}>
              <Button variant="outline" size="sm">
                <Download className="h-3.5 w-3.5" aria-hidden="true" />
                Download Original Video
              </Button>
            </a>
          </div>
        ) : playbackMode === "server" ? (
          <>
            <img
              ref={imgRef}
              src={serverSrc}
              alt={`Video frame preview for ${video.original_filename}`}
              className="max-h-[340px] w-full bg-black object-contain select-none"
              onLoad={() => setLoading(false)}
              onError={() => {
                setLoading(false);
                setPlaying(false);
                setPlaybackError(
                  `Unable to decode ${video.format.toUpperCase()} (${video.codec ?? "unknown codec"}) via server stream. Download the original file or inspect extracted evidence frames.`,
                );
              }}
            />
            <span className="pointer-events-none absolute top-2 right-2 rounded bg-black/70 px-2 py-0.5 text-[10px] font-medium text-white">
              Server stream · {video.format.toUpperCase()}
              {video.codec ? ` (${video.codec})` : ""}
            </span>
            {loading && (
              <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-black/40 text-xs text-white">
                Loading video stream…
              </div>
            )}
          </>
        ) : (
          <>
            <video
              ref={videoRef}
              src={streamUrl}
              preload="metadata"
              playsInline
              className="max-h-[340px] w-full bg-black object-contain"
              onLoadedMetadata={(e) => {
                setLoading(false);
                const d = e.currentTarget.duration;
                if (Number.isFinite(d) && d > 0) setDuration(d);
              }}
              onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
              onPlay={() => setPlaying(true)}
              onPause={() => setPlaying(false)}
              onEnded={() => setPlaying(false)}
              onWaiting={() => setLoading(true)}
              onCanPlay={() => setLoading(false)}
              onError={() => {
                setPlaying(false);
                setPlaybackMode("server");
                setServerFrameTime(currentTime);
                setLoading(true);
              }}
            />
            {loading && (
              <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-black/40 text-xs text-white">
                Loading video metadata…
              </div>
            )}
          </>
        )}
      </div>

      {!playbackError && (
        <div className="flex flex-wrap items-center justify-between gap-2 rounded-md border border-border bg-surface-muted px-3 py-2">
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={togglePlay}
              aria-label={playing ? "Pause video" : "Play video"}
            >
              {playing ? (
                <Pause className="h-3.5 w-3.5" aria-hidden="true" />
              ) : (
                <Play className="h-3.5 w-3.5" aria-hidden="true" />
              )}
            </Button>
            <span className="text-xs tabular-nums text-foreground">
              {formatDuration(currentTime)} / {formatDuration(duration)}
            </span>
          </div>

          <input
            type="range"
            min={0}
            max={Math.max(duration, 0.1)}
            step={0.1}
            value={Math.min(currentTime, duration || 0)}
            onChange={(e) => handleSeek(Number(e.target.value))}
            aria-label="Seek video position"
            className="min-w-[140px] flex-1 accent-primary"
          />

          <div className="flex items-center gap-2">
            {playbackMode === "native" && (
              <>
                <button
                  type="button"
                  onClick={toggleMute}
                  aria-label={muted ? "Unmute" : "Mute"}
                  className="rounded p-1 text-muted-foreground hover:text-foreground"
                >
                  {muted ? <VolumeX className="h-3.5 w-3.5" /> : <Volume2 className="h-3.5 w-3.5" />}
                </button>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.1}
                  value={muted ? 0 : volume}
                  onChange={(e) => handleVolumeChange(Number(e.target.value))}
                  aria-label="Volume"
                  className="hidden w-16 accent-primary sm:block"
                />
              </>
            )}
            <select
              aria-label="Playback speed"
              value={speed}
              onChange={(e) => handleSpeedChange(Number(e.target.value))}
              className="rounded border border-border bg-surface px-1.5 py-1 text-xs text-foreground"
            >
              <option value={0.5}>0.5x</option>
              <option value={1}>1x</option>
              <option value={1.5}>1.5x</option>
              <option value={2}>2x</option>
            </select>
            <Button
              variant="outline"
              size="sm"
              onClick={handleCaptureFrame}
              title="Capture current frame as PNG"
            >
              <Camera className="h-3.5 w-3.5" aria-hidden="true" />
            </Button>
            <a href={streamUrl} download={video.original_filename} title="Download original video">
              <Button variant="outline" size="sm" aria-label="Download original video">
                <Download className="h-3.5 w-3.5" aria-hidden="true" />
              </Button>
            </a>
            <Button
              variant="outline"
              size="sm"
              onClick={handleFullscreen}
              title="Fullscreen"
              aria-label="Fullscreen video"
            >
              <Maximize2 className="h-3.5 w-3.5" aria-hidden="true" />
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
