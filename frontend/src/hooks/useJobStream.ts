import { useEffect, useRef, useState } from "react";
import { trinetraApi } from "@/api/client";
import type { ProcessingJobRecord } from "@/types/api";

export interface LiveJobTelemetry {
  status: string;
  stage: string;
  progress_percentage: number | null;
  processed_frames: number;
  total_frames: number | null;
  current_video_timestamp_seconds: number | null;
  processing_fps: number | null;
  elapsed_seconds: number | null;
  estimated_remaining_seconds: number | null;
  active_tracks_count: number;
  confirmed_events_count: number;
  review_required_events_count: number;
  total_candidates_count?: number;
  error_code: string | null;
  error_message: string | null;
}

export interface StreamNotification {
  id: string;
  type: "event_confirmed" | "review_required";
  track_id: number;
  timestamp_seconds: number;
  detail: string;
}

const TERMINAL_STATES = new Set(["COMPLETED", "FAILED", "CANCELLED"]);

export function useJobStream(
  jobId: string | null,
  initialJob: ProcessingJobRecord | undefined,
  onTerminal?: () => void,
) {
  const [telemetry, setTelemetry] = useState<LiveJobTelemetry | null>(null);
  const [connectionState, setConnectionState] = useState<
    "idle" | "connecting" | "connected" | "reconnecting" | "closed"
  >("idle");
  const [notifications, setNotifications] = useState<StreamNotification[]>([]);
  const lastEventIdRef = useRef<number>(0);
  const seenNotifIdsRef = useRef<Set<string>>(new Set());
  const onTerminalRef = useRef(onTerminal);
  onTerminalRef.current = onTerminal;

  // Reconcile server snapshot whenever initialJob updates from React Query
  useEffect(() => {
    if (!initialJob) return;
    setTelemetry((prev) => {
      if (prev && TERMINAL_STATES.has(prev.status) && !TERMINAL_STATES.has(initialJob.status)) {
        return prev;
      }
      return {
        status: initialJob.status,
        stage: initialJob.current_stage,
        progress_percentage: initialJob.progress_percentage,
        processed_frames: initialJob.processed_frames,
        total_frames: initialJob.total_frames,
        current_video_timestamp_seconds: initialJob.current_video_timestamp_seconds,
        processing_fps: initialJob.processing_fps,
        elapsed_seconds: initialJob.elapsed_seconds,
        estimated_remaining_seconds: initialJob.estimated_remaining_seconds,
        active_tracks_count: initialJob.active_tracks_count,
        confirmed_events_count: initialJob.confirmed_events_count,
        review_required_events_count: initialJob.review_required_events_count,
        total_candidates_count: initialJob.total_candidates_count,
        error_code: initialJob.error_code,
        error_message: initialJob.error_message,
      };
    });
  }, [initialJob]);

  useEffect(() => {
    if (!jobId) {
      setTelemetry(null);
      setNotifications([]);
      setConnectionState("idle");
      return;
    }

    lastEventIdRef.current = 0;
    seenNotifIdsRef.current.clear();
    setNotifications([]);

    if (initialJob && TERMINAL_STATES.has(initialJob.status)) {
      setConnectionState("closed");
      return;
    }

    if (typeof EventSource === "undefined") {
      setConnectionState("closed");
      return;
    }

    let closed = false;
    let source: EventSource | null = null;
    let reconnectTimer: number | null = null;

    const connect = () => {
      if (closed) return;
      const base = trinetraApi.getProcessingJobStreamUrl(jobId);
      const url =
        lastEventIdRef.current > 0
          ? `${base}?last_event_id=${lastEventIdRef.current}`
          : base;

      setConnectionState((prev) =>
        prev === "reconnecting" ? "reconnecting" : "connecting",
      );

      source = new EventSource(url);

      source.onopen = () => {
        if (!closed) setConnectionState("connected");
      };

      const recordSeq = (e: MessageEvent) => {
        const numericId = Number(e.lastEventId);
        if (Number.isFinite(numericId) && numericId > lastEventIdRef.current) {
          lastEventIdRef.current = numericId;
        }
      };

      const handleStatePayload = (e: MessageEvent) => {
        recordSeq(e);
        try {
          const data = JSON.parse(e.data) as Partial<LiveJobTelemetry>;
          setTelemetry((prev) => ({
            status: data.status ?? prev?.status ?? "QUEUED",
            stage: data.stage ?? prev?.stage ?? "QUEUED",
            progress_percentage:
              data.progress_percentage !== undefined
                ? data.progress_percentage
                : (prev?.progress_percentage ?? null),
            processed_frames:
              data.processed_frames ?? prev?.processed_frames ?? 0,
            total_frames:
              data.total_frames !== undefined
                ? data.total_frames
                : (prev?.total_frames ?? null),
            current_video_timestamp_seconds:
              data.current_video_timestamp_seconds !== undefined
                ? data.current_video_timestamp_seconds
                : (prev?.current_video_timestamp_seconds ?? null),
            processing_fps:
              data.processing_fps !== undefined
                ? data.processing_fps
                : (prev?.processing_fps ?? null),
            elapsed_seconds:
              data.elapsed_seconds !== undefined
                ? data.elapsed_seconds
                : (prev?.elapsed_seconds ?? null),
            estimated_remaining_seconds:
              data.estimated_remaining_seconds !== undefined
                ? data.estimated_remaining_seconds
                : (prev?.estimated_remaining_seconds ?? null),
            active_tracks_count:
              data.active_tracks_count ?? prev?.active_tracks_count ?? 0,
            confirmed_events_count:
              data.confirmed_events_count ?? prev?.confirmed_events_count ?? 0,
            review_required_events_count:
              data.review_required_events_count ??
              prev?.review_required_events_count ??
              0,
            total_candidates_count:
              data.total_candidates_count ?? prev?.total_candidates_count,
            error_code:
              data.error_code !== undefined
                ? data.error_code
                : (prev?.error_code ?? null),
            error_message:
              data.error_message !== undefined
                ? data.error_message
                : (prev?.error_message ?? null),
          }));
        } catch {
          // ignore malformed SSE chunk
        }
      };

      const handleTerminal = (e: MessageEvent) => {
        handleStatePayload(e);
        closed = true;
        setConnectionState("closed");
        source?.close();
        onTerminalRef.current?.();
      };

      const handleEventNotice = (type: "event_confirmed" | "review_required") => (e: MessageEvent) => {
        recordSeq(e);
        try {
          const data = JSON.parse(e.data) as {
            track_id?: number;
            timestamp_seconds?: number;
            confidence?: number;
            reasons?: string[];
          };
          const dedupKey = `${type}:${e.lastEventId || data.track_id}:${data.timestamp_seconds}`;
          if (seenNotifIdsRef.current.has(dedupKey)) return;
          seenNotifIdsRef.current.add(dedupKey);
          setNotifications((prev) => [
            {
              id: dedupKey,
              type,
              track_id: data.track_id ?? 0,
              timestamp_seconds: data.timestamp_seconds ?? 0,
              detail:
                type === "event_confirmed"
                  ? `Heuristic confidence ${typeof data.confidence === "number" ? data.confidence.toFixed(3) : "—"}`
                  : `Degradation: ${(data.reasons ?? []).join(", ") || "Post-confirmation review"}`,
            },
            ...prev.slice(0, 19),
          ]);
        } catch {
          // ignore
        }
      };

      source.addEventListener("snapshot", handleStatePayload as EventListener);
      source.addEventListener("stage_changed", handleStatePayload as EventListener);
      source.addEventListener("progress", handleStatePayload as EventListener);
      source.addEventListener(
        "event_confirmed",
        handleEventNotice("event_confirmed") as EventListener,
      );
      source.addEventListener(
        "review_required",
        handleEventNotice("review_required") as EventListener,
      );
      source.addEventListener("completed", handleTerminal as EventListener);
      source.addEventListener("failed", handleTerminal as EventListener);
      source.addEventListener("cancelled", handleTerminal as EventListener);

      source.onerror = () => {
        source?.close();
        if (closed) return;
        setConnectionState("reconnecting");
        // Reconcile state via standard HTTP API when SSE drops
        void trinetraApi
          .getProcessingJob(jobId)
          .then((latest) => {
            if (closed) return;
            if (TERMINAL_STATES.has(latest.status)) {
              closed = true;
              setConnectionState("closed");
              onTerminalRef.current?.();
            } else {
              reconnectTimer = window.setTimeout(connect, 2000);
            }
          })
          .catch(() => {
            if (!closed) {
              reconnectTimer = window.setTimeout(connect, 3000);
            }
          });
      };
    };

    connect();

    return () => {
      closed = true;
      if (reconnectTimer !== null) window.clearTimeout(reconnectTimer);
      source?.close();
    };
  }, [jobId, initialJob?.status]);

  return { telemetry, connectionState, notifications };
}
