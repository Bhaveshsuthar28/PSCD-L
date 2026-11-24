import type {
  AnalyticsOverviewResponse,
  ApiErrorEnvelope,
  EventAnalyticsResponse,
  EventDetailResponse,
  EventListResponse,
  EvidenceListResponse,
  EvidencePaginatedResponse,
  EvidenceRecord,
  HealthResponse,
  JobCancelResponse,
  ProcessingAnalyticsResponse,
  ProcessingJobCreateRequest,
  ProcessingJobListResponse,
  ProcessingJobLogsResponse,
  ProcessingJobRecord,
  ProcessingJobSummaryResponse,
  SystemModelResponse,
  SystemResourcesResponse,
  SystemSettingsResponse,
  SystemStatusResponse,
  TimelineAnalyticsResponse,
  VideoAnalyticsResponse,
  VideoDeleteResponse,
  VideoListResponse,
  VideoMetadataDetail,
  VideoRecord,
} from "@/types/api";

const RAW_BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "";
export const API_BASE_URL = RAW_BASE_URL.replace(/\/+$/, "");
const API_V1 = "/api/v1";

export class TrinetraApiError extends Error {
  status: number;
  code: string;
  details: unknown;
  requestId: string | null;

  constructor(
    message: string,
    options: {
      status: number;
      code: string;
      details?: unknown;
      requestId?: string | null;
    },
  ) {
    super(message);
    this.name = "TrinetraApiError";
    this.status = options.status;
    this.code = options.code;
    this.details = options.details ?? null;
    this.requestId = options.requestId ?? null;
  }
}

export function resolveApiUrl(path: string): string {
  if (!path) return "";
  if (/^https?:\/\//i.test(path)) return path;
  const normalized = path.startsWith("/") ? path : `/${path}`;
  return `${API_BASE_URL}${normalized}`;
}

function parseErrorPayload(rawText: string, status: number): TrinetraApiError {
  try {
    const parsed = JSON.parse(rawText) as Partial<ApiErrorEnvelope>;
    if (parsed && parsed.error && typeof parsed.error.message === "string") {
      return new TrinetraApiError(parsed.error.message, {
        status,
        code: parsed.error.code || `HTTP_${status}`,
        details: parsed.error.details ?? null,
        requestId: parsed.error.request_id ?? null,
      });
    }
  } catch {
    // Non-JSON response fallback
  }
  return new TrinetraApiError(
    rawText.trim() || `Request failed with status ${status}.`,
    { status, code: `HTTP_${status}` },
  );
}

interface RequestOptions {
  method?: "GET" | "POST" | "PUT" | "DELETE";
  body?: unknown;
  signal?: AbortSignal;
  timeoutMs?: number;
}

async function requestJson<T>(
  path: string,
  options: RequestOptions = {},
): Promise<T> {
  const { method = "GET", body, signal, timeoutMs = 20_000 } = options;
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort("Request timed out"), timeoutMs);

  const forwardAbort = () => controller.abort(signal?.reason);
  if (signal) {
    if (signal.aborted) {
      controller.abort(signal.reason);
    } else {
      signal.addEventListener("abort", forwardAbort, { once: true });
    }
  }

  try {
    const response = await fetch(resolveApiUrl(path), {
      method,
      headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
      body: body !== undefined ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });

    const text = await response.text();
    if (!response.ok) {
      throw parseErrorPayload(text, response.status);
    }

    if (!text) {
      return {} as T;
    }
    return JSON.parse(text) as T;
  } catch (err) {
    if (err instanceof TrinetraApiError) throw err;
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new TrinetraApiError("Request was cancelled or timed out.", {
        status: 0,
        code: "REQUEST_ABORTED",
      });
    }
    throw new TrinetraApiError(
      err instanceof Error
        ? err.message
        : "Unable to reach TRINETRA backend server.",
      { status: 0, code: "NETWORK_ERROR" },
    );
  } finally {
    window.clearTimeout(timer);
    if (signal) {
      signal.removeEventListener("abort", forwardAbort);
    }
  }
}

function buildQuery(params: Record<string, string | number | boolean | undefined | null>): string {
  const sp = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== "") {
      sp.set(key, String(value));
    }
  }
  const qs = sp.toString();
  return qs ? `?${qs}` : "";
}

export interface UploadTaskHandle {
  promise: Promise<VideoRecord>;
  abort: () => void;
}

export function uploadVideoWithProgress(
  file: File,
  onProgress: (percent: number) => void,
): UploadTaskHandle {
  const xhr = new XMLHttpRequest();

  const promise = new Promise<VideoRecord>((resolve, reject) => {
    xhr.open("POST", resolveApiUrl(`${API_V1}/videos/upload`));

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && event.total > 0) {
        onProgress(Math.min(100, Math.round((event.loaded / event.total) * 100)));
      }
    };

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText) as VideoRecord);
        } catch {
          reject(
            new TrinetraApiError("Invalid JSON response from upload endpoint.", {
              status: xhr.status,
              code: "INVALID_SERVER_RESPONSE",
            }),
          );
        }
      } else {
        reject(parseErrorPayload(xhr.responseText, xhr.status));
      }
    };

    xhr.onerror = () => {
      reject(
        new TrinetraApiError("Network error interrupted the video upload.", {
          status: 0,
          code: "UPLOAD_NETWORK_ERROR",
        }),
      );
    };

    xhr.onabort = () => {
      reject(
        new TrinetraApiError("Upload was cancelled.", {
          status: 0,
          code: "UPLOAD_CANCELLED",
        }),
      );
    };

    const formData = new FormData();
    formData.append("file", file);
    xhr.send(formData);
  });

  return {
    promise,
    abort: () => xhr.abort(),
  };
}

export const trinetraApi = {
  // Health & System
  getHealth: (signal?: AbortSignal) =>
    requestJson<HealthResponse>("/health", { signal, timeoutMs: 8000 }),

  getSystemStatus: (signal?: AbortSignal) =>
    requestJson<SystemStatusResponse>(`${API_V1}/system/status`, { signal }),

  getSystemResources: (signal?: AbortSignal) =>
    requestJson<SystemResourcesResponse>(`${API_V1}/system/resources`, { signal }),

  getSystemModel: (signal?: AbortSignal) =>
    requestJson<SystemModelResponse>(`${API_V1}/system/model`, { signal }),

  // Settings
  getSettings: (signal?: AbortSignal) =>
    requestJson<SystemSettingsResponse>(`${API_V1}/settings`, { signal }),

  updateSettings: (settings: Record<string, unknown>) =>
    requestJson<SystemSettingsResponse>(`${API_V1}/settings`, {
      method: "PUT",
      body: { settings },
    }),

  // Videos
  listVideos: (
    params: {
      page?: number;
      page_size?: number;
      sort_by?: string;
      sort_order?: "asc" | "desc";
    } = {},
    signal?: AbortSignal,
  ) =>
    requestJson<VideoListResponse>(
      `${API_V1}/videos${buildQuery(params)}`,
      { signal },
    ),

  getVideo: (videoId: string, signal?: AbortSignal) =>
    requestJson<VideoRecord>(`${API_V1}/videos/${encodeURIComponent(videoId)}`, {
      signal,
    }),

  getVideoMetadata: (videoId: string, signal?: AbortSignal) =>
    requestJson<VideoMetadataDetail>(
      `${API_V1}/videos/${encodeURIComponent(videoId)}/metadata`,
      { signal },
    ),

  getVideoStreamUrl: (videoId: string) =>
    resolveApiUrl(`${API_V1}/videos/${encodeURIComponent(videoId)}/stream`),

  getVideoFrameUrl: (videoId: string, t = 0) =>
    resolveApiUrl(
      `${API_V1}/videos/${encodeURIComponent(videoId)}/frame?t=${Math.max(0, t).toFixed(3)}`,
    ),

  getVideoMjpegUrl: (videoId: string, start = 0, speed = 1, nonce = 0) =>
    resolveApiUrl(
      `${API_V1}/videos/${encodeURIComponent(videoId)}/mjpeg?start=${Math.max(0, start).toFixed(3)}&speed=${speed}&_=${nonce}`,
    ),

  deleteVideo: (videoId: string) =>
    requestJson<VideoDeleteResponse>(
      `${API_V1}/videos/${encodeURIComponent(videoId)}`,
      { method: "DELETE" },
    ),

  // Processing
  createProcessingJob: (payload: ProcessingJobCreateRequest) =>
    requestJson<ProcessingJobRecord>(`${API_V1}/processing/jobs`, {
      method: "POST",
      body: payload,
    }),

  listProcessingJobs: (
    params: {
      video_id?: string;
      status?: string;
      page?: number;
      page_size?: number;
      sort_order?: "asc" | "desc";
    } = {},
    signal?: AbortSignal,
  ) =>
    requestJson<ProcessingJobListResponse>(
      `${API_V1}/processing/jobs${buildQuery(params)}`,
      { signal },
    ),

  getProcessingJob: (jobId: string, signal?: AbortSignal) =>
    requestJson<ProcessingJobRecord>(
      `${API_V1}/processing/jobs/${encodeURIComponent(jobId)}`,
      { signal },
    ),

  cancelProcessingJob: (jobId: string) =>
    requestJson<JobCancelResponse>(
      `${API_V1}/processing/jobs/${encodeURIComponent(jobId)}/cancel`,
      { method: "POST" },
    ),

  getProcessingJobLogs: (
    jobId: string,
    params: { limit?: number; offset?: number } = {},
    signal?: AbortSignal,
  ) =>
    requestJson<ProcessingJobLogsResponse>(
      `${API_V1}/processing/jobs/${encodeURIComponent(jobId)}/logs${buildQuery(params)}`,
      { signal },
    ),

  getProcessingJobSummary: (jobId: string, signal?: AbortSignal) =>
    requestJson<ProcessingJobSummaryResponse>(
      `${API_V1}/processing/jobs/${encodeURIComponent(jobId)}/summary`,
      { signal },
    ),

  getProcessingJobStreamUrl: (jobId: string) =>
    resolveApiUrl(`${API_V1}/processing/jobs/${encodeURIComponent(jobId)}/stream`),

  getProcessingJobExportUrl: (
    jobId: string,
    format: "json" | "csv" | "zip",
    overwrite = false,
  ) =>
    resolveApiUrl(
      `${API_V1}/processing/jobs/${encodeURIComponent(jobId)}/export${buildQuery({ format, overwrite })}`,
    ),

  // Events
  listEvents: (
    params: {
      video_id?: string;
      job_id?: string;
      event_state?: string;
      was_confirmed?: boolean;
      review_required?: boolean;
      post_confirmation_degraded?: boolean;
      min_confidence?: number;
      start_time_seconds?: number;
      end_time_seconds?: number;
      min_duration?: number;
      sort_by?: string;
      sort_order?: "asc" | "desc";
      page?: number;
      page_size?: number;
    } = {},
    signal?: AbortSignal,
  ) =>
    requestJson<EventListResponse>(`${API_V1}/events${buildQuery(params)}`, {
      signal,
    }),

  getEventDetail: (eventId: string, signal?: AbortSignal) =>
    requestJson<EventDetailResponse>(
      `${API_V1}/events/${encodeURIComponent(eventId)}`,
      { signal },
    ),

  listVideoEvents: (
    videoId: string,
    params: { page?: number; page_size?: number } = {},
    signal?: AbortSignal,
  ) =>
    requestJson<EventListResponse>(
      `${API_V1}/videos/${encodeURIComponent(videoId)}/events${buildQuery(params)}`,
      { signal },
    ),

  // Evidence
  listEvidence: (
    params: {
      video_id?: string;
      job_id?: string;
      event_id?: string;
      evidence_type?: string;
      page?: number;
      page_size?: number;
    } = {},
    signal?: AbortSignal,
  ) =>
    requestJson<EvidencePaginatedResponse>(
      `${API_V1}/evidence${buildQuery(params)}`,
      { signal },
    ),

  listEventEvidence: (eventId: string, signal?: AbortSignal) =>
    requestJson<EvidenceListResponse>(
      `${API_V1}/events/${encodeURIComponent(eventId)}/evidence`,
      { signal },
    ),

  getEvidence: (evidenceId: string, signal?: AbortSignal) =>
    requestJson<EvidenceRecord>(
      `${API_V1}/evidence/${encodeURIComponent(evidenceId)}`,
      { signal },
    ),

  // Analytics
  getAnalyticsOverview: (signal?: AbortSignal) =>
    requestJson<AnalyticsOverviewResponse>(`${API_V1}/analytics/overview`, {
      signal,
    }),

  getAnalyticsVideos: (signal?: AbortSignal) =>
    requestJson<VideoAnalyticsResponse>(`${API_V1}/analytics/videos`, { signal }),

  getAnalyticsEvents: (
    params: { start_date?: string; end_date?: string } = {},
    signal?: AbortSignal,
  ) =>
    requestJson<EventAnalyticsResponse>(
      `${API_V1}/analytics/events${buildQuery(params)}`,
      { signal },
    ),

  getAnalyticsProcessing: (signal?: AbortSignal) =>
    requestJson<ProcessingAnalyticsResponse>(
      `${API_V1}/analytics/processing`,
      { signal },
    ),

  getAnalyticsTimeline: (
    params: { start_date?: string; end_date?: string } = {},
    signal?: AbortSignal,
  ) =>
    requestJson<TimelineAnalyticsResponse>(
      `${API_V1}/analytics/timeline${buildQuery(params)}`,
      { signal },
    ),
};
