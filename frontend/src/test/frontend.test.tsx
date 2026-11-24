import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import { TrinetraApiError, trinetraApi } from "@/api/client";
import { EvidenceOverlayViewer } from "@/components/evidence/EvidenceOverlayViewer";
import { AppShell } from "@/components/layout/AppShell";
import { Dialog } from "@/components/ui/Dialog";
import { JobStatusBadge } from "@/components/ui/StatusBadge";
import { ToastProvider } from "@/components/ui/Toast";
import { VideoPlayer } from "@/components/video/VideoPlayer";
import { ThemeProvider, useTheme } from "@/hooks/useTheme";
import { DashboardPage } from "@/pages/DashboardPage";
import { ProcessingPage } from "@/pages/ProcessingPage";
import { SystemPage } from "@/pages/SystemPage";
import type { EvidenceRecord, VideoRecord } from "@/types/api";

function renderWithProviders(
  ui: React.ReactElement,
  { initialRoute = "/dashboard" } = {},
) {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <ThemeProvider>
      <QueryClientProvider client={qc}>
        <ToastProvider>
          <MemoryRouter initialEntries={[initialRoute]}>{ui}</MemoryRouter>
        </ToastProvider>
      </QueryClientProvider>
    </ThemeProvider>,
  );
}

function ThemeProbe() {
  const { preference, resolvedTheme, setPreference } = useTheme();
  return (
    <div>
      <span data-testid="pref">{preference}</span>
      <span data-testid="resolved">{resolvedTheme}</span>
      <button type="button" onClick={() => setPreference("dark")}>
        Go Dark
      </button>
      <button type="button" onClick={() => setPreference("light")}>
        Go Light
      </button>
    </div>
  );
}

describe("Theme & UI Components", () => {
  it("persists theme preference and updates documentElement classes", async () => {
    const user = userEvent.setup();
    render(
      <ThemeProvider>
        <ThemeProbe />
      </ThemeProvider>,
    );

    expect(screen.getByTestId("pref")).toHaveTextContent("system");
    await user.click(screen.getByText("Go Dark"));
    expect(screen.getByTestId("resolved")).toHaveTextContent("dark");
    expect(document.documentElement.classList.contains("dark")).toBe(true);
    expect(localStorage.getItem("trinetra-theme-preference")).toBe("dark");

    await user.click(screen.getByText("Go Light"));
    expect(document.documentElement.classList.contains("light")).toBe(true);
  });

  it("renders readable status labels alongside icons", () => {
    render(<JobStatusBadge status="COMPLETED" />);
    expect(screen.getByText("Completed")).toBeInTheDocument();
  });

  it("closes Dialog on Escape key", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    render(
      <Dialog open onClose={onClose} title="Test Modal">
        <p>Body</p>
      </Dialog>,
    );
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    await user.keyboard("{Escape}");
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});

describe("API Client Error & Response Handling", () => {
  it("parses structured backend error envelopes", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          success: false,
          error: {
            code: "DUPLICATE_VIDEO",
            message: "Duplicate file",
            details: { existing_video_id: "vid-1" },
            request_id: "req-123",
          },
        }),
        { status: 409, headers: { "Content-Type": "application/json" } },
      ),
    );

    await expect(trinetraApi.getHealth()).rejects.toMatchObject({
      name: "TrinetraApiError",
      status: 409,
      code: "DUPLICATE_VIDEO",
      requestId: "req-123",
    });
    vi.restoreAllMocks();
  });

  it("handles network disconnection cleanly", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValueOnce(new Error("ECONNREFUSED"));
    try {
      await trinetraApi.getHealth();
      expect.fail("Should have thrown");
    } catch (err) {
      expect(err).toBeInstanceOf(TrinetraApiError);
      expect((err as TrinetraApiError).code).toBe("NETWORK_ERROR");
    }
    vi.restoreAllMocks();
  });
});

describe("Pages & Integration Behavior", () => {
  it("renders AppShell navigation and handles offline backend indicator", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("Offline"));
    renderWithProviders(
      <AppShell>
        <div>Shell Content</div>
      </AppShell>,
    );

    expect(screen.getByText("TRINETRA")).toBeInTheDocument();
    expect(screen.getByText("Video Library")).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getByText("Backend Offline")).toBeInTheDocument();
    });
    vi.restoreAllMocks();
  });

  it("renders DashboardPage with separated real metrics and empty states", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.includes("/analytics/overview")) {
        return new Response(
          JSON.stringify({
            total_uploaded_videos: 2,
            total_processing_jobs: 3,
            completed_jobs: 2,
            failed_jobs: 1,
            cancelled_jobs: 0,
            queued_jobs: 0,
            active_jobs: 0,
            total_candidate_events: 10,
            historical_confirmations: 4,
            non_review_confirmations: 3,
            review_required_events: 1,
            post_confirmation_degraded_events: 1,
            total_processing_duration_seconds: 12.5,
            note: "Separated",
          }),
          { status: 200 },
        );
      }
      if (url.includes("/analytics/timeline")) {
        return new Response(JSON.stringify({ items: [], count: 0 }), { status: 200 });
      }
      if (url.includes("/analytics/processing")) {
        return new Response(
          JSON.stringify({
            jobs_by_status: {},
            completed_jobs_count: 0,
            average_processing_fps: null,
            average_elapsed_seconds: null,
            total_processed_frames_completed_jobs: 0,
          }),
          { status: 200 },
        );
      }
      if (url.includes("/health")) {
        return new Response(
          JSON.stringify({
            status: "healthy",
            api: "up",
            database: "connected",
            storage: "writable",
            worker: "running",
            checkpoint_available: true,
            model_initialized: false,
            cuda_available: true,
            timestamp: "2026-11-25T00:00:00Z",
          }),
          { status: 200 },
        );
      }
      return new Response(
        JSON.stringify({
          items: [],
          pagination: {
            page: 1,
            page_size: 10,
            total_items: 0,
            total_pages: 0,
            has_next: false,
            has_previous: false,
          },
        }),
        { status: 200 },
      );
    });

    renderWithProviders(<DashboardPage />);
    await waitFor(() => {
      expect(screen.getByText("Non-Review Confirmations")).toBeInTheDocument();
      expect(screen.getByText("10")).toBeInTheDocument();
      expect(screen.getByText("4")).toBeInTheDocument();
    });
    expect(screen.getAllByText("3").length).toBeGreaterThanOrEqual(2);
    expect(screen.getByText("No event timeline data")).toBeInTheDocument();
    vi.restoreAllMocks();
  });

  it("validates ProcessingPage form timestamps and required video", async () => {
    const user = userEvent.setup();
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          items: [],
          pagination: {
            page: 1,
            page_size: 10,
            total_items: 0,
            total_pages: 0,
            has_next: false,
            has_previous: false,
          },
        }),
        { status: 200 },
      ),
    );

    renderWithProviders(<ProcessingPage />);
    await user.click(screen.getByRole("button", { name: /Start Processing/i }));
    await waitFor(() => {
      expect(screen.getByText("Please select a video.")).toBeInTheDocument();
    });
    vi.restoreAllMocks();
  });

  it("displays Unavailable instead of zero for missing GPU metrics on SystemPage", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.includes("/health")) {
        return new Response(
          JSON.stringify({
            status: "healthy",
            api: "up",
            database: "connected",
            storage: "writable",
            worker: "running",
            checkpoint_available: true,
            model_initialized: false,
            cuda_available: false,
            timestamp: "2026-11-25T00:00:00Z",
          }),
          { status: 200 },
        );
      }
      if (url.includes("/system/status")) {
        return new Response(
          JSON.stringify({
            project: "TRINETRA",
            version: "1.0.0",
            environment: "testing",
            database_connected: true,
            storage_writable: true,
            worker_running: true,
            queued_jobs: 0,
            active_jobs: 0,
            checkpoint_available: true,
            model_initialized: false,
            cuda_available: false,
            scientific_scope: "Visual scene changes",
          }),
          { status: 200 },
        );
      }
      if (url.includes("/system/resources")) {
        return new Response(
          JSON.stringify({
            cpu_percent: null,
            ram_total_bytes: null,
            ram_available_bytes: null,
            ram_used_percent: null,
            disk_total_bytes: 1000,
            disk_used_bytes: 500,
            disk_free_bytes: 500,
            cuda_available: false,
            gpu_name: null,
            gpu_memory_allocated_bytes: null,
            gpu_memory_reserved_bytes: null,
            gpu_memory_total_bytes: null,
          }),
          { status: 200 },
        );
      }
      return new Response(
        JSON.stringify({
          architecture: "DINOv2",
          backbone: "facebook/dinov2-base",
          feature_dimensions: [1, 768, 37, 37],
          output_logits_dimensions: [1, 1, 512, 512],
          checkpoint_exists: true,
          expected_sha256: "A".repeat(64),
          checkpoint_sha256_verified: null,
          model_initialized: false,
          device: "cpu",
          load_time_seconds: null,
          last_error: null,
        }),
        { status: 200 },
      );
    });

    renderWithProviders(<SystemPage />);
    await waitFor(() => {
      expect(screen.getByText("CPU Only")).toBeInTheDocument();
    });
    expect(screen.getAllByText("Unavailable").length).toBeGreaterThanOrEqual(3);
    vi.restoreAllMocks();
  });

  it("renders EvidenceOverlayViewer bounding box and handles missing download URL", () => {
    const item: EvidenceRecord = {
      id: "ev-1",
      event_id: "evt-1",
      job_id: "job-1",
      evidence_type: "probability_map",
      original_filename: "prob.png",
      mime_type: "image/png",
      file_size_bytes: 512,
      sha256: "B".repeat(64),
      timestamp_seconds: 12.5,
      width: 640,
      height: 480,
      bbox_xywh: [100, 80, 50, 40],
      download_url: null,
      created_at: "2026-11-25T00:00:00Z",
    };

    render(<EvidenceOverlayViewer item={item} />);
    expect(
      screen.getByText(/Evidence image unavailable or failed to load/i),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Model spatial probability visualization/i),
    ).toBeInTheDocument();
  });

  it("uses server-decoded stream playback automatically for AVI FMP4 videos", async () => {
    const user = userEvent.setup();
    const aviVideo: VideoRecord = {
      id: "vid-avi-1",
      original_filename: "sample_cam.avi",
      stored_filename: "vid-avi-1.avi",
      format: "avi",
      codec: "FMP4",
      duration_seconds: 120,
      fps: 29.97,
      frame_count: 3596,
      width: 720,
      height: 480,
      file_size_bytes: 45_000_000,
      sha256: "C".repeat(64),
      upload_status: "ready",
      created_at: "2026-11-25T00:00:00Z",
      updated_at: "2026-11-25T00:00:00Z",
    };

    render(
      <ToastProvider>
        <VideoPlayer video={aviVideo} />
      </ToastProvider>,
    );

    const img = screen.getByAltText(/Video frame preview for sample_cam.avi/i) as HTMLImageElement;
    expect(img.getAttribute("src")).toContain("/api/v1/videos/vid-avi-1/frame?t=0.000");
    expect(screen.getByText(/Server stream · AVI \(FMP4\)/i)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Play video" }));
    expect(img.getAttribute("src")).toContain("/api/v1/videos/vid-avi-1/mjpeg?start=0.000&speed=1");
  });

  it("loads and renders CSV and JSON evidence artifacts inline", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    fetchSpy.mockResolvedValueOnce(
      new Response("track_id,state,confidence\n1,CONFIRMED,0.95\n2,REVIEW_REQUIRED,0.61\n", {
        status: 200,
        headers: { "Content-Type": "text/csv" },
      }),
    );

    const csvItem: EvidenceRecord = {
      id: "ev-csv",
      event_id: null,
      job_id: "job-1",
      evidence_type: "exported_report",
      original_filename: "events.csv",
      mime_type: "text/csv",
      file_size_bytes: 64,
      sha256: "D".repeat(64),
      timestamp_seconds: null,
      width: null,
      height: null,
      bbox_xywh: null,
      download_url: "/api/v1/evidence/ev-csv/download",
      created_at: "2026-11-25T00:00:00Z",
    };

    const { rerender } = render(<EvidenceOverlayViewer key={csvItem.id} item={csvItem} />);
    await waitFor(() => {
      expect(screen.getByText("CONFIRMED")).toBeInTheDocument();
    });
    expect(screen.getByText("2 rows × 3 columns")).toBeInTheDocument();

    fetchSpy.mockResolvedValueOnce(
      new Response(JSON.stringify({ track_id: "55", state: "CONFIRMED" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    const jsonItem: EvidenceRecord = {
      ...csvItem,
      id: "ev-json",
      evidence_type: "metadata",
      original_filename: "000146_146_metadata.json",
      mime_type: "application/json",
    };

    rerender(<EvidenceOverlayViewer key={jsonItem.id} item={jsonItem} />);
    await waitFor(() => {
      expect(screen.getByText(/"track_id": "55"/)).toBeInTheDocument();
    });
    vi.restoreAllMocks();
  });
});
