import { useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import {
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  Download,
  Play,
  Radio,
  RotateCcw,
  Square,
} from "lucide-react";
import { trinetraApi } from "@/api/client";
import { Button } from "@/components/ui/Button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { PaginationControls } from "@/components/ui/PaginationControls";
import {
  EmptyState,
  ErrorState,
  TableSkeleton,
} from "@/components/ui/StateFeedback";
import { JobStatusBadge, StatusBadge } from "@/components/ui/StatusBadge";
import { useToast } from "@/components/ui/Toast";
import { useJobStream } from "@/hooks/useJobStream";
import { formatDuration, formatTimestampIso, truncateId } from "@/lib/utils";
import type { ProcessingConfigurationInput, ProcessingJobRecord } from "@/types/api";

const processingSchema = z
  .object({
    video_id: z.string().min(1, "Please select a video."),
    reference_seconds: z.coerce.number().min(0, "Reference timestamp cannot be negative."),
    start_seconds: z.coerce.number().min(0, "Start timestamp cannot be negative."),
    end_seconds: z
      .union([z.literal(""), z.coerce.number().positive("End timestamp must be > 0.")])
      .optional(),
    sampling_interval_seconds: z.coerce
      .number()
      .min(0.1, "Minimum sampling interval is 0.1s.")
      .max(60, "Maximum sampling interval is 60s."),
    min_area: z.union([z.literal(""), z.coerce.number().int().min(20).max(100000)]).optional(),
    min_persistence_seconds: z
      .union([z.literal(""), z.coerce.number().min(1).max(3600)])
      .optional(),
    min_observations: z
      .union([z.literal(""), z.coerce.number().int().min(2).max(1000)])
      .optional(),
    settling_min_seconds: z
      .union([z.literal(""), z.coerce.number().min(0).max(600)])
      .optional(),
  })
  .refine(
    (vals) =>
      vals.end_seconds === "" ||
      vals.end_seconds === undefined ||
      vals.end_seconds > vals.start_seconds,
    {
      message: "End timestamp must be strictly greater than start timestamp.",
      path: ["end_seconds"],
    },
  );

type ProcessingFormValues = z.infer<typeof processingSchema>;

const ACTIVE_STATUSES = new Set([
  "QUEUED",
  "VALIDATING",
  "INITIALIZING",
  "PROCESSING",
  "FINALIZING",
  "CANCEL_REQUESTED",
]);

export function ProcessingPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const queryClient = useQueryClient();
  const { pushToast } = useToast();

  const initialVideoId = searchParams.get("videoId") ?? "";
  const selectedJobId = searchParams.get("jobId");

  const [jobsPage, setJobsPage] = useState(1);
  const [statusFilter, setStatusFilter] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);

  const videosQuery = useQuery({
    queryKey: ["videos", "selector"],
    queryFn: ({ signal }) =>
      trinetraApi.listVideos({ page: 1, page_size: 100 }, signal),
  });

  const jobsQuery = useQuery({
    queryKey: ["jobs", "list", jobsPage, statusFilter],
    queryFn: ({ signal }) =>
      trinetraApi.listProcessingJobs(
        {
          page: jobsPage,
          page_size: 10,
          status: statusFilter || undefined,
        },
        signal,
      ),
    refetchInterval: 8_000,
  });

  const selectedJobQuery = useQuery({
    queryKey: ["job", selectedJobId],
    enabled: Boolean(selectedJobId),
    queryFn: ({ signal }) => trinetraApi.getProcessingJob(selectedJobId!, signal),
    refetchInterval: (query) => {
      const st = query.state.data?.status;
      return st && ACTIVE_STATUSES.has(st) ? 4_000 : false;
    },
  });

  const logsQuery = useQuery({
    queryKey: ["job-logs", selectedJobId],
    enabled: Boolean(selectedJobId),
    queryFn: ({ signal }) =>
      trinetraApi.getProcessingJobLogs(selectedJobId!, { limit: 100 }, signal),
    refetchInterval: selectedJobQuery.data && ACTIVE_STATUSES.has(selectedJobQuery.data.status)
      ? 5_000
      : false,
  });

  const summaryQuery = useQuery({
    queryKey: ["job-summary", selectedJobId],
    enabled: Boolean(
      selectedJobId && selectedJobQuery.data?.status === "COMPLETED",
    ),
    queryFn: ({ signal }) =>
      trinetraApi.getProcessingJobSummary(selectedJobId!, signal),
  });

  const { telemetry, connectionState, notifications } = useJobStream(
    selectedJobId,
    selectedJobQuery.data,
    () => {
      void queryClient.invalidateQueries({ queryKey: ["job", selectedJobId] });
      void queryClient.invalidateQueries({ queryKey: ["jobs"] });
      void queryClient.invalidateQueries({ queryKey: ["job-logs", selectedJobId] });
      void queryClient.invalidateQueries({ queryKey: ["job-summary", selectedJobId] });
      void queryClient.invalidateQueries({ queryKey: ["analytics-overview"] });
      void queryClient.invalidateQueries({ queryKey: ["events"] });
    },
  );

  const {
    register,
    handleSubmit,
    watch,
    setValue,
    setError,
    formState: { errors },
  } = useForm<ProcessingFormValues>({
    resolver: zodResolver(processingSchema),
    defaultValues: {
      video_id: initialVideoId,
      reference_seconds: 0,
      start_seconds: 0,
      end_seconds: "",
      sampling_interval_seconds: 1.0,
      min_area: "",
      min_persistence_seconds: "",
      min_observations: "",
      settling_min_seconds: "",
    },
  });

  useEffect(() => {
    if (initialVideoId && videosQuery.data?.items.some((v) => v.id === initialVideoId)) {
      setValue("video_id", initialVideoId);
    }
  }, [initialVideoId, videosQuery.data, setValue]);

  const watchedVideoId = watch("video_id");
  const selectedVideo = useMemo(
    () => videosQuery.data?.items.find((v) => v.id === watchedVideoId),
    [videosQuery.data, watchedVideoId],
  );

  const createJobMutation = useMutation({
    mutationFn: (payload: {
      video_id: string;
      reference_seconds: number;
      start_seconds: number;
      end_seconds: number | null;
      sampling_interval_seconds: number;
      configuration: ProcessingConfigurationInput;
    }) => trinetraApi.createProcessingJob(payload),
    onSuccess: (job) => {
      pushToast({
        tone: "success",
        title: "Processing job queued",
        description: `Job ${truncateId(job.id)} queued for background inference.`,
      });
      setSearchParams((prev) => {
        const next = new URLSearchParams(prev);
        next.set("jobId", job.id);
        return next;
      });
      void queryClient.invalidateQueries({ queryKey: ["jobs"] });
      void queryClient.invalidateQueries({ queryKey: ["system-status"] });
    },
    onError: (err) => {
      pushToast({
        tone: "error",
        title: "Job submission rejected",
        description: err instanceof Error ? err.message : "Failed to create job.",
      });
    },
  });

  const cancelMutation = useMutation({
    mutationFn: (jobId: string) => trinetraApi.cancelProcessingJob(jobId),
    onSuccess: (res) => {
      pushToast({
        tone: "info",
        title: "Cancellation requested",
        description: res.message,
      });
      void queryClient.invalidateQueries({ queryKey: ["job", res.job_id] });
      void queryClient.invalidateQueries({ queryKey: ["jobs"] });
    },
    onError: (err) => {
      pushToast({
        tone: "error",
        title: "Cancellation failed",
        description: err instanceof Error ? err.message : "Could not cancel job.",
      });
    },
  });

  const onSubmit = (values: ProcessingFormValues) => {
    if (selectedVideo) {
      if (values.reference_seconds >= selectedVideo.duration_seconds) {
        setError("reference_seconds", {
          message: `Must be less than video duration (${selectedVideo.duration_seconds.toFixed(2)}s).`,
        });
        return;
      }
      if (values.start_seconds >= selectedVideo.duration_seconds) {
        setError("start_seconds", {
          message: `Must be less than video duration (${selectedVideo.duration_seconds.toFixed(2)}s).`,
        });
        return;
      }
      if (
        typeof values.end_seconds === "number" &&
        values.end_seconds > selectedVideo.duration_seconds
      ) {
        setError("end_seconds", {
          message: `Cannot exceed video duration (${selectedVideo.duration_seconds.toFixed(2)}s).`,
        });
        return;
      }
    }

    const configuration: ProcessingConfigurationInput = {};
    if (typeof values.min_area === "number") {
      configuration.candidate_config = { min_area: values.min_area };
    }
    const tempCfg: NonNullable<ProcessingConfigurationInput["temporal_config"]> = {};
    if (typeof values.min_persistence_seconds === "number") {
      tempCfg.min_persistence_seconds = values.min_persistence_seconds;
    }
    if (typeof values.min_observations === "number") {
      tempCfg.min_observations = values.min_observations;
    }
    if (typeof values.settling_min_seconds === "number") {
      tempCfg.settling_min_seconds = values.settling_min_seconds;
    }
    if (Object.keys(tempCfg).length > 0) {
      configuration.temporal_config = tempCfg;
    }

    createJobMutation.mutate({
      video_id: values.video_id,
      reference_seconds: values.reference_seconds,
      start_seconds: values.start_seconds,
      end_seconds:
        typeof values.end_seconds === "number" ? values.end_seconds : null,
      sampling_interval_seconds: values.sampling_interval_seconds,
      configuration,
    });
  };

  const handleRetryJob = (job: ProcessingJobRecord) => {
    createJobMutation.mutate({
      video_id: job.video_id,
      reference_seconds: job.reference_seconds,
      start_seconds: job.start_seconds,
      end_seconds: job.end_seconds,
      sampling_interval_seconds: job.sampling_interval_seconds,
      configuration: (job.configuration_snapshot ?? {}) as ProcessingConfigurationInput,
    });
  };

  const selectJob = (jobId: string) => {
    setSearchParams((prev) => {
      const next = new URLSearchParams(prev);
      next.set("jobId", jobId);
      return next;
    });
  };

  const liveStatus = telemetry?.status ?? selectedJobQuery.data?.status ?? "";
  const liveStage = telemetry?.stage ?? selectedJobQuery.data?.current_stage ?? "";
  const livePct = telemetry?.progress_percentage ?? selectedJobQuery.data?.progress_percentage ?? null;
  const isJobActive = ACTIVE_STATUSES.has(liveStatus);

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-12">
        {/* Left: Processing Configuration Form */}
        <Card className="xl:col-span-5">
          <CardHeader>
            <CardTitle>Configure Processing Job</CardTitle>
            <span className="text-[11px] text-muted-foreground">
              Frozen Task 1 + Task 2 Pipeline
            </span>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit(onSubmit)} className="space-y-4" noValidate>
              <div>
                <label
                  htmlFor="video_id"
                  className="block text-xs font-medium text-foreground mb-1"
                >
                  Surveillance Video
                </label>
                <select
                  id="video_id"
                  {...register("video_id")}
                  className="h-9 w-full rounded-md border border-border bg-surface px-2.5 text-xs text-foreground focus-visible:outline-2 focus-visible:outline-ring"
                >
                  <option value="">Select an uploaded video…</option>
                  {(videosQuery.data?.items ?? []).map((v) => (
                    <option key={v.id} value={v.id}>
                      {v.original_filename} ({formatDuration(v.duration_seconds)}, {v.width}×{v.height})
                    </option>
                  ))}
                </select>
                {errors.video_id && (
                  <p className="mt-1 text-xs text-status-danger">
                    {errors.video_id.message}
                  </p>
                )}
                {selectedVideo && (
                  <p className="mt-1 text-[11px] text-muted-foreground tabular-nums">
                    Duration: {selectedVideo.duration_seconds.toFixed(2)}s ·{" "}
                    {selectedVideo.fps.toFixed(2)} fps · {selectedVideo.frame_count} frames
                  </p>
                )}
              </div>

              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                <div>
                  <label
                    htmlFor="reference_seconds"
                    className="block text-xs font-medium text-foreground mb-1"
                  >
                    Reference Timestamp (s)
                  </label>
                  <input
                    id="reference_seconds"
                    type="number"
                    step="0.5"
                    min="0"
                    {...register("reference_seconds")}
                    className="h-9 w-full rounded-md border border-border bg-surface px-2.5 text-xs tabular-nums text-foreground"
                  />
                  {errors.reference_seconds && (
                    <p className="mt-1 text-xs text-status-danger">
                      {errors.reference_seconds.message}
                    </p>
                  )}
                </div>

                <div>
                  <label
                    htmlFor="sampling_interval_seconds"
                    className="block text-xs font-medium text-foreground mb-1"
                  >
                    Sampling Interval (s)
                  </label>
                  <input
                    id="sampling_interval_seconds"
                    type="number"
                    step="0.1"
                    min="0.1"
                    max="60"
                    {...register("sampling_interval_seconds")}
                    className="h-9 w-full rounded-md border border-border bg-surface px-2.5 text-xs tabular-nums text-foreground"
                  />
                  {errors.sampling_interval_seconds && (
                    <p className="mt-1 text-xs text-status-danger">
                      {errors.sampling_interval_seconds.message}
                    </p>
                  )}
                </div>

                <div>
                  <label
                    htmlFor="start_seconds"
                    className="block text-xs font-medium text-foreground mb-1"
                  >
                    Start Timestamp (s)
                  </label>
                  <input
                    id="start_seconds"
                    type="number"
                    step="0.5"
                    min="0"
                    {...register("start_seconds")}
                    className="h-9 w-full rounded-md border border-border bg-surface px-2.5 text-xs tabular-nums text-foreground"
                  />
                  {errors.start_seconds && (
                    <p className="mt-1 text-xs text-status-danger">
                      {errors.start_seconds.message}
                    </p>
                  )}
                </div>

                <div>
                  <label
                    htmlFor="end_seconds"
                    className="block text-xs font-medium text-foreground mb-1"
                  >
                    End Timestamp (s, optional)
                  </label>
                  <input
                    id="end_seconds"
                    type="number"
                    step="0.5"
                    min="0.1"
                    placeholder="End of video"
                    {...register("end_seconds")}
                    className="h-9 w-full rounded-md border border-border bg-surface px-2.5 text-xs tabular-nums text-foreground"
                  />
                  {errors.end_seconds && (
                    <p className="mt-1 text-xs text-status-danger">
                      {errors.end_seconds.message}
                    </p>
                  )}
                </div>
              </div>

              <div className="border-t border-border pt-3">
                <button
                  type="button"
                  onClick={() => setShowAdvanced((v) => !v)}
                  className="flex w-full items-center justify-between text-xs font-medium text-muted-foreground hover:text-foreground"
                >
                  <span>Allowlisted Temporal &amp; Candidate Overrides</span>
                  {showAdvanced ? (
                    <ChevronUp className="h-4 w-4" />
                  ) : (
                    <ChevronDown className="h-4 w-4" />
                  )}
                </button>
                {showAdvanced && (
                  <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
                    <div>
                      <label
                        htmlFor="min_area"
                        className="block text-[11px] font-medium text-muted-foreground mb-1"
                      >
                        Min Candidate Area (px²)
                      </label>
                      <input
                        id="min_area"
                        type="number"
                        placeholder="Default (100)"
                        {...register("min_area")}
                        className="h-8 w-full rounded border border-border bg-surface px-2 text-xs tabular-nums"
                      />
                    </div>
                    <div>
                      <label
                        htmlFor="min_persistence_seconds"
                        className="block text-[11px] font-medium text-muted-foreground mb-1"
                      >
                        Min Persistence (s)
                      </label>
                      <input
                        id="min_persistence_seconds"
                        type="number"
                        step="0.5"
                        placeholder="Default (10.0)"
                        {...register("min_persistence_seconds")}
                        className="h-8 w-full rounded border border-border bg-surface px-2 text-xs tabular-nums"
                      />
                    </div>
                    <div>
                      <label
                        htmlFor="min_observations"
                        className="block text-[11px] font-medium text-muted-foreground mb-1"
                      >
                        Min Observations
                      </label>
                      <input
                        id="min_observations"
                        type="number"
                        placeholder="Default (6)"
                        {...register("min_observations")}
                        className="h-8 w-full rounded border border-border bg-surface px-2 text-xs tabular-nums"
                      />
                    </div>
                    <div>
                      <label
                        htmlFor="settling_min_seconds"
                        className="block text-[11px] font-medium text-muted-foreground mb-1"
                      >
                        Settling Min Duration (s)
                      </label>
                      <input
                        id="settling_min_seconds"
                        type="number"
                        step="0.5"
                        placeholder="Default (3.0)"
                        {...register("settling_min_seconds")}
                        className="h-8 w-full rounded border border-border bg-surface px-2 text-xs tabular-nums"
                      />
                    </div>
                  </div>
                )}
              </div>

              <div className="flex justify-end pt-1">
                <Button
                  type="submit"
                  loading={createJobMutation.isPending}
                  disabled={createJobMutation.isPending}
                >
                  <Play className="h-3.5 w-3.5" aria-hidden="true" />
                  Start Processing
                </Button>
              </div>
            </form>
          </CardContent>
        </Card>

        {/* Right: Live Job Monitor */}
        <Card className="xl:col-span-7">
          <CardHeader>
            <div className="flex items-center gap-2">
              <CardTitle>Live Processing Monitor</CardTitle>
              {selectedJobId && (
                <span className="inline-flex items-center gap-1 rounded border border-border bg-surface-muted px-2 py-0.5 text-[11px] text-muted-foreground">
                  <Radio className="h-3 w-3" aria-hidden="true" />
                  Stream: {connectionState}
                </span>
              )}
            </div>
            {selectedJobId && selectedJobQuery.data && (
              <div className="flex items-center gap-2">
                <JobStatusBadge status={liveStatus} />
                {isJobActive && liveStatus !== "CANCEL_REQUESTED" && (
                  <Button
                    variant="danger"
                    size="sm"
                    loading={cancelMutation.isPending}
                    onClick={() => cancelMutation.mutate(selectedJobId)}
                  >
                    <Square className="h-3 w-3" aria-hidden="true" />
                    Cancel Job
                  </Button>
                )}
                {(liveStatus === "FAILED" || liveStatus === "CANCELLED") && (
                  <Button
                    variant="outline"
                    size="sm"
                    loading={createJobMutation.isPending}
                    onClick={() => handleRetryJob(selectedJobQuery.data!)}
                  >
                    <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
                    Retry Job
                  </Button>
                )}
              </div>
            )}
          </CardHeader>
          <CardContent>
            {!selectedJobId ? (
              <EmptyState
                title="No job selected"
                description="Submit a new processing job or select a run from the history table below to observe live progress and logs."
              />
            ) : selectedJobQuery.isLoading ? (
              <TableSkeleton rows={4} />
            ) : selectedJobQuery.isError ? (
              <ErrorState
                message={selectedJobQuery.error.message}
                onRetry={() => void selectedJobQuery.refetch()}
              />
            ) : (
              <div className="space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                  <div>
                    <span className="text-muted-foreground">Job ID: </span>
                    <span className="font-mono text-foreground">{selectedJobId}</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Stage: </span>
                    <span className="font-medium text-foreground">{liveStage}</span>
                  </div>
                </div>

                {/* Progress Bar (Actual or Indeterminate) */}
                <div className="space-y-1.5">
                  <div className="flex justify-between text-xs tabular-nums">
                    <span>
                      Frames: {telemetry?.processed_frames ?? 0}
                      {telemetry?.total_frames !== null && telemetry?.total_frames !== undefined
                        ? ` / ${telemetry.total_frames}`
                        : " (total pending)"}
                    </span>
                    <span>
                      {livePct !== null
                        ? `${livePct.toFixed(1)}%`
                        : "Indeterminate"}
                    </span>
                  </div>
                  <div className="h-2.5 w-full overflow-hidden rounded-full bg-surface-muted">
                    {livePct !== null ? (
                      <div
                        className="h-full bg-primary transition-[width] duration-200"
                        style={{ width: `${Math.min(100, Math.max(0, livePct))}%` }}
                      />
                    ) : (
                      <div className="h-full w-1/3 animate-pulse bg-primary" />
                    )}
                  </div>
                </div>

                {/* Telemetry Grid */}
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 rounded-md border border-border bg-surface-muted p-3 text-xs">
                  <div>
                    <span className="text-muted-foreground">Video Time</span>
                    <p className="font-medium tabular-nums">
                      {telemetry?.current_video_timestamp_seconds !== null &&
                      telemetry?.current_video_timestamp_seconds !== undefined
                        ? `${telemetry.current_video_timestamp_seconds.toFixed(2)}s`
                        : "—"}
                    </p>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Throughput</span>
                    <p className="font-medium tabular-nums">
                      {telemetry?.processing_fps !== null &&
                      telemetry?.processing_fps !== undefined
                        ? `${telemetry.processing_fps.toFixed(2)} fps`
                        : "Unavailable"}
                    </p>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Elapsed / ETA</span>
                    <p className="font-medium tabular-nums">
                      {formatDuration(telemetry?.elapsed_seconds)}
                      {telemetry?.estimated_remaining_seconds !== null &&
                      telemetry?.estimated_remaining_seconds !== undefined
                        ? ` / ~${formatDuration(telemetry.estimated_remaining_seconds)}`
                        : ""}
                    </p>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Active Tracks</span>
                    <p className="font-medium tabular-nums">
                      {telemetry?.active_tracks_count ?? 0}
                    </p>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Historical Confirmed</span>
                    <p className="font-medium tabular-nums text-status-success">
                      {telemetry?.confirmed_events_count ?? 0}
                    </p>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Review Required</span>
                    <p className="font-medium tabular-nums text-status-warning">
                      {telemetry?.review_required_events_count ?? 0}
                    </p>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Evidence Status</span>
                    <p className="font-medium">
                      {selectedJobQuery.data?.evidence_generation_status ?? "PENDING"}
                    </p>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Candidates</span>
                    <p className="font-medium tabular-nums">
                      {telemetry?.total_candidates_count ?? 0}
                    </p>
                  </div>
                </div>

                {telemetry?.error_message && (
                  <div
                    role="alert"
                    className="flex items-start gap-2 rounded-md border border-status-danger-border bg-status-danger-bg p-3 text-xs text-status-danger"
                  >
                    <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
                    <div>
                      <p className="font-medium">
                        {telemetry.error_code ?? "JOB_ERROR"}
                      </p>
                      <p className="mt-0.5">{telemetry.error_message}</p>
                    </div>
                  </div>
                )}

                {notifications.length > 0 && (
                  <div className="space-y-1.5">
                    <p className="text-xs font-medium text-foreground">
                      Live Event Notifications
                    </p>
                    <div className="max-h-28 overflow-y-auto space-y-1">
                      {notifications.map((n) => (
                        <div
                          key={n.id}
                          className="flex items-center justify-between rounded border border-border px-2.5 py-1.5 text-xs"
                        >
                          <div className="flex items-center gap-2">
                            <StatusBadge
                              tone={n.type === "event_confirmed" ? "success" : "warning"}
                              label={
                                n.type === "event_confirmed"
                                  ? `Track #${n.track_id} Confirmed`
                                  : `Track #${n.track_id} Review`
                              }
                              icon={n.type === "event_confirmed" ? "check" : "warn"}
                            />
                            <span className="text-muted-foreground">{n.detail}</span>
                          </div>
                          <span className="font-mono text-[11px] tabular-nums">
                            {n.timestamp_seconds.toFixed(2)}s
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {liveStatus === "COMPLETED" && (
                  <div className="flex flex-wrap items-center justify-between gap-2 rounded-md border border-border bg-surface-muted px-3 py-2.5">
                    <div className="text-xs">
                      <span className="font-medium text-foreground">
                        Run Complete
                      </span>
                      {summaryQuery.data?.summary && (
                        <span className="ml-2 text-muted-foreground">
                          {String(
                            (summaryQuery.data.summary as { total_tracks?: number })
                              .total_tracks ?? 0,
                          )}{" "}
                          tracks recorded
                        </span>
                      )}
                    </div>
                    <div className="flex flex-wrap items-center gap-2">
                      <Link to={`/events?jobId=${selectedJobId}`}>
                        <Button size="sm" variant="outline">
                          View Job Events
                        </Button>
                      </Link>
                      {(["json", "csv", "zip"] as const).map((fmt) => (
                        <a
                          key={fmt}
                          href={trinetraApi.getProcessingJobExportUrl(selectedJobId, fmt)}
                          download
                        >
                          <Button size="sm" variant="secondary">
                            <Download className="h-3.5 w-3.5" aria-hidden="true" />
                            {fmt.toUpperCase()}
                          </Button>
                        </a>
                      ))}
                    </div>
                  </div>
                )}

                {/* Job Logs */}
                <div className="space-y-1.5">
                  <p className="text-xs font-medium text-foreground">Execution Logs</p>
                  <div className="max-h-40 overflow-y-auto rounded-md border border-border bg-surface-muted p-2 font-mono text-[11px] space-y-1">
                    {(logsQuery.data?.items ?? []).length === 0 ? (
                      <span className="text-muted-foreground">No log entries yet.</span>
                    ) : (
                      logsQuery.data!.items.map((entry) => (
                        <div key={entry.id} className="flex gap-2">
                          <span className="text-muted-foreground shrink-0">
                            [{entry.level}]
                          </span>
                          <span className="text-foreground break-words">
                            {entry.stage ? `(${entry.stage}) ` : ""}
                            {entry.message}
                          </span>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Job History Table */}
      <Card>
        <CardHeader>
          <CardTitle>Processing Runs</CardTitle>
          <div className="flex items-center gap-2">
            <select
              aria-label="Filter jobs by status"
              value={statusFilter}
              onChange={(e) => {
                setStatusFilter(e.target.value);
                setJobsPage(1);
              }}
              className="h-8 rounded-md border border-border bg-surface px-2.5 text-xs text-foreground"
            >
              <option value="">All statuses</option>
              <option value="QUEUED">Queued</option>
              <option value="PROCESSING">Processing</option>
              <option value="COMPLETED">Completed</option>
              <option value="FAILED">Failed</option>
              <option value="CANCELLED">Cancelled</option>
            </select>
          </div>
        </CardHeader>
        <CardContent className="p-0">
          {jobsQuery.isLoading ? (
            <div className="p-4">
              <TableSkeleton rows={4} />
            </div>
          ) : jobsQuery.isError ? (
            <div className="p-4">
              <ErrorState
                message={jobsQuery.error.message}
                onRetry={() => void jobsQuery.refetch()}
              />
            </div>
          ) : (jobsQuery.data?.items.length ?? 0) === 0 ? (
            <div className="p-4">
              <EmptyState
                title="No processing runs"
                description="Create a job above to analyze an uploaded video."
              />
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="border-b border-border bg-surface-muted text-muted-foreground">
                  <tr>
                    <th className="px-4 py-2.5 font-medium">Job ID</th>
                    <th className="px-4 py-2.5 font-medium">Status</th>
                    <th className="px-4 py-2.5 font-medium">Window</th>
                    <th className="px-4 py-2.5 font-medium">Frames</th>
                    <th className="px-4 py-2.5 font-medium">Confirmed / Review</th>
                    <th className="px-4 py-2.5 font-medium">Created</th>
                    <th className="px-4 py-2.5 font-medium text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {jobsQuery.data!.items.map((job) => (
                    <tr
                      key={job.id}
                      className={
                        job.id === selectedJobId
                          ? "bg-primary-soft/50"
                          : "hover:bg-surface-muted/50"
                      }
                    >
                      <td className="px-4 py-2.5 font-mono">
                        {truncateId(job.id)}
                      </td>
                      <td className="px-4 py-2.5">
                        <JobStatusBadge status={job.status} />
                      </td>
                      <td className="px-4 py-2.5 tabular-nums">
                        ref {job.reference_seconds}s · {job.start_seconds}s–
                        {job.end_seconds !== null ? `${job.end_seconds}s` : "end"}
                      </td>
                      <td className="px-4 py-2.5 tabular-nums">
                        {job.processed_frames}
                        {job.total_frames !== null ? ` / ${job.total_frames}` : ""}
                      </td>
                      <td className="px-4 py-2.5 tabular-nums">
                        {job.confirmed_events_count} /{" "}
                        <span
                          className={
                            job.review_required_events_count > 0
                              ? "text-status-warning font-medium"
                              : ""
                          }
                        >
                          {job.review_required_events_count}
                        </span>
                      </td>
                      <td className="px-4 py-2.5 text-muted-foreground tabular-nums">
                        {formatTimestampIso(job.created_at)}
                      </td>
                      <td className="px-4 py-2.5 text-right">
                        <Button
                          variant={job.id === selectedJobId ? "primary" : "outline"}
                          size="sm"
                          onClick={() => selectJob(job.id)}
                        >
                          Inspect
                        </Button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <PaginationControls
            pagination={jobsQuery.data?.pagination}
            onPageChange={setJobsPage}
          />
        </CardContent>
      </Card>
    </div>
  );
}
