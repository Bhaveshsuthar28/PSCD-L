import { useMemo } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import type { EChartsOption } from "echarts";
import {
  ArrowRight,
  Film,
  Info,
  PlayCircle,
  ShieldAlert,
} from "lucide-react";
import { resolveApiUrl, trinetraApi } from "@/api/client";
import { EChartBox, getChartPalette } from "@/components/charts/EChartBox";
import { Button } from "@/components/ui/Button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";
import {
  EmptyState,
  ErrorState,
  TableSkeleton,
} from "@/components/ui/StateFeedback";
import { EventLifecycleBadge, JobStatusBadge, StatusBadge } from "@/components/ui/StatusBadge";
import { useTheme } from "@/hooks/useTheme";
import { formatDuration, formatTimestampIso, truncateId } from "@/lib/utils";

function MetricCard({
  label,
  value,
  sublabel,
  loading,
}: {
  label: string;
  value: string | number;
  sublabel?: string;
  loading?: boolean;
}) {
  return (
    <Card>
      <CardContent className="p-4">
        <p className="text-xs font-medium text-muted-foreground">{label}</p>
        {loading ? (
          <Skeleton className="mt-2 h-7 w-20" />
        ) : (
          <p className="mt-1 text-2xl font-semibold tracking-tight text-foreground tabular-nums">
            {value}
          </p>
        )}
        {sublabel && (
          <p className="mt-1 text-[11px] text-muted-foreground">{sublabel}</p>
        )}
      </CardContent>
    </Card>
  );
}

export function DashboardPage() {
  const { resolvedTheme } = useTheme();
  const palette = getChartPalette(resolvedTheme === "dark");

  const overviewQuery = useQuery({
    queryKey: ["analytics-overview"],
    queryFn: ({ signal }) => trinetraApi.getAnalyticsOverview(signal),
    refetchInterval: 15_000,
  });

  const timelineQuery = useQuery({
    queryKey: ["analytics-timeline"],
    queryFn: ({ signal }) => trinetraApi.getAnalyticsTimeline({}, signal),
    refetchInterval: 30_000,
  });

  const processingAnalyticsQuery = useQuery({
    queryKey: ["analytics-processing"],
    queryFn: ({ signal }) => trinetraApi.getAnalyticsProcessing(signal),
    refetchInterval: 15_000,
  });

  const recentJobsQuery = useQuery({
    queryKey: ["jobs", "recent"],
    queryFn: ({ signal }) =>
      trinetraApi.listProcessingJobs({ page: 1, page_size: 5 }, signal),
    refetchInterval: 10_000,
  });

  const recentEventsQuery = useQuery({
    queryKey: ["events", "recent"],
    queryFn: ({ signal }) =>
      trinetraApi.listEvents({ page: 1, page_size: 5 }, signal),
    refetchInterval: 15_000,
  });

  const recentEvidenceQuery = useQuery({
    queryKey: ["evidence", "recent"],
    queryFn: ({ signal }) =>
      trinetraApi.listEvidence(
        { page: 1, page_size: 6, evidence_type: "annotated_frame" },
        signal,
      ),
    refetchInterval: 30_000,
  });

  const healthQuery = useQuery({
    queryKey: ["health"],
    queryFn: ({ signal }) => trinetraApi.getHealth(signal),
  });

  const ov = overviewQuery.data;

  const timelineOption = useMemo<EChartsOption | null>(() => {
    const items = timelineQuery.data?.items ?? [];
    if (items.length === 0) return null;
    return {
      tooltip: { trigger: "axis" },
      legend: {
        top: 0,
        textStyle: { color: palette.legendText, fontSize: 11 },
      },
      grid: { left: 40, right: 16, top: 36, bottom: 28 },
      xAxis: {
        type: "category",
        data: items.map((i) => i.date),
        axisLine: { lineStyle: { color: palette.axisLine } },
        axisLabel: { color: palette.axisLabel, fontSize: 11 },
      },
      yAxis: {
        type: "value",
        minInterval: 1,
        splitLine: { lineStyle: { color: palette.splitLine } },
        axisLabel: { color: palette.axisLabel, fontSize: 11 },
      },
      series: [
        {
          name: "Candidate Tracks",
          type: "bar",
          data: items.map((i) => i.candidate_events),
          itemStyle: { color: palette.neutral, borderRadius: [3, 3, 0, 0] },
        },
        {
          name: "Historical Confirmations",
          type: "bar",
          data: items.map((i) => i.historical_confirmations),
          itemStyle: { color: palette.success, borderRadius: [3, 3, 0, 0] },
        },
        {
          name: "Review Required",
          type: "line",
          smooth: false,
          data: items.map((i) => i.review_required_events),
          itemStyle: { color: palette.warning },
        },
      ],
    };
  }, [timelineQuery.data, palette]);

  const jobStatusOption = useMemo<EChartsOption | null>(() => {
    const byStatus = processingAnalyticsQuery.data?.jobs_by_status;
    if (!byStatus) return null;
    const entries = Object.entries(byStatus).filter(([, count]) => count > 0);
    if (entries.length === 0) return null;
    const colorFor = (status: string) => {
      if (status === "COMPLETED") return palette.success;
      if (status === "FAILED") return palette.danger;
      if (status === "CANCEL_REQUESTED") return palette.warning;
      if (status === "PROCESSING" || status === "INITIALIZING" || status === "VALIDATING" || status === "FINALIZING") {
        return palette.primary;
      }
      return palette.neutral;
    };
    return {
      tooltip: { trigger: "item" },
      legend: {
        bottom: 0,
        textStyle: { color: palette.legendText, fontSize: 11 },
      },
      series: [
        {
          name: "Jobs",
          type: "pie",
          radius: ["45%", "70%"],
          center: ["50%", "44%"],
          label: { show: false },
          data: entries.map(([name, value]) => ({
            name,
            value,
            itemStyle: { color: colorFor(name) },
          })),
        },
      ],
    };
  }, [processingAnalyticsQuery.data, palette]);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-border bg-surface px-4 py-3 text-xs text-muted-foreground">
        <div className="flex items-start sm:items-center gap-2">
          <Info className="h-4 w-4 shrink-0 text-primary mt-0.5 sm:mt-0" aria-hidden="true" />
          <span>
            TRINETRA reports <strong className="font-medium text-foreground">persistent visual scene-change events</strong>.
            Candidate tracks, historical confirmations, non-review confirmations, and review-required events are tracked separately.
          </span>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <Link to="/videos">
            <Button variant="outline" size="sm">
              <Film className="h-3.5 w-3.5" aria-hidden="true" />
              Upload Video
            </Button>
          </Link>
          <Link to="/processing">
            <Button size="sm">
              <PlayCircle className="h-3.5 w-3.5" aria-hidden="true" />
              New Job
            </Button>
          </Link>
        </div>
      </div>

      {overviewQuery.isError && (
        <ErrorState
          title="Failed to load overview metrics"
          message={overviewQuery.error.message}
          onRetry={() => void overviewQuery.refetch()}
        />
      )}

      {/* Row 1: Video & Job Metrics */}
      <section aria-label="Video and processing metrics">
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
          <MetricCard
            label="Uploaded Videos"
            value={ov?.total_uploaded_videos ?? 0}
            loading={overviewQuery.isLoading}
          />
          <MetricCard
            label="Total Jobs"
            value={ov?.total_processing_jobs ?? 0}
            sublabel={`Duration: ${formatDuration(ov?.total_processing_duration_seconds ?? 0)}`}
            loading={overviewQuery.isLoading}
          />
          <MetricCard
            label="Completed Jobs"
            value={ov?.completed_jobs ?? 0}
            loading={overviewQuery.isLoading}
          />
          <MetricCard
            label="Active / Queued"
            value={ov?.active_jobs ?? 0}
            sublabel={`${ov?.queued_jobs ?? 0} queued`}
            loading={overviewQuery.isLoading}
          />
          <MetricCard
            label="Failed Jobs"
            value={ov?.failed_jobs ?? 0}
            sublabel={`${ov?.cancelled_jobs ?? 0} cancelled`}
            loading={overviewQuery.isLoading}
          />
        </div>
      </section>

      {/* Row 2: Event Classification Metrics (strictly separated) */}
      <section aria-label="Scene-change event metrics">
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
          <MetricCard
            label="Candidate Events"
            value={ov?.total_candidate_events ?? 0}
            sublabel="All tracked regions"
            loading={overviewQuery.isLoading}
          />
          <MetricCard
            label="Historical Confirmations"
            value={ov?.historical_confirmations ?? 0}
            sublabel="Ever met persistence"
            loading={overviewQuery.isLoading}
          />
          <MetricCard
            label="Non-Review Confirmations"
            value={ov?.non_review_confirmations ?? 0}
            sublabel="No post-confirm degradation"
            loading={overviewQuery.isLoading}
          />
          <MetricCard
            label="Review Required"
            value={ov?.review_required_events ?? 0}
            sublabel="Requires human inspection"
            loading={overviewQuery.isLoading}
          />
          <MetricCard
            label="Post-Confirm Degraded"
            value={ov?.post_confirmation_degraded_events ?? 0}
            sublabel="Stability or settling drift"
            loading={overviewQuery.isLoading}
          />
        </div>
      </section>

      {/* Row 3: Charts */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Event Timeline</CardTitle>
            <span className="text-xs text-muted-foreground">Daily database aggregation</span>
          </CardHeader>
          <CardContent>
            <EChartBox
              option={timelineOption}
              loading={timelineQuery.isLoading}
              error={timelineQuery.isError ? timelineQuery.error.message : null}
              onRetry={() => void timelineQuery.refetch()}
              isEmpty={!timelineOption}
              emptyTitle="No event timeline data"
              emptyDescription="Process a video to record candidate and confirmed scene-change events over time."
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Processing Jobs by Status</CardTitle>
            <Link to="/analytics" className="text-xs text-primary hover:underline">
              Full Analytics
            </Link>
          </CardHeader>
          <CardContent>
            <EChartBox
              option={jobStatusOption}
              loading={processingAnalyticsQuery.isLoading}
              error={
                processingAnalyticsQuery.isError
                  ? processingAnalyticsQuery.error.message
                  : null
              }
              onRetry={() => void processingAnalyticsQuery.refetch()}
              isEmpty={!jobStatusOption}
              emptyTitle="No processing jobs recorded"
              emptyDescription="Submit a job from Video Processing to view status distribution."
            />
          </CardContent>
        </Card>
      </div>

      {/* Row 4: Recent Activity & System Health */}
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Recent Processing Activity</CardTitle>
            <Link
              to="/processing"
              className="inline-flex items-center gap-1 text-xs font-medium text-primary hover:underline"
            >
              View Queue <ArrowRight className="h-3 w-3" />
            </Link>
          </CardHeader>
          <CardContent className="p-0">
            {recentJobsQuery.isLoading ? (
              <div className="p-4">
                <TableSkeleton rows={3} />
              </div>
            ) : recentJobsQuery.isError ? (
              <div className="p-4">
                <ErrorState
                  message={recentJobsQuery.error.message}
                  onRetry={() => void recentJobsQuery.refetch()}
                />
              </div>
            ) : (recentJobsQuery.data?.items.length ?? 0) === 0 ? (
              <div className="p-4">
                <EmptyState
                  title="No processing jobs yet"
                  description="Upload a surveillance video and start a processing job."
                />
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="border-b border-border bg-surface-muted text-muted-foreground">
                    <tr>
                      <th className="px-4 py-2.5 font-medium">Job</th>
                      <th className="px-4 py-2.5 font-medium">Status</th>
                      <th className="px-4 py-2.5 font-medium">Progress</th>
                      <th className="px-4 py-2.5 font-medium">Confirmed</th>
                      <th className="px-4 py-2.5 font-medium">Created</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border">
                    {recentJobsQuery.data!.items.map((job) => (
                      <tr key={job.id} className="hover:bg-surface-muted/50">
                        <td className="px-4 py-2.5 font-mono">
                          <Link
                            to={`/processing?jobId=${job.id}`}
                            className="text-primary hover:underline"
                          >
                            {truncateId(job.id)}
                          </Link>
                        </td>
                        <td className="px-4 py-2.5">
                          <JobStatusBadge status={job.status} />
                        </td>
                        <td className="px-4 py-2.5 tabular-nums">
                          {job.progress_percentage !== null
                            ? `${job.progress_percentage.toFixed(0)}%`
                            : "Indeterminate"}
                        </td>
                        <td className="px-4 py-2.5 tabular-nums">
                          {job.confirmed_events_count}
                          {job.review_required_events_count > 0 && (
                            <span className="ml-1 text-status-warning">
                              ({job.review_required_events_count} review)
                            </span>
                          )}
                        </td>
                        <td className="px-4 py-2.5 text-muted-foreground tabular-nums">
                          {formatTimestampIso(job.created_at)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Recent Scene-Change Events</CardTitle>
            <Link
              to="/events"
              className="inline-flex items-center gap-1 text-xs font-medium text-primary hover:underline"
            >
              Explore Events <ArrowRight className="h-3 w-3" />
            </Link>
          </CardHeader>
          <CardContent className="p-0">
            {recentEventsQuery.isLoading ? (
              <div className="p-4">
                <TableSkeleton rows={3} />
              </div>
            ) : recentEventsQuery.isError ? (
              <div className="p-4">
                <ErrorState
                  message={recentEventsQuery.error.message}
                  onRetry={() => void recentEventsQuery.refetch()}
                />
              </div>
            ) : (recentEventsQuery.data?.items.length ?? 0) === 0 ? (
              <div className="p-4">
                <EmptyState
                  title="No events recorded"
                  description="Detected candidate and persistent scene-change events appear here."
                />
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="border-b border-border bg-surface-muted text-muted-foreground">
                    <tr>
                      <th className="px-4 py-2.5 font-medium">Track</th>
                      <th className="px-4 py-2.5 font-medium">Lifecycle</th>
                      <th className="px-4 py-2.5 font-medium">Review</th>
                      <th className="px-4 py-2.5 font-medium">Confidence</th>
                      <th className="px-4 py-2.5 font-medium">Duration</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border">
                    {recentEventsQuery.data!.items.map((ev) => (
                      <tr key={ev.id} className="hover:bg-surface-muted/50">
                        <td className="px-4 py-2.5 font-medium tabular-nums">
                          <Link
                            to={`/events?eventId=${ev.id}`}
                            className="text-primary hover:underline"
                          >
                            Track #{ev.track_id}
                          </Link>
                        </td>
                        <td className="px-4 py-2.5">
                          <EventLifecycleBadge state={ev.event_state} />
                        </td>
                        <td className="px-4 py-2.5">
                          {ev.review_required ? (
                            <StatusBadge
                              tone="warning"
                              label="Review Required"
                              icon="warn"
                            />
                          ) : ev.was_confirmed ? (
                            <StatusBadge
                              tone="success"
                              label="Confirmed"
                              icon="check"
                            />
                          ) : (
                            <StatusBadge
                              tone="neutral"
                              label="Unconfirmed"
                              icon="neutral"
                            />
                          )}
                        </td>
                        <td className="px-4 py-2.5 tabular-nums">
                          {ev.confidence.toFixed(3)}
                        </td>
                        <td className="px-4 py-2.5 tabular-nums">
                          {formatDuration(ev.duration_seconds)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Row 5: System Health & Recent Evidence Previews */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle>System Readiness</CardTitle>
            <Link to="/system" className="text-xs text-primary hover:underline">
              Diagnostics
            </Link>
          </CardHeader>
          <CardContent className="space-y-2.5 text-xs">
            {healthQuery.isLoading ? (
              <Skeleton className="h-24 w-full" />
            ) : healthQuery.isError ? (
              <ErrorState message={healthQuery.error.message} />
            ) : (
              <>
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">API Process</span>
                  <StatusBadge tone="success" label={healthQuery.data?.api ?? "up"} icon="check" />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Database</span>
                  <StatusBadge
                    tone={healthQuery.data?.database === "connected" ? "success" : "danger"}
                    label={healthQuery.data?.database ?? "unknown"}
                    icon={healthQuery.data?.database === "connected" ? "check" : "error"}
                  />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Background Worker</span>
                  <StatusBadge
                    tone={healthQuery.data?.worker === "running" ? "success" : "danger"}
                    label={healthQuery.data?.worker ?? "stopped"}
                    icon={healthQuery.data?.worker === "running" ? "activity" : "error"}
                  />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Frozen Checkpoint</span>
                  <StatusBadge
                    tone={healthQuery.data?.checkpoint_available ? "success" : "danger"}
                    label={healthQuery.data?.checkpoint_available ? "Available" : "Missing"}
                    icon={healthQuery.data?.checkpoint_available ? "check" : "error"}
                  />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Inference Engine</span>
                  <StatusBadge
                    tone={healthQuery.data?.model_initialized ? "info" : "neutral"}
                    label={healthQuery.data?.model_initialized ? "Loaded" : "Standby (Lazy)"}
                    icon={healthQuery.data?.model_initialized ? "check" : "clock"}
                  />
                </div>
              </>
            )}
          </CardContent>
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Recent Evidence Previews</CardTitle>
            <Link
              to="/evidence"
              className="inline-flex items-center gap-1 text-xs font-medium text-primary hover:underline"
            >
              Open Evidence Viewer <ArrowRight className="h-3 w-3" />
            </Link>
          </CardHeader>
          <CardContent>
            {recentEvidenceQuery.isLoading ? (
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                <Skeleton className="h-28 w-full" />
                <Skeleton className="h-28 w-full" />
                <Skeleton className="h-28 w-full" />
              </div>
            ) : recentEvidenceQuery.isError ? (
              <ErrorState message={recentEvidenceQuery.error.message} />
            ) : (recentEvidenceQuery.data?.items.length ?? 0) === 0 ? (
              <EmptyState
                title="No evidence images generated"
                description="Annotated frames and probability maps appear once an event satisfies persistence criteria."
                icon={<ShieldAlert className="h-5 w-5" />}
              />
            ) : (
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                {recentEvidenceQuery.data!.items.map((item) => (
                  <Link
                    key={item.id}
                    to={`/evidence?evidenceId=${item.id}`}
                    className="group overflow-hidden rounded-md border border-border bg-surface-muted transition-colors hover:border-primary"
                  >
                    {item.download_url ? (
                      <img
                        src={resolveApiUrl(item.download_url)}
                        alt={item.original_filename}
                        loading="lazy"
                        className="h-28 w-full object-cover"
                      />
                    ) : (
                      <div className="flex h-28 items-center justify-center text-xs text-muted-foreground">
                        No preview
                      </div>
                    )}
                    <div className="px-2.5 py-1.5 text-[11px] flex items-center justify-between gap-2 bg-surface">
                      <span className="truncate font-mono text-foreground">
                        {item.timestamp_seconds !== null
                          ? `${item.timestamp_seconds.toFixed(2)}s`
                          : item.original_filename}
                      </span>
                      <span className="text-muted-foreground shrink-0">
                        {item.width && item.height ? `${item.width}×${item.height}` : ""}
                      </span>
                    </div>
                  </Link>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
