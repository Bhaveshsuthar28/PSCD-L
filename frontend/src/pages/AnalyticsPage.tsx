import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import type { EChartsOption } from "echarts";
import { Download, FilterX } from "lucide-react";
import { trinetraApi } from "@/api/client";
import { EChartBox, getChartPalette } from "@/components/charts/EChartBox";
import { Button } from "@/components/ui/Button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { EmptyState, ErrorState, TableSkeleton } from "@/components/ui/StateFeedback";
import { useToast } from "@/components/ui/Toast";
import { useTheme } from "@/hooks/useTheme";
import { formatDuration, truncateId } from "@/lib/utils";

export function AnalyticsPage() {
  const { resolvedTheme } = useTheme();
  const palette = getChartPalette(resolvedTheme === "dark");
  const { pushToast } = useToast();

  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [exportJobId, setExportJobId] = useState("");
  const [exportingFmt, setExportingFmt] = useState<string | null>(null);

  const dateRangeInvalid =
    Boolean(startDate && endDate && endDate < startDate);

  const overviewQ = useQuery({
    queryKey: ["analytics-overview"],
    queryFn: ({ signal }) => trinetraApi.getAnalyticsOverview(signal),
  });

  const eventsAnalyticsQ = useQuery({
    queryKey: ["analytics-events", startDate, endDate],
    enabled: !dateRangeInvalid,
    queryFn: ({ signal }) =>
      trinetraApi.getAnalyticsEvents(
        {
          start_date: startDate || undefined,
          end_date: endDate || undefined,
        },
        signal,
      ),
  });

  const timelineQ = useQuery({
    queryKey: ["analytics-timeline", startDate, endDate],
    enabled: !dateRangeInvalid,
    queryFn: ({ signal }) =>
      trinetraApi.getAnalyticsTimeline(
        {
          start_date: startDate || undefined,
          end_date: endDate || undefined,
        },
        signal,
      ),
  });

  const videosQ = useQuery({
    queryKey: ["analytics-videos"],
    queryFn: ({ signal }) => trinetraApi.getAnalyticsVideos(signal),
  });

  const processingQ = useQuery({
    queryKey: ["analytics-processing"],
    queryFn: ({ signal }) => trinetraApi.getAnalyticsProcessing(signal),
  });

  const completedJobsQ = useQuery({
    queryKey: ["jobs", "exportable"],
    queryFn: ({ signal }) =>
      trinetraApi.listProcessingJobs({ page: 1, page_size: 50 }, signal),
  });

  const exportableJobs = useMemo(
    () =>
      (completedJobsQ.data?.items ?? []).filter((j) =>
        ["COMPLETED", "FAILED", "CANCELLED"].includes(j.status),
      ),
    [completedJobsQ.data],
  );

  const activeExportJobId = exportJobId || exportableJobs[0]?.id || "";

  const handleExport = async (fmt: "json" | "csv" | "zip") => {
    if (!activeExportJobId) return;
    setExportingFmt(fmt);
    try {
      const url = trinetraApi.getProcessingJobExportUrl(activeExportJobId, fmt);
      const res = await fetch(url);
      if (!res.ok) {
        throw new Error(`Export failed with HTTP ${res.status}`);
      }
      const blob = await res.blob();
      const blobUrl = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = blobUrl;
      a.download = `trinetra_job_${activeExportJobId}.${fmt}`;
      a.click();
      URL.revokeObjectURL(blobUrl);
      pushToast({
        tone: "success",
        title: `${fmt.toUpperCase()} report exported`,
        description: `Job ${truncateId(activeExportJobId)} downloaded.`,
      });
    } catch (err) {
      pushToast({
        tone: "error",
        title: "Export failed",
        description: err instanceof Error ? err.message : "Could not export report.",
      });
    } finally {
      setExportingFmt(null);
    }
  };

  const timelineOption = useMemo<EChartsOption | null>(() => {
    const items = timelineQ.data?.items ?? [];
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
          data: items.map((i) => i.review_required_events),
          itemStyle: { color: palette.warning },
        },
      ],
    };
  }, [timelineQ.data, palette]);

  const durationOption = useMemo<EChartsOption | null>(() => {
    const buckets = eventsAnalyticsQ.data?.duration_distribution ?? [];
    if (buckets.every((b) => b.count === 0)) return null;
    return {
      tooltip: { trigger: "axis" },
      grid: { left: 40, right: 16, top: 20, bottom: 28 },
      xAxis: {
        type: "category",
        data: buckets.map((b) => b.bucket),
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
          name: "Events",
          type: "bar",
          data: buckets.map((b) => b.count),
          itemStyle: { color: palette.primary, borderRadius: [3, 3, 0, 0] },
        },
      ],
    };
  }, [eventsAnalyticsQ.data, palette]);

  const confidenceOption = useMemo<EChartsOption | null>(() => {
    const buckets = eventsAnalyticsQ.data?.confidence_distribution ?? [];
    if (buckets.every((b) => b.count === 0)) return null;
    return {
      tooltip: { trigger: "axis" },
      grid: { left: 40, right: 16, top: 20, bottom: 28 },
      xAxis: {
        type: "category",
        data: buckets.map((b) => b.bucket),
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
          name: "Tracks",
          type: "bar",
          data: buckets.map((b) => b.count),
          itemStyle: { color: palette.success, borderRadius: [3, 3, 0, 0] },
        },
      ],
    };
  }, [eventsAnalyticsQ.data, palette]);

  const reviewBreakdownOption = useMemo<EChartsOption | null>(() => {
    const ev = eventsAnalyticsQ.data;
    if (!ev || ev.total_candidate_events === 0) return null;
    const unconfirmed = Math.max(
      0,
      ev.total_candidate_events - ev.historical_confirmations,
    );
    return {
      tooltip: { trigger: "item" },
      legend: {
        bottom: 0,
        textStyle: { color: palette.legendText, fontSize: 11 },
      },
      series: [
        {
          name: "Event Classification",
          type: "pie",
          radius: ["42%", "68%"],
          center: ["50%", "44%"],
          label: { show: false },
          data: [
            {
              name: "Non-Review Confirmed",
              value: ev.non_review_confirmations,
              itemStyle: { color: palette.success },
            },
            {
              name: "Review Required",
              value: ev.review_required_events,
              itemStyle: { color: palette.warning },
            },
            {
              name: "Unconfirmed Candidates",
              value: unconfirmed,
              itemStyle: { color: palette.neutral },
            },
          ],
        },
      ],
    };
  }, [eventsAnalyticsQ.data, palette]);

  const scopeLabel =
    startDate || endDate
      ? `Filtered: ${startDate || "start"} to ${endDate || "now"}`
      : "All recorded history";

  return (
    <div className="space-y-6">
      {/* Filter & Export Controls */}
      <Card>
        <CardContent className="p-4 flex flex-wrap items-end justify-between gap-4">
          <div className="flex flex-wrap items-end gap-3">
            <div>
              <label
                htmlFor="analytics-start-date"
                className="block text-[11px] font-medium text-muted-foreground mb-1"
              >
                Start Date (UTC)
              </label>
              <input
                id="analytics-start-date"
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="h-8 rounded-md border border-border bg-surface px-2.5 text-xs"
              />
            </div>
            <div>
              <label
                htmlFor="analytics-end-date"
                className="block text-[11px] font-medium text-muted-foreground mb-1"
              >
                End Date (UTC)
              </label>
              <input
                id="analytics-end-date"
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className="h-8 rounded-md border border-border bg-surface px-2.5 text-xs"
              />
            </div>
            {(startDate || endDate) && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  setStartDate("");
                  setEndDate("");
                }}
              >
                <FilterX className="h-3.5 w-3.5" aria-hidden="true" />
                Clear Dates
              </Button>
            )}
          </div>

          <div className="flex flex-wrap items-end gap-2">
            <div>
              <label
                htmlFor="export-job-select"
                className="block text-[11px] font-medium text-muted-foreground mb-1"
              >
                Export Job Report
              </label>
              <select
                id="export-job-select"
                value={activeExportJobId}
                onChange={(e) => setExportJobId(e.target.value)}
                className="h-8 rounded-md border border-border bg-surface px-2.5 text-xs min-w-[200px]"
              >
                {exportableJobs.length === 0 ? (
                  <option value="">No finished jobs</option>
                ) : (
                  exportableJobs.map((j) => (
                    <option key={j.id} value={j.id}>
                      {truncateId(j.id)} · {j.status} ({j.confirmed_events_count} conf.)
                    </option>
                  ))
                )}
              </select>
            </div>
            {(["json", "csv", "zip"] as const).map((fmt) => (
              <Button
                key={fmt}
                variant="outline"
                size="sm"
                disabled={!activeExportJobId || exportingFmt !== null}
                loading={exportingFmt === fmt}
                onClick={() => void handleExport(fmt)}
              >
                <Download className="h-3.5 w-3.5" aria-hidden="true" />
                {fmt.toUpperCase()}
              </Button>
            ))}
          </div>
        </CardContent>
      </Card>

      {dateRangeInvalid && (
        <ErrorState
          title="Invalid date range"
          message="End date cannot precede start date."
        />
      )}

      {/* Summary Strip */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Card>
          <CardContent className="p-4">
            <p className="text-xs text-muted-foreground">Avg Processing Throughput</p>
            <p className="mt-1 text-xl font-semibold tabular-nums">
              {processingQ.data?.average_processing_fps !== null &&
              processingQ.data?.average_processing_fps !== undefined
                ? `${processingQ.data.average_processing_fps.toFixed(2)} fps`
                : "Unavailable"}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <p className="text-xs text-muted-foreground">Avg Job Elapsed Time</p>
            <p className="mt-1 text-xl font-semibold tabular-nums">
              {processingQ.data?.average_elapsed_seconds !== null &&
              processingQ.data?.average_elapsed_seconds !== undefined
                ? formatDuration(processingQ.data.average_elapsed_seconds)
                : "Unavailable"}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <p className="text-xs text-muted-foreground">Frames Processed (Completed)</p>
            <p className="mt-1 text-xl font-semibold tabular-nums">
              {processingQ.data?.total_processed_frames_completed_jobs ?? 0}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <p className="text-xs text-muted-foreground">Post-Confirm Degraded</p>
            <p className="mt-1 text-xl font-semibold tabular-nums text-status-warning">
              {overviewQ.data?.post_confirmation_degraded_events ?? 0}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Detection Timeline</CardTitle>
            <span className="text-[11px] text-muted-foreground">{scopeLabel}</span>
          </CardHeader>
          <CardContent>
            <EChartBox
              option={timelineOption}
              loading={timelineQ.isLoading}
              error={timelineQ.isError ? timelineQ.error.message : null}
              onRetry={() => void timelineQ.refetch()}
              isEmpty={!timelineOption}
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Review vs Non-Review Breakdown</CardTitle>
            <span className="text-[11px] text-muted-foreground">{scopeLabel}</span>
          </CardHeader>
          <CardContent>
            <EChartBox
              option={reviewBreakdownOption}
              loading={eventsAnalyticsQ.isLoading}
              error={eventsAnalyticsQ.isError ? eventsAnalyticsQ.error.message : null}
              onRetry={() => void eventsAnalyticsQ.refetch()}
              isEmpty={!reviewBreakdownOption}
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Event Duration Distribution</CardTitle>
            <span className="text-[11px] text-muted-foreground">{scopeLabel}</span>
          </CardHeader>
          <CardContent>
            <EChartBox
              option={durationOption}
              loading={eventsAnalyticsQ.isLoading}
              error={eventsAnalyticsQ.isError ? eventsAnalyticsQ.error.message : null}
              onRetry={() => void eventsAnalyticsQ.refetch()}
              isEmpty={!durationOption}
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Heuristic Confidence Distribution</CardTitle>
            <span className="text-[11px] text-muted-foreground">
              Heuristic score (not probability) · {scopeLabel}
            </span>
          </CardHeader>
          <CardContent>
            <EChartBox
              option={confidenceOption}
              loading={eventsAnalyticsQ.isLoading}
              error={eventsAnalyticsQ.isError ? eventsAnalyticsQ.error.message : null}
              onRetry={() => void eventsAnalyticsQ.refetch()}
              isEmpty={!confidenceOption}
            />
          </CardContent>
        </Card>
      </div>

      {/* Per-Video Comparison Table */}
      <Card>
        <CardHeader>
          <CardTitle>Per-Video Comparison</CardTitle>
          <span className="text-xs text-muted-foreground tabular-nums">
            {videosQ.data?.count ?? 0} videos
          </span>
        </CardHeader>
        <CardContent className="p-0">
          {videosQ.isLoading ? (
            <div className="p-4">
              <TableSkeleton rows={3} />
            </div>
          ) : videosQ.isError ? (
            <div className="p-4">
              <ErrorState message={videosQ.error.message} />
            </div>
          ) : (videosQ.data?.items.length ?? 0) === 0 ? (
            <div className="p-4">
              <EmptyState
                title="No video analytics available"
                description="Upload and process videos to compare per-video candidate and confirmation counts."
              />
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="border-b border-border bg-surface-muted text-muted-foreground">
                  <tr>
                    <th className="px-4 py-2.5 font-medium">Video</th>
                    <th className="px-4 py-2.5 font-medium">Duration</th>
                    <th className="px-4 py-2.5 font-medium">Jobs</th>
                    <th className="px-4 py-2.5 font-medium">Candidate Tracks</th>
                    <th className="px-4 py-2.5 font-medium">Historical Confirmations</th>
                    <th className="px-4 py-2.5 font-medium">Review Required</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {videosQ.data!.items.map((row) => (
                    <tr key={row.video_id} className="hover:bg-surface-muted/50">
                      <td className="px-4 py-2.5 font-medium text-foreground">
                        {row.original_filename}
                      </td>
                      <td className="px-4 py-2.5 tabular-nums">
                        {formatDuration(row.duration_seconds)}
                      </td>
                      <td className="px-4 py-2.5 tabular-nums">{row.job_count}</td>
                      <td className="px-4 py-2.5 tabular-nums">
                        {row.candidate_event_count}
                      </td>
                      <td className="px-4 py-2.5 tabular-nums text-status-success font-medium">
                        {row.historical_confirmed_event_count}
                      </td>
                      <td className="px-4 py-2.5 tabular-nums text-status-warning font-medium">
                        {row.review_required_event_count}
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
  );
}
