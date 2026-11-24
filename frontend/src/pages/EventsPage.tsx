import { useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, Eye, FilterX, Info } from "lucide-react";
import { trinetraApi } from "@/api/client";
import { EvidenceOverlayViewer } from "@/components/evidence/EvidenceOverlayViewer";
import { Button } from "@/components/ui/Button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { Dialog } from "@/components/ui/Dialog";
import { PaginationControls } from "@/components/ui/PaginationControls";
import {
  EmptyState,
  ErrorState,
  TableSkeleton,
} from "@/components/ui/StateFeedback";
import { EventLifecycleBadge, StatusBadge } from "@/components/ui/StatusBadge";
import { formatDuration, formatNumber, truncateId } from "@/lib/utils";
import type { EvidenceRecord } from "@/types/api";

export function EventsPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const selectedEventId = searchParams.get("eventId");

  const [videoFilter, setVideoFilter] = useState(
    searchParams.get("videoId") ?? "",
  );
  const [jobFilter, setJobFilter] = useState(searchParams.get("jobId") ?? "");
  const [stateFilter, setStateFilter] = useState("");
  const [confirmedFilter, setConfirmedFilter] = useState<string>("");
  const [reviewFilter, setReviewFilter] = useState<string>("");
  const [degradedFilter, setDegradedFilter] = useState<string>("");
  const [minConfidence, setMinConfidence] = useState("");
  const [minDuration, setMinDuration] = useState("");
  const [startTime, setStartTime] = useState("");
  const [endTime, setEndTime] = useState("");
  const [sortBy, setSortBy] = useState("created_at");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");
  const [page, setPage] = useState(1);

  const [activeEvidenceTab, setActiveEvidenceTab] = useState<string>("annotated_frame");

  const videosQuery = useQuery({
    queryKey: ["videos", "filter-list"],
    queryFn: ({ signal }) => trinetraApi.listVideos({ page: 1, page_size: 100 }, signal),
  });

  const jobsQuery = useQuery({
    queryKey: ["jobs", "filter-list"],
    queryFn: ({ signal }) =>
      trinetraApi.listProcessingJobs({ page: 1, page_size: 100 }, signal),
  });

  const eventsQuery = useQuery({
    queryKey: [
      "events",
      videoFilter,
      jobFilter,
      stateFilter,
      confirmedFilter,
      reviewFilter,
      degradedFilter,
      minConfidence,
      minDuration,
      startTime,
      endTime,
      sortBy,
      sortOrder,
      page,
    ],
    queryFn: ({ signal }) =>
      trinetraApi.listEvents(
        {
          video_id: videoFilter || undefined,
          job_id: jobFilter || undefined,
          event_state: stateFilter || undefined,
          was_confirmed:
            confirmedFilter === ""
              ? undefined
              : confirmedFilter === "true",
          review_required:
            reviewFilter === "" ? undefined : reviewFilter === "true",
          post_confirmation_degraded:
            degradedFilter === "" ? undefined : degradedFilter === "true",
          min_confidence: minConfidence !== "" ? Number(minConfidence) : undefined,
          min_duration: minDuration !== "" ? Number(minDuration) : undefined,
          start_time_seconds: startTime !== "" ? Number(startTime) : undefined,
          end_time_seconds: endTime !== "" ? Number(endTime) : undefined,
          sort_by: sortBy,
          sort_order: sortOrder,
          page,
          page_size: 15,
        },
        signal,
      ),
  });

  const detailQuery = useQuery({
    queryKey: ["event-detail", selectedEventId],
    enabled: Boolean(selectedEventId),
    queryFn: ({ signal }) => trinetraApi.getEventDetail(selectedEventId!, signal),
  });

  const videoNameMap = useMemo(() => {
    const map = new Map<string, string>();
    for (const v of videosQuery.data?.items ?? []) {
      map.set(v.id, v.original_filename);
    }
    return map;
  }, [videosQuery.data]);

  const openEvent = (eventId: string) => {
    setSearchParams((prev) => {
      const next = new URLSearchParams(prev);
      next.set("eventId", eventId);
      return next;
    });
  };

  const closeEvent = () => {
    setSearchParams((prev) => {
      const next = new URLSearchParams(prev);
      next.delete("eventId");
      return next;
    });
  };

  const resetFilters = () => {
    setVideoFilter("");
    setJobFilter("");
    setStateFilter("");
    setConfirmedFilter("");
    setReviewFilter("");
    setDegradedFilter("");
    setMinConfidence("");
    setMinDuration("");
    setStartTime("");
    setEndTime("");
    setPage(1);
    setSearchParams({});
  };

  const detail = detailQuery.data;
  const imageEvidenceByType = useMemo(() => {
    const map = new Map<string, EvidenceRecord>();
    for (const item of detail?.evidence_items ?? []) {
      if (!map.has(item.evidence_type)) {
        map.set(item.evidence_type, item);
      }
    }
    return map;
  }, [detail]);

  const activeEvidenceItem =
    imageEvidenceByType.get(activeEvidenceTab) ??
    detail?.evidence_items.find((i) => i.evidence_type !== "metadata");

  return (
    <div className="space-y-4">
      {/* Filter Bar */}
      <Card>
        <CardContent className="p-4 space-y-3">
          <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-6">
            <select
              aria-label="Filter by video"
              value={videoFilter}
              onChange={(e) => {
                setVideoFilter(e.target.value);
                setPage(1);
              }}
              className="h-8 rounded-md border border-border bg-surface px-2 text-xs"
            >
              <option value="">All videos</option>
              {(videosQuery.data?.items ?? []).map((v) => (
                <option key={v.id} value={v.id}>
                  {v.original_filename}
                </option>
              ))}
            </select>

            <select
              aria-label="Filter by processing job"
              value={jobFilter}
              onChange={(e) => {
                setJobFilter(e.target.value);
                setPage(1);
              }}
              className="h-8 rounded-md border border-border bg-surface px-2 text-xs"
            >
              <option value="">All jobs</option>
              {(jobsQuery.data?.items ?? []).map((j) => (
                <option key={j.id} value={j.id}>
                  {truncateId(j.id)} ({j.status})
                </option>
              ))}
            </select>

            <select
              aria-label="Filter by event state"
              value={stateFilter}
              onChange={(e) => {
                setStateFilter(e.target.value);
                setPage(1);
              }}
              className="h-8 rounded-md border border-border bg-surface px-2 text-xs"
            >
              <option value="">All lifecycle states</option>
              <option value="CANDIDATE">CANDIDATE</option>
              <option value="OBSERVING">OBSERVING</option>
              <option value="SETTLING">SETTLING</option>
              <option value="CONFIRMED">CONFIRMED</option>
              <option value="COMPLETED">COMPLETED</option>
            </select>

            <select
              aria-label="Filter by confirmation status"
              value={confirmedFilter}
              onChange={(e) => {
                setConfirmedFilter(e.target.value);
                setPage(1);
              }}
              className="h-8 rounded-md border border-border bg-surface px-2 text-xs"
            >
              <option value="">Confirmation: All</option>
              <option value="true">Historically Confirmed</option>
              <option value="false">Never Confirmed</option>
            </select>

            <select
              aria-label="Filter by review requirement"
              value={reviewFilter}
              onChange={(e) => {
                setReviewFilter(e.target.value);
                setPage(1);
              }}
              className="h-8 rounded-md border border-border bg-surface px-2 text-xs"
            >
              <option value="">Review: All</option>
              <option value="true">Review Required</option>
              <option value="false">No Review Flag</option>
            </select>

            <select
              aria-label="Filter by post-confirmation degradation"
              value={degradedFilter}
              onChange={(e) => {
                setDegradedFilter(e.target.value);
                setPage(1);
              }}
              className="h-8 rounded-md border border-border bg-surface px-2 text-xs"
            >
              <option value="">Degradation: All</option>
              <option value="true">Degraded</option>
              <option value="false">Not Degraded</option>
            </select>
          </div>

          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex flex-wrap items-center gap-2">
              <input
                type="number"
                min="0"
                max="1"
                step="0.05"
                placeholder="Min conf (0–1)"
                aria-label="Minimum heuristic confidence"
                value={minConfidence}
                onChange={(e) => {
                  setMinConfidence(e.target.value);
                  setPage(1);
                }}
                className="h-8 w-32 rounded-md border border-border bg-surface px-2 text-xs tabular-nums"
              />
              <input
                type="number"
                min="0"
                step="1"
                placeholder="Min dur (s)"
                aria-label="Minimum duration seconds"
                value={minDuration}
                onChange={(e) => {
                  setMinDuration(e.target.value);
                  setPage(1);
                }}
                className="h-8 w-28 rounded-md border border-border bg-surface px-2 text-xs tabular-nums"
              />
              <input
                type="number"
                min="0"
                step="1"
                placeholder="From t (s)"
                aria-label="Window start seconds"
                value={startTime}
                onChange={(e) => {
                  setStartTime(e.target.value);
                  setPage(1);
                }}
                className="h-8 w-24 rounded-md border border-border bg-surface px-2 text-xs tabular-nums"
              />
              <input
                type="number"
                min="0"
                step="1"
                placeholder="To t (s)"
                aria-label="Window end seconds"
                value={endTime}
                onChange={(e) => {
                  setEndTime(e.target.value);
                  setPage(1);
                }}
                className="h-8 w-24 rounded-md border border-border bg-surface px-2 text-xs tabular-nums"
              />
            </div>

            <div className="flex items-center gap-2">
              <select
                aria-label="Sort events"
                value={`${sortBy}:${sortOrder}`}
                onChange={(e) => {
                  const [col, dir] = e.target.value.split(":");
                  setSortBy(col);
                  setSortOrder(dir as "asc" | "desc");
                  setPage(1);
                }}
                className="h-8 rounded-md border border-border bg-surface px-2 text-xs"
              >
                <option value="created_at:desc">Newest first</option>
                <option value="confidence:desc">Highest confidence</option>
                <option value="duration_seconds:desc">Longest duration</option>
                <option value="first_seen_seconds:asc">Earliest timestamp</option>
                <option value="track_id:asc">Track ID</option>
              </select>
              <Button variant="outline" size="sm" onClick={resetFilters}>
                <FilterX className="h-3.5 w-3.5" aria-hidden="true" />
                Reset
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Events Table */}
      <Card>
        <CardHeader>
          <CardTitle>Scene-Change Events</CardTitle>
          <span className="text-xs text-muted-foreground tabular-nums">
            {eventsQuery.data?.pagination.total_items ?? 0} matching tracks
          </span>
        </CardHeader>
        <CardContent className="p-0">
          {eventsQuery.isLoading ? (
            <div className="p-4">
              <TableSkeleton rows={6} />
            </div>
          ) : eventsQuery.isError ? (
            <div className="p-4">
              <ErrorState
                message={eventsQuery.error.message}
                onRetry={() => void eventsQuery.refetch()}
              />
            </div>
          ) : (eventsQuery.data?.items.length ?? 0) === 0 ? (
            <div className="p-4">
              <EmptyState
                title="No matching events"
                description="Adjust filter criteria or process a surveillance video to populate scene-change events."
              />
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="border-b border-border bg-surface-muted text-muted-foreground">
                  <tr>
                    <th className="px-4 py-2.5 font-medium">Event / Track</th>
                    <th className="px-4 py-2.5 font-medium">Video</th>
                    <th className="px-4 py-2.5 font-medium">Window</th>
                    <th className="px-4 py-2.5 font-medium">Duration</th>
                    <th className="px-4 py-2.5 font-medium">Lifecycle</th>
                    <th className="px-4 py-2.5 font-medium">Confirmation / Review</th>
                    <th className="px-4 py-2.5 font-medium">Heuristic Conf.</th>
                    <th className="px-4 py-2.5 font-medium text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {eventsQuery.data!.items.map((ev) => (
                    <tr key={ev.id} className="hover:bg-surface-muted/50">
                      <td className="px-4 py-2.5">
                        <div className="font-medium text-foreground tabular-nums">
                          Track #{ev.track_id}
                        </div>
                        <div className="font-mono text-[11px] text-muted-foreground">
                          {truncateId(ev.id)}
                        </div>
                      </td>
                      <td className="px-4 py-2.5 max-w-[180px] truncate">
                        {videoNameMap.get(ev.video_id) ?? truncateId(ev.video_id)}
                      </td>
                      <td className="px-4 py-2.5 tabular-nums">
                        {ev.first_seen_seconds.toFixed(2)}s → {ev.last_seen_seconds.toFixed(2)}s
                      </td>
                      <td className="px-4 py-2.5 tabular-nums">
                        {formatDuration(ev.duration_seconds)}
                      </td>
                      <td className="px-4 py-2.5">
                        <EventLifecycleBadge state={ev.event_state} />
                      </td>
                      <td className="px-4 py-2.5">
                        <div className="flex flex-wrap gap-1">
                          {ev.was_confirmed ? (
                            <StatusBadge
                              tone="success"
                              label={`Confirmed @ ${ev.confirmation_seconds?.toFixed(1) ?? "?"}s`}
                              icon="check"
                            />
                          ) : (
                            <StatusBadge tone="neutral" label="Not Confirmed" icon="neutral" />
                          )}
                          {ev.review_required && (
                            <StatusBadge
                              tone="warning"
                              label="Review Required"
                              icon="warn"
                            />
                          )}
                          {ev.post_confirmation_degraded && (
                            <StatusBadge
                              tone="danger"
                              label="Degraded"
                              icon="warn"
                            />
                          )}
                        </div>
                      </td>
                      <td className="px-4 py-2.5 tabular-nums font-medium">
                        {ev.confidence.toFixed(4)}
                      </td>
                      <td className="px-4 py-2.5 text-right">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => openEvent(ev.id)}
                        >
                          <Eye className="h-3.5 w-3.5" aria-hidden="true" />
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
            pagination={eventsQuery.data?.pagination}
            onPageChange={setPage}
          />
        </CardContent>
      </Card>

      {/* Event Detail Modal / Drawer */}
      <Dialog
        open={Boolean(selectedEventId)}
        onClose={closeEvent}
        title={detail ? `Persistent Scene-Change Track #${detail.track_id}` : "Event Detail"}
        description={detail ? `Event ID: ${detail.id}` : undefined}
        maxWidthClass="max-w-4xl"
      >
        {detailQuery.isLoading ? (
          <TableSkeleton rows={5} />
        ) : detailQuery.isError ? (
          <ErrorState
            message={detailQuery.error.message}
            onRetry={() => void detailQuery.refetch()}
          />
        ) : detail ? (
          <div className="space-y-5 text-xs">
            {/* Scientific Caveat */}
            <div className="flex items-start gap-2 rounded-md border border-border bg-surface-muted p-3 text-muted-foreground">
              <Info className="mt-0.5 h-4 w-4 shrink-0 text-primary" aria-hidden="true" />
              <div>
                <p className="font-medium text-foreground">{detail.description}</p>
                <p className="mt-0.5">
                  <strong>Heuristic confidence ({detail.confidence.toFixed(4)})</strong> is a persistence and stability score (<span className="font-mono">{detail.confidence_type}</span>), not a calibrated probability. Motion is settling evidence and does not establish object identity or ownership.
                </p>
              </div>
            </div>

            {detail.review_required && (
              <div className="flex items-start gap-2 rounded-md border border-status-warning-border bg-status-warning-bg p-3 text-status-warning">
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
                <div>
                  <p className="font-medium">
                    Human Review Required (Post-Confirmation Degradation)
                  </p>
                  <p className="mt-0.5">
                    Degradation first observed at{" "}
                    {detail.degradation_first_seen_seconds !== null
                      ? `${detail.degradation_first_seen_seconds.toFixed(2)}s`
                      : "unknown timestamp"}
                    . Reasons:{" "}
                    {detail.post_confirmation_degradation_reasons.join(", ") || "unspecified"}.
                  </p>
                </div>
              </div>
            )}

            {/* Metrics Grid */}
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 rounded-md border border-border bg-surface p-3">
              <div>
                <span className="text-muted-foreground">Lifecycle State</span>
                <div className="mt-1">
                  <EventLifecycleBadge state={detail.event_state} />
                </div>
              </div>
              <div>
                <span className="text-muted-foreground">Confirmation Time</span>
                <p className="mt-1 font-medium tabular-nums">
                  {detail.confirmation_seconds !== null
                    ? `${detail.confirmation_seconds.toFixed(3)}s`
                    : "Never confirmed"}
                </p>
              </div>
              <div>
                <span className="text-muted-foreground">Observations / Misses</span>
                <p className="mt-1 font-medium tabular-nums">
                  {detail.observations} obs / {detail.total_missed_observations} total misses
                </p>
              </div>
              <div>
                <span className="text-muted-foreground">Detection Consistency</span>
                <p className="mt-1 font-medium tabular-nums">
                  {(detail.detection_consistency * 100).toFixed(1)}%
                </p>
              </div>
              <div>
                <span className="text-muted-foreground">Center Deviation</span>
                <p className="mt-1 font-medium tabular-nums">
                  {detail.center_deviation_pixels.toFixed(2)} px
                </p>
              </div>
              <div>
                <span className="text-muted-foreground">BBox IoU Stability</span>
                <p className="mt-1 font-medium tabular-nums">
                  {detail.bbox_stability.toFixed(4)}
                </p>
              </div>
              <div>
                <span className="text-muted-foreground">Full Motion (med / p90)</span>
                <p className="mt-1 font-medium tabular-nums">
                  {formatNumber(detail.motion_median_ratio, 3)} /{" "}
                  {formatNumber(detail.motion_p90_ratio, 3)} ({detail.motion_samples} samples)
                </p>
              </div>
              <div>
                <span className="text-muted-foreground">Recent Settling (med / p90)</span>
                <p className="mt-1 font-medium tabular-nums">
                  {formatNumber(detail.settling_motion_median_ratio, 3)} /{" "}
                  {formatNumber(detail.settling_motion_p90_ratio, 3)} ({detail.settling_motion_samples} samples)
                </p>
              </div>
            </div>

            {/* State History */}
            <div>
              <h3 className="mb-2 font-semibold text-foreground">
                Monotonic State History
              </h3>
              <div className="flex flex-wrap items-center gap-2">
                {detail.state_history.map((step, idx) => (
                  <div
                    key={`${step.state}-${idx}`}
                    className="flex items-center gap-1.5 rounded border border-border bg-surface-muted px-2.5 py-1"
                  >
                    <span className="font-medium text-foreground">{step.state}</span>
                    <span className="font-mono text-[11px] text-muted-foreground tabular-nums">
                      @ {step.timestamp_seconds.toFixed(2)}s
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Evidence Gallery */}
            <div className="space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h3 className="font-semibold text-foreground">
                  Generated Evidence ({detail.evidence_items.length})
                </h3>
                <div className="flex flex-wrap gap-1.5">
                  {["annotated_frame", "original_frame", "probability_map", "metadata"].map((etype) => {
                    const available = imageEvidenceByType.has(etype);
                    return (
                      <Button
                        key={etype}
                        size="sm"
                        variant={activeEvidenceTab === etype ? "primary" : "outline"}
                        disabled={!available}
                        onClick={() => setActiveEvidenceTab(etype)}
                      >
                        {etype === "metadata" ? "metadata (JSON)" : etype.replace("_", " ")}
                      </Button>
                    );
                  })}
                </div>
              </div>

              {activeEvidenceItem ? (
                <EvidenceOverlayViewer key={activeEvidenceItem.id} item={activeEvidenceItem} />
              ) : (
                <EmptyState
                  title="No visual evidence for this track"
                  description="Evidence frames are saved only when a track reaches confirmation."
                />
              )}
            </div>

            <div className="flex flex-wrap items-center justify-between gap-2 border-t border-border pt-3">
              <div className="flex gap-2">
                <Link to={`/processing?jobId=${detail.job_id}`} onClick={closeEvent}>
                  <Button variant="outline" size="sm">
                    Open Processing Job
                  </Button>
                </Link>
                <Link to={`/evidence?eventId=${detail.id}`} onClick={closeEvent}>
                  <Button variant="outline" size="sm">
                    Open in Evidence Viewer
                  </Button>
                </Link>
              </div>
              <Button onClick={closeEvent}>Close</Button>
            </div>
          </div>
        ) : null}
      </Dialog>
    </div>
  );
}
