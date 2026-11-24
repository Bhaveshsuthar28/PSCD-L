import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  ChevronLeft,
  ChevronRight,
  Columns2,
  Download,
  FileJson,
  FileSpreadsheet,
  FileText,
  Images,
  Maximize2,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import { resolveApiUrl, trinetraApi } from "@/api/client";
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
import { StatusBadge } from "@/components/ui/StatusBadge";
import { formatBytes, formatTimestampIso, truncateId } from "@/lib/utils";
import type { EvidenceRecord } from "@/types/api";

export function EvidencePage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const initialEventId = searchParams.get("eventId") ?? "";
  const initialEvidenceId = searchParams.get("evidenceId") ?? "";

  const [videoFilter, setVideoFilter] = useState("");
  const [eventFilter, setEventFilter] = useState(initialEventId);
  const [typeFilter, setTypeFilter] = useState("");
  const [page, setPage] = useState(1);

  const [activeIndex, setActiveIndex] = useState<number | null>(null);
  const [comparisonMode, setComparisonMode] = useState(false);
  const [zoom, setZoom] = useState(1);

  const videosQuery = useQuery({
    queryKey: ["videos", "ev-filter"],
    queryFn: ({ signal }) => trinetraApi.listVideos({ page: 1, page_size: 100 }, signal),
  });

  const evidenceQuery = useQuery({
    queryKey: ["evidence", videoFilter, eventFilter, typeFilter, page],
    queryFn: ({ signal }) =>
      trinetraApi.listEvidence(
        {
          video_id: videoFilter || undefined,
          event_id: eventFilter || undefined,
          evidence_type: typeFilter || undefined,
          page,
          page_size: 18,
        },
        signal,
      ),
  });

  const items = evidenceQuery.data?.items ?? [];

  useEffect(() => {
    if (initialEvidenceId && items.length > 0) {
      const idx = items.findIndex((i) => i.id === initialEvidenceId);
      if (idx >= 0) setActiveIndex(idx);
    }
  }, [initialEvidenceId, items]);

  const activeItem: EvidenceRecord | null =
    activeIndex !== null && activeIndex >= 0 && activeIndex < items.length
      ? items[activeIndex]
      : null;

  // Fetch sibling evidence items for comparison view when viewing an event's evidence
  const siblingQuery = useQuery({
    queryKey: ["event-evidence", activeItem?.event_id],
    enabled: Boolean(activeItem?.event_id && comparisonMode),
    queryFn: ({ signal }) =>
      trinetraApi.listEventEvidence(activeItem!.event_id!, signal),
  });

  const videoNames = useMemo(() => {
    const map = new Map<string, string>();
    for (const v of videosQuery.data?.items ?? []) {
      map.set(v.id, v.original_filename);
    }
    return map;
  }, [videosQuery.data]);

  const openModal = (idx: number) => {
    setZoom(1);
    setComparisonMode(false);
    setActiveIndex(idx);
  };

  const closeModal = () => {
    setActiveIndex(null);
    if (searchParams.has("evidenceId")) {
      setSearchParams((prev) => {
        const next = new URLSearchParams(prev);
        next.delete("evidenceId");
        return next;
      });
    }
  };

  const isImageType = (item: EvidenceRecord) =>
    item.mime_type.startsWith("image/");

  const siblingByType = useMemo(() => {
    const map = new Map<string, EvidenceRecord>();
    for (const s of siblingQuery.data?.items ?? []) {
      if (!map.has(s.evidence_type)) {
        map.set(s.evidence_type, s);
      }
    }
    return map;
  }, [siblingQuery.data]);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <select
            aria-label="Filter evidence by video"
            value={videoFilter}
            onChange={(e) => {
              setVideoFilter(e.target.value);
              setPage(1);
            }}
            className="h-8 rounded-md border border-border bg-surface px-2.5 text-xs"
          >
            <option value="">All videos</option>
            {(videosQuery.data?.items ?? []).map((v) => (
              <option key={v.id} value={v.id}>
                {v.original_filename}
              </option>
            ))}
          </select>

          <select
            aria-label="Filter by evidence type"
            value={typeFilter}
            onChange={(e) => {
              setTypeFilter(e.target.value);
              setPage(1);
            }}
            className="h-8 rounded-md border border-border bg-surface px-2.5 text-xs"
          >
            <option value="">All evidence types</option>
            <option value="annotated_frame">Annotated Frame</option>
            <option value="original_frame">Original Frame</option>
            <option value="probability_map">Probability Map</option>
            <option value="metadata">Event Metadata (JSON)</option>
            <option value="exported_report">Exported Report</option>
          </select>

          {eventFilter && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                setEventFilter("");
                setSearchParams({});
              }}
            >
              Event: {truncateId(eventFilter)} ×
            </Button>
          )}
        </div>

        <span className="text-xs text-muted-foreground tabular-nums">
          {evidenceQuery.data?.pagination.total_items ?? 0} artifacts
        </span>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Evidence Gallery</CardTitle>
          <span className="text-[11px] text-muted-foreground">
            Served via verified backend download endpoints
          </span>
        </CardHeader>
        <CardContent>
          {evidenceQuery.isLoading ? (
            <TableSkeleton rows={4} />
          ) : evidenceQuery.isError ? (
            <ErrorState
              message={evidenceQuery.error.message}
              onRetry={() => void evidenceQuery.refetch()}
            />
          ) : items.length === 0 ? (
            <EmptyState
              title="No evidence artifacts found"
              description="Evidence images and metadata files are recorded when persistent scene-change events are confirmed."
              icon={<Images className="h-5 w-5" />}
            />
          ) : (
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 md:grid-cols-3 xl:grid-cols-4">
              {items.map((item, idx) => (
                <div
                  key={item.id}
                  className="flex flex-col overflow-hidden rounded-lg border border-border bg-surface"
                >
                  <button
                    type="button"
                    onClick={() => openModal(idx)}
                    className="group relative flex h-40 w-full items-center justify-center bg-surface-muted overflow-hidden focus-visible:outline-2 focus-visible:outline-ring"
                  >
                    {isImageType(item) && item.download_url ? (
                      <img
                        src={resolveApiUrl(item.download_url)}
                        alt={item.original_filename}
                        loading="lazy"
                        className="h-full w-full object-cover"
                      />
                    ) : (
                      <div className="flex flex-col items-center gap-2 px-3 text-xs text-muted-foreground">
                        {item.mime_type.includes("csv") ||
                        item.original_filename.toLowerCase().endsWith(".csv") ? (
                          <FileSpreadsheet className="h-7 w-7 text-primary" aria-hidden="true" />
                        ) : item.mime_type.includes("json") ||
                          item.original_filename.toLowerCase().endsWith(".json") ? (
                          <FileJson className="h-7 w-7 text-primary" aria-hidden="true" />
                        ) : (
                          <FileText className="h-7 w-7 text-muted-foreground" aria-hidden="true" />
                        )}
                        <span className="max-w-full truncate font-medium text-foreground">
                          {item.original_filename}
                        </span>
                        <span className="rounded border border-border bg-surface px-1.5 py-0.5 font-mono text-[10px] uppercase">
                          {item.mime_type}
                        </span>
                      </div>
                    )}
                    <span className="absolute right-2 top-2 rounded bg-black/70 px-1.5 py-0.5 text-[10px] text-white opacity-0 transition-opacity group-hover:opacity-100">
                      <Maximize2 className="h-3 w-3 inline mr-1" />
                      View
                    </span>
                  </button>

                  <div className="flex flex-1 flex-col justify-between gap-2 p-3 text-xs">
                    <div className="space-y-1">
                      <div className="flex items-center justify-between gap-2">
                        <StatusBadge
                          tone={
                            item.evidence_type === "annotated_frame"
                              ? "success"
                              : item.evidence_type === "probability_map"
                                ? "info"
                                : "neutral"
                          }
                          label={item.evidence_type.replace("_", " ")}
                          icon="neutral"
                        />
                        <span className="font-mono text-[11px] tabular-nums text-muted-foreground">
                          {item.timestamp_seconds !== null
                            ? `${item.timestamp_seconds.toFixed(2)}s`
                            : "—"}
                        </span>
                      </div>
                      <p className="truncate font-medium text-foreground" title={item.original_filename}>
                        {item.original_filename}
                      </p>
                      <p className="truncate text-[11px] text-muted-foreground">
                        {item.video_id ? videoNames.get(item.video_id) ?? truncateId(item.video_id) : "Job report"} ·{" "}
                        {formatBytes(item.file_size_bytes)}
                      </p>
                    </div>

                    <div className="flex items-center justify-between gap-2 pt-1 border-t border-border">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => openModal(idx)}
                      >
                        Inspect
                      </Button>
                      {item.download_url && (
                        <a
                          href={resolveApiUrl(item.download_url)}
                          download={item.original_filename}
                        >
                          <Button variant="ghost" size="sm" aria-label={`Download ${item.original_filename}`}>
                            <Download className="h-3.5 w-3.5" aria-hidden="true" />
                          </Button>
                        </a>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          <PaginationControls
            pagination={evidenceQuery.data?.pagination}
            onPageChange={setPage}
          />
        </CardContent>
      </Card>

      {/* Lightbox & Comparison Dialog */}
      <Dialog
        open={Boolean(activeItem)}
        onClose={closeModal}
        title={activeItem?.original_filename ?? "Evidence Preview"}
        description={
          activeItem
            ? `${activeItem.evidence_type} · ${formatBytes(activeItem.file_size_bytes)} · ${formatTimestampIso(activeItem.created_at)}`
            : undefined
        }
        maxWidthClass="max-w-5xl"
      >
        {activeItem && (
          <div className="space-y-4 text-xs">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  disabled={activeIndex === null || activeIndex <= 0}
                  onClick={() => setActiveIndex((i) => (i !== null ? i - 1 : i))}
                >
                  <ChevronLeft className="h-3.5 w-3.5" aria-hidden="true" />
                  Previous
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  disabled={activeIndex === null || activeIndex >= items.length - 1}
                  onClick={() => setActiveIndex((i) => (i !== null ? i + 1 : i))}
                >
                  Next
                  <ChevronRight className="h-3.5 w-3.5" aria-hidden="true" />
                </Button>
                {activeItem.event_id && (
                  <Button
                    variant={comparisonMode ? "primary" : "outline"}
                    size="sm"
                    onClick={() => setComparisonMode((v) => !v)}
                  >
                    <Columns2 className="h-3.5 w-3.5" aria-hidden="true" />
                    {comparisonMode ? "Single View" : "Compare Event Frames"}
                  </Button>
                )}
              </div>

              {!comparisonMode && isImageType(activeItem) && (
                <div className="flex items-center gap-1.5">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setZoom((z) => Math.max(0.5, z - 0.25))}
                    aria-label="Zoom out"
                  >
                    <ZoomOut className="h-3.5 w-3.5" />
                  </Button>
                  <span className="w-12 text-center tabular-nums">
                    {Math.round(zoom * 100)}%
                  </span>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setZoom((z) => Math.min(3, z + 0.25))}
                    aria-label="Zoom in"
                  >
                    <ZoomIn className="h-3.5 w-3.5" />
                  </Button>
                  <Button variant="outline" size="sm" onClick={() => setZoom(1)}>
                    Fit
                  </Button>
                </div>
              )}
            </div>

            {comparisonMode ? (
              siblingQuery.isLoading ? (
                <TableSkeleton rows={3} />
              ) : (
                <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
                  {(["original_frame", "annotated_frame", "probability_map"] as const).map(
                    (etype) => {
                      const sibling = siblingByType.get(etype);
                      return (
                        <div key={etype} className="space-y-1.5">
                          <p className="font-medium capitalize text-foreground">
                            {etype.replace("_", " ")}
                          </p>
                          {sibling ? (
                            <EvidenceOverlayViewer
                              item={sibling}
                              showBboxToggle={false}
                              maxHeightClass="max-h-[260px]"
                            />
                          ) : (
                            <div className="flex h-44 items-center justify-center rounded-md border border-dashed border-border bg-surface-muted text-muted-foreground">
                              Not generated for this event
                            </div>
                          )}
                        </div>
                      );
                    },
                  )}
                </div>
              )
            ) : isImageType(activeItem) ? (
              <div className="overflow-auto max-h-[60vh]">
                <div
                  style={{
                    transform: `scale(${zoom})`,
                    transformOrigin: "top center",
                  }}
                >
                  <EvidenceOverlayViewer key={activeItem.id} item={activeItem} />
                </div>
              </div>
            ) : (
              <EvidenceOverlayViewer key={activeItem.id} item={activeItem} />
            )}

            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 rounded-md border border-border bg-surface-muted p-3">
              <div>
                <span className="text-muted-foreground">Evidence ID</span>
                <p className="font-mono text-[11px] truncate">{activeItem.id}</p>
              </div>
              <div>
                <span className="text-muted-foreground">Dimensions</span>
                <p className="tabular-nums">
                  {activeItem.width && activeItem.height
                    ? `${activeItem.width}×${activeItem.height}`
                    : "Unavailable"}
                </p>
              </div>
              <div>
                <span className="text-muted-foreground">SHA-256</span>
                <p className="font-mono text-[11px] truncate" title={activeItem.sha256}>
                  {truncateId(activeItem.sha256, 16)}
                </p>
              </div>
              <div>
                <span className="text-muted-foreground">Bounding Box</span>
                <p className="font-mono text-[11px]">
                  {activeItem.bbox_xywh ? JSON.stringify(activeItem.bbox_xywh) : "—"}
                </p>
              </div>
            </div>
          </div>
        )}
      </Dialog>
    </div>
  );
}
