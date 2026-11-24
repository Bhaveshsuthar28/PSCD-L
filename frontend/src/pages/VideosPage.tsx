import { useMemo, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Eye,
  Film,
  PlayCircle,
  Search,
  Trash2,
  Upload,
  X,
} from "lucide-react";
import {
  TrinetraApiError,
  trinetraApi,
  uploadVideoWithProgress,
  type UploadTaskHandle,
} from "@/api/client";
import { Button } from "@/components/ui/Button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { Dialog } from "@/components/ui/Dialog";
import { PaginationControls } from "@/components/ui/PaginationControls";
import {
  EmptyState,
  ErrorState,
  TableSkeleton,
} from "@/components/ui/StateFeedback";
import { JobStatusBadge, StatusBadge } from "@/components/ui/StatusBadge";
import { useToast } from "@/components/ui/Toast";
import { VideoPlayer } from "@/components/video/VideoPlayer";
import {
  formatBytes,
  formatDuration,
  formatTimestampIso,
  truncateId,
} from "@/lib/utils";
import type { VideoRecord } from "@/types/api";

const ACCEPTED_EXTENSIONS = [".avi", ".mp4", ".mov", ".mkv"];
const MAX_UPLOAD_DISPLAY_MB = 500;

export function VideosPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { pushToast } = useToast();

  const [page, setPage] = useState(1);
  const [sortBy, setSortBy] = useState("created_at");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");
  const [search, setSearch] = useState("");
  const [formatFilter, setFormatFilter] = useState("");

  // Upload dialog state
  const [uploadOpen, setUploadOpen] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [duplicateVideoId, setDuplicateVideoId] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const uploadHandleRef = useRef<UploadTaskHandle | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Detail & Delete dialog state
  const [detailVideo, setDetailVideo] = useState<VideoRecord | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<VideoRecord | null>(null);

  const videosQuery = useQuery({
    queryKey: ["videos", page, sortBy, sortOrder],
    queryFn: ({ signal }) =>
      trinetraApi.listVideos(
        { page, page_size: 15, sort_by: sortBy, sort_order: sortOrder },
        signal,
      ),
  });

  const videoStatsQuery = useQuery({
    queryKey: ["analytics-videos"],
    queryFn: ({ signal }) => trinetraApi.getAnalyticsVideos(signal),
  });

  const detailJobsQuery = useQuery({
    queryKey: ["video-detail-jobs", detailVideo?.id],
    enabled: Boolean(detailVideo),
    queryFn: ({ signal }) =>
      trinetraApi.listProcessingJobs(
        { video_id: detailVideo!.id, page: 1, page_size: 10 },
        signal,
      ),
  });

  const detailEventsQuery = useQuery({
    queryKey: ["video-detail-events", detailVideo?.id],
    enabled: Boolean(detailVideo),
    queryFn: ({ signal }) =>
      trinetraApi.listVideoEvents(
        detailVideo!.id,
        { page: 1, page_size: 10 },
        signal,
      ),
  });

  const deleteMutation = useMutation({
    mutationFn: (videoId: string) => trinetraApi.deleteVideo(videoId),
    onSuccess: (res) => {
      pushToast({
        tone: "success",
        title: "Video removed",
        description: res.message,
      });
      setDeleteTarget(null);
      if (detailVideo?.id === res.video_id) {
        setDetailVideo(null);
      }
      void queryClient.invalidateQueries({ queryKey: ["videos"] });
      void queryClient.invalidateQueries({ queryKey: ["analytics-videos"] });
      void queryClient.invalidateQueries({ queryKey: ["analytics-overview"] });
    },
    onError: (err) => {
      pushToast({
        tone: "error",
        title: "Deletion blocked",
        description: err instanceof Error ? err.message : "Could not delete video.",
      });
    },
  });

  const statsMap = useMemo(() => {
    const map = new Map<
      string,
      { jobCount: number; candidateCount: number; confirmedCount: number }
    >();
    for (const item of videoStatsQuery.data?.items ?? []) {
      map.set(item.video_id, {
        jobCount: item.job_count,
        candidateCount: item.candidate_event_count,
        confirmedCount: item.historical_confirmed_event_count,
      });
    }
    return map;
  }, [videoStatsQuery.data]);

  const filteredVideos = useMemo(() => {
    const items = videosQuery.data?.items ?? [];
    const term = search.trim().toLowerCase();
    return items.filter((v) => {
      if (formatFilter && v.format.toLowerCase() !== formatFilter) return false;
      if (term && !v.original_filename.toLowerCase().includes(term) && !v.id.includes(term)) {
        return false;
      }
      return true;
    });
  }, [videosQuery.data, search, formatFilter]);

  const validateAndSelectFile = (file: File | null) => {
    setUploadError(null);
    setDuplicateVideoId(null);
    setUploadProgress(0);
    if (!file) {
      setSelectedFile(null);
      return;
    }
    if (file.size === 0) {
      setUploadError("Selected file is empty (0 bytes).");
      setSelectedFile(file);
      return;
    }
    const lower = file.name.toLowerCase();
    const validExt = ACCEPTED_EXTENSIONS.some((ext) => lower.endsWith(ext));
    if (!validExt) {
      setUploadError(
        `Unsupported extension. Allowed formats: ${ACCEPTED_EXTENSIONS.join(", ")}`,
      );
      setSelectedFile(file);
      return;
    }
    if (file.size > MAX_UPLOAD_DISPLAY_MB * 1024 * 1024) {
      setUploadError(`File exceeds ${MAX_UPLOAD_DISPLAY_MB} MB upload limit.`);
      setSelectedFile(file);
      return;
    }
    setSelectedFile(file);
  };

  const handleUploadSubmit = async () => {
    if (!selectedFile) {
      setUploadError("Please select a surveillance video file first.");
      return;
    }
    setUploadError(null);
    setDuplicateVideoId(null);
    setUploading(true);
    setUploadProgress(0);

    const handle = uploadVideoWithProgress(selectedFile, (pct) => {
      setUploadProgress(pct);
    });
    uploadHandleRef.current = handle;

    try {
      const created = await handle.promise;
      pushToast({
        tone: "success",
        title: "Video uploaded",
        description: `${created.original_filename} (${created.width}×${created.height}, ${formatDuration(created.duration_seconds)})`,
      });
      setUploadOpen(false);
      setSelectedFile(null);
      setUploadProgress(0);
      void queryClient.invalidateQueries({ queryKey: ["videos"] });
      void queryClient.invalidateQueries({ queryKey: ["analytics-videos"] });
      void queryClient.invalidateQueries({ queryKey: ["analytics-overview"] });
    } catch (err) {
      if (err instanceof TrinetraApiError) {
        setUploadError(`${err.code}: ${err.message}`);
        const details = err.details as { existing_video_id?: string } | null;
        if (err.code === "DUPLICATE_VIDEO" && details?.existing_video_id) {
          setDuplicateVideoId(details.existing_video_id);
        }
      } else {
        setUploadError(err instanceof Error ? err.message : "Upload failed.");
      }
    } finally {
      setUploading(false);
      uploadHandleRef.current = null;
    }
  };

  const handleCancelUpload = () => {
    uploadHandleRef.current?.abort();
  };

  const closeUploadDialog = () => {
    if (uploading) {
      handleCancelUpload();
    }
    setUploadOpen(false);
    setUploadError(null);
    setDuplicateVideoId(null);
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2 flex-1 min-w-[240px]">
          <div className="relative flex-1 max-w-xs">
            <Search
              className="pointer-events-none absolute left-2.5 top-2.5 h-3.5 w-3.5 text-muted-foreground"
              aria-hidden="true"
            />
            <input
              type="search"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Filter filename or ID…"
              aria-label="Filter videos"
              className="h-8 w-full rounded-md border border-border bg-surface pl-8 pr-3 text-xs text-foreground placeholder:text-muted-foreground focus-visible:outline-2 focus-visible:outline-ring"
            />
          </div>

          <select
            aria-label="Filter by format"
            value={formatFilter}
            onChange={(e) => setFormatFilter(e.target.value)}
            className="h-8 rounded-md border border-border bg-surface px-2.5 text-xs text-foreground"
          >
            <option value="">All formats</option>
            <option value="avi">AVI</option>
            <option value="mp4">MP4</option>
            <option value="mov">MOV</option>
            <option value="mkv">MKV</option>
          </select>

          <select
            aria-label="Sort videos"
            value={`${sortBy}:${sortOrder}`}
            onChange={(e) => {
              const [col, dir] = e.target.value.split(":");
              setSortBy(col);
              setSortOrder(dir as "asc" | "desc");
              setPage(1);
            }}
            className="h-8 rounded-md border border-border bg-surface px-2.5 text-xs text-foreground"
          >
            <option value="created_at:desc">Newest first</option>
            <option value="created_at:asc">Oldest first</option>
            <option value="original_filename:asc">Name (A–Z)</option>
            <option value="duration_seconds:desc">Longest duration</option>
            <option value="file_size_bytes:desc">Largest file</option>
          </select>
        </div>

        <Button
          onClick={() => {
            setSelectedFile(null);
            setUploadError(null);
            setDuplicateVideoId(null);
            setUploadProgress(0);
            setUploadOpen(true);
          }}
        >
          <Upload className="h-4 w-4" aria-hidden="true" />
          Upload Video
        </Button>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Uploaded Surveillance Videos</CardTitle>
          <span className="text-xs text-muted-foreground tabular-nums">
            {videosQuery.data?.pagination.total_items ?? 0} videos
          </span>
        </CardHeader>
        <CardContent className="p-0">
          {videosQuery.isLoading ? (
            <div className="p-4">
              <TableSkeleton rows={5} />
            </div>
          ) : videosQuery.isError ? (
            <div className="p-4">
              <ErrorState
                message={videosQuery.error.message}
                onRetry={() => void videosQuery.refetch()}
              />
            </div>
          ) : filteredVideos.length === 0 ? (
            <div className="p-4">
              <EmptyState
                title="No surveillance videos found"
                description="Upload an .avi, .mp4, .mov, or .mkv video to run scene-change analysis."
                icon={<Film className="h-5 w-5" />}
                action={
                  <Button size="sm" onClick={() => setUploadOpen(true)}>
                    <Upload className="h-3.5 w-3.5" aria-hidden="true" />
                    Upload Video
                  </Button>
                }
              />
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="border-b border-border bg-surface-muted text-muted-foreground">
                  <tr>
                    <th className="px-4 py-2.5 font-medium">Video Name</th>
                    <th className="px-4 py-2.5 font-medium">Resolution</th>
                    <th className="px-4 py-2.5 font-medium">FPS</th>
                    <th className="px-4 py-2.5 font-medium">Duration</th>
                    <th className="px-4 py-2.5 font-medium">Size</th>
                    <th className="px-4 py-2.5 font-medium">Uploaded</th>
                    <th className="px-4 py-2.5 font-medium">Jobs / Events</th>
                    <th className="px-4 py-2.5 font-medium text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {filteredVideos.map((video) => {
                    const stats = statsMap.get(video.id);
                    return (
                      <tr key={video.id} className="hover:bg-surface-muted/50">
                        <td className="px-4 py-2.5">
                          <div className="font-medium text-foreground max-w-[240px] truncate" title={video.original_filename}>
                            {video.original_filename}
                          </div>
                          <div className="font-mono text-[11px] text-muted-foreground">
                            {truncateId(video.id)} · {video.format.toUpperCase()}
                            {video.codec ? ` (${video.codec})` : ""}
                          </div>
                        </td>
                        <td className="px-4 py-2.5 tabular-nums">
                          {video.width}×{video.height}
                        </td>
                        <td className="px-4 py-2.5 tabular-nums">
                          {video.fps.toFixed(2)}
                        </td>
                        <td className="px-4 py-2.5 tabular-nums">
                          {formatDuration(video.duration_seconds)}
                        </td>
                        <td className="px-4 py-2.5 tabular-nums">
                          {formatBytes(video.file_size_bytes)}
                        </td>
                        <td className="px-4 py-2.5 text-muted-foreground tabular-nums">
                          {formatTimestampIso(video.created_at)}
                        </td>
                        <td className="px-4 py-2.5 tabular-nums">
                          {stats ? (
                            <span>
                              {stats.jobCount} jobs · {stats.candidateCount} tracks
                              {stats.confirmedCount > 0 && (
                                <span className="ml-1 text-status-success">
                                  ({stats.confirmedCount} conf.)
                                </span>
                              )}
                            </span>
                          ) : (
                            <span className="text-muted-foreground">0 jobs</span>
                          )}
                        </td>
                        <td className="px-4 py-2.5">
                          <div className="flex items-center justify-end gap-1.5">
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => setDetailVideo(video)}
                              title="View metadata and preview"
                            >
                              <Eye className="h-3.5 w-3.5" aria-hidden="true" />
                              Details
                            </Button>
                            <Button
                              variant="secondary"
                              size="sm"
                              onClick={() =>
                                navigate(`/processing?videoId=${video.id}`)
                              }
                            >
                              <PlayCircle className="h-3.5 w-3.5" aria-hidden="true" />
                              Process
                            </Button>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => setDeleteTarget(video)}
                              aria-label={`Delete ${video.original_filename}`}
                              className="text-status-danger hover:bg-status-danger-bg"
                            >
                              <Trash2 className="h-3.5 w-3.5" aria-hidden="true" />
                            </Button>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}

          <PaginationControls
            pagination={videosQuery.data?.pagination}
            onPageChange={setPage}
          />
        </CardContent>
      </Card>

      {/* Upload Dialog */}
      <Dialog
        open={uploadOpen}
        onClose={closeUploadDialog}
        title="Upload Surveillance Video"
        description={`Accepted formats: ${ACCEPTED_EXTENSIONS.join(", ")} · Max size: ${MAX_UPLOAD_DISPLAY_MB} MB`}
      >
        <div className="space-y-4">
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              const f = e.dataTransfer.files?.[0] ?? null;
              validateAndSelectFile(f);
            }}
            className={`flex flex-col items-center justify-center rounded-lg border-2 border-dashed px-4 py-8 text-center transition-colors ${
              dragOver
                ? "border-primary bg-primary-soft"
                : "border-border bg-surface-muted/40"
            }`}
          >
            <Upload className="mb-2 h-6 w-6 text-muted-foreground" aria-hidden="true" />
            <p className="text-sm font-medium text-foreground">
              Drag and drop a video file here
            </p>
            <p className="mt-1 text-xs text-muted-foreground">
              or choose a file from disk for OpenCV metadata validation
            </p>
            <input
              ref={fileInputRef}
              type="file"
              accept={ACCEPTED_EXTENSIONS.join(",")}
              className="sr-only"
              aria-label="Select video file"
              onChange={(e) => validateAndSelectFile(e.target.files?.[0] ?? null)}
            />
            <Button
              variant="outline"
              size="sm"
              className="mt-3"
              disabled={uploading}
              onClick={() => fileInputRef.current?.click()}
            >
              Browse File
            </Button>
          </div>

          {selectedFile && (
            <div className="flex items-center justify-between gap-3 rounded-md border border-border bg-surface-muted px-3 py-2 text-xs">
              <div className="min-w-0">
                <p className="font-medium text-foreground truncate">
                  {selectedFile.name}
                </p>
                <p className="text-muted-foreground tabular-nums">
                  {formatBytes(selectedFile.size)}
                </p>
              </div>
              {!uploading && (
                <button
                  type="button"
                  onClick={() => {
                    setSelectedFile(null);
                    setUploadError(null);
                  }}
                  aria-label="Clear selected file"
                  className="rounded p-1 text-muted-foreground hover:text-foreground"
                >
                  <X className="h-4 w-4" />
                </button>
              )}
            </div>
          )}

          {uploading && (
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs tabular-nums">
                <span>Uploading and validating media…</span>
                <span>{uploadProgress}%</span>
              </div>
              <div className="h-2 w-full overflow-hidden rounded-full bg-surface-muted">
                <div
                  className="h-full bg-primary transition-all"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
            </div>
          )}

          {uploadError && (
            <div
              role="alert"
              className="rounded-md border border-status-danger-border bg-status-danger-bg p-3 text-xs text-status-danger space-y-2"
            >
              <p>{uploadError}</p>
              {duplicateVideoId && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    setUploadOpen(false);
                    navigate(`/processing?videoId=${duplicateVideoId}`);
                  }}
                >
                  Open Existing Video in Processing
                </Button>
              )}
            </div>
          )}

          <div className="flex items-center justify-end gap-2 pt-2">
            {uploading ? (
              <Button variant="outline" onClick={handleCancelUpload}>
                Cancel Upload
              </Button>
            ) : (
              <Button variant="outline" onClick={closeUploadDialog}>
                Close
              </Button>
            )}
            <Button
              onClick={() => void handleUploadSubmit()}
              loading={uploading}
              disabled={!selectedFile || Boolean(uploadError && !duplicateVideoId && selectedFile.size === 0)}
            >
              {uploadError ? "Retry Upload" : "Start Upload"}
            </Button>
          </div>
        </div>
      </Dialog>

      {/* Video Detail & Player Dialog */}
      <Dialog
        open={Boolean(detailVideo)}
        onClose={() => setDetailVideo(null)}
        title={detailVideo?.original_filename ?? "Video Details"}
        description={detailVideo ? `ID: ${detailVideo.id}` : undefined}
        maxWidthClass="max-w-3xl"
      >
        {detailVideo && (
          <div className="space-y-5">
            <VideoPlayer video={detailVideo} />

            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 rounded-md border border-border bg-surface-muted p-3 text-xs">
              <div>
                <span className="text-muted-foreground">Resolution</span>
                <p className="font-medium tabular-nums">
                  {detailVideo.width}×{detailVideo.height}
                </p>
              </div>
              <div>
                <span className="text-muted-foreground">Frame Rate / Count</span>
                <p className="font-medium tabular-nums">
                  {detailVideo.fps.toFixed(2)} fps ({detailVideo.frame_count})
                </p>
              </div>
              <div>
                <span className="text-muted-foreground">Duration</span>
                <p className="font-medium tabular-nums">
                  {formatDuration(detailVideo.duration_seconds)}
                </p>
              </div>
              <div>
                <span className="text-muted-foreground">Format / Codec</span>
                <p className="font-medium">
                  {detailVideo.format.toUpperCase()} / {detailVideo.codec ?? "Unknown"}
                </p>
              </div>
              <div className="col-span-2 sm:col-span-4">
                <span className="text-muted-foreground">SHA-256 Digest</span>
                <p className="font-mono text-[11px] break-all text-foreground">
                  {detailVideo.sha256}
                </p>
              </div>
            </div>

            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <div className="space-y-2">
                <h3 className="text-xs font-semibold text-foreground">
                  Processing History
                </h3>
                {detailJobsQuery.isLoading ? (
                  <TableSkeleton rows={2} />
                ) : (detailJobsQuery.data?.items.length ?? 0) === 0 ? (
                  <p className="text-xs text-muted-foreground">
                    No processing jobs run on this video yet.
                  </p>
                ) : (
                  <div className="space-y-1.5">
                    {detailJobsQuery.data!.items.map((job) => (
                      <Link
                        key={job.id}
                        to={`/processing?jobId=${job.id}`}
                        onClick={() => setDetailVideo(null)}
                        className="flex items-center justify-between rounded border border-border px-2.5 py-2 text-xs hover:bg-surface-muted"
                      >
                        <span className="font-mono">{truncateId(job.id)}</span>
                        <JobStatusBadge status={job.status} />
                      </Link>
                    ))}
                  </div>
                )}
              </div>

              <div className="space-y-2">
                <h3 className="text-xs font-semibold text-foreground">
                  Related Events
                </h3>
                {detailEventsQuery.isLoading ? (
                  <TableSkeleton rows={2} />
                ) : (detailEventsQuery.data?.items.length ?? 0) === 0 ? (
                  <p className="text-xs text-muted-foreground">
                    No events detected for this video yet.
                  </p>
                ) : (
                  <div className="space-y-1.5">
                    {detailEventsQuery.data!.items.map((ev) => (
                      <Link
                        key={ev.id}
                        to={`/events?eventId=${ev.id}`}
                        onClick={() => setDetailVideo(null)}
                        className="flex items-center justify-between rounded border border-border px-2.5 py-2 text-xs hover:bg-surface-muted"
                      >
                        <span>Track #{ev.track_id} ({formatDuration(ev.duration_seconds)})</span>
                        <StatusBadge
                          tone={ev.review_required ? "warning" : ev.was_confirmed ? "success" : "neutral"}
                          label={ev.review_required ? "Review" : ev.was_confirmed ? "Confirmed" : ev.event_state}
                          icon={ev.review_required ? "warn" : ev.was_confirmed ? "check" : "neutral"}
                        />
                      </Link>
                    ))}
                  </div>
                )}
              </div>
            </div>

            <div className="flex justify-end gap-2">
              <Button variant="outline" onClick={() => setDetailVideo(null)}>
                Close
              </Button>
              <Button
                onClick={() => {
                  const id = detailVideo.id;
                  setDetailVideo(null);
                  navigate(`/processing?videoId=${id}`);
                }}
              >
                Configure Processing Job
              </Button>
            </div>
          </div>
        )}
      </Dialog>

      {/* Safe Deletion Dialog */}
      <Dialog
        open={Boolean(deleteTarget)}
        onClose={() => setDeleteTarget(null)}
        title="Delete Video"
        description="Removes the uploaded media file while preserving historical job and event records if already processed."
      >
        {deleteTarget && (
          <div className="space-y-4 text-xs">
            <p className="text-foreground">
              Confirm deletion of{" "}
              <strong className="font-semibold">
                {deleteTarget.original_filename}
              </strong>{" "}
              ({formatBytes(deleteTarget.file_size_bytes)})?
            </p>
            <p className="text-muted-foreground">
              Deletion is blocked if a processing job is currently queued or running for this video.
            </p>
            <div className="flex justify-end gap-2">
              <Button variant="outline" onClick={() => setDeleteTarget(null)}>
                Cancel
              </Button>
              <Button
                variant="danger"
                loading={deleteMutation.isPending}
                onClick={() => deleteMutation.mutate(deleteTarget.id)}
              >
                Delete Video
              </Button>
            </div>
          </div>
        )}
      </Dialog>
    </div>
  );
}
