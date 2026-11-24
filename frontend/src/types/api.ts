export interface ApiErrorDetail {
  code: string;
  message: string;
  details: unknown;
  request_id: string | null;
}

export interface ApiErrorEnvelope {
  success: false;
  error: ApiErrorDetail;
}

export interface PaginationMeta {
  page: number;
  page_size: number;
  total_items: number;
  total_pages: number;
  has_next: boolean;
  has_previous: boolean;
}

export interface VideoRecord {
  id: string;
  original_filename: string;
  stored_filename: string;
  file_size_bytes: number;
  sha256: string;
  duration_seconds: number;
  fps: number;
  frame_count: number;
  width: number;
  height: number;
  codec: string | null;
  format: string;
  upload_status: string;
  created_at: string;
  updated_at: string;
}

export interface VideoListResponse {
  items: VideoRecord[];
  pagination: PaginationMeta;
}

export interface VideoMetadataDetail {
  duration_seconds: number;
  fps: number;
  frame_count: number;
  width: number;
  height: number;
  codec: string | null;
  format: string;
  file_size_bytes: number;
  sha256: string;
}

export interface VideoDeleteResponse {
  success: boolean;
  video_id: string;
  file_removed: boolean;
  record_deleted: boolean;
  historical_records_preserved: boolean;
  message: string;
}

export interface ProcessingConfigurationInput {
  candidate_config?: {
    min_area?: number;
    opening_kernel?: number;
    closing_kernel?: number;
    boundary_margin?: number;
  };
  temporal_config?: {
    min_persistence_seconds?: number;
    min_observations?: number;
    settling_min_seconds?: number;
    min_motion_samples?: number;
    settling_motion_window_samples?: number;
    max_motion_history?: number;
  };
}

export interface ProcessingJobCreateRequest {
  video_id: string;
  reference_seconds: number;
  start_seconds: number;
  end_seconds: number | null;
  sampling_interval_seconds: number;
  configuration: ProcessingConfigurationInput;
}

export interface ProcessingJobRecord {
  id: string;
  video_id: string;
  status: string;
  current_stage: string;
  progress_percentage: number | null;
  processed_frames: number;
  total_frames: number | null;
  processed_seconds: number;
  current_video_timestamp_seconds: number | null;
  processing_fps: number | null;
  elapsed_seconds: number | null;
  estimated_remaining_seconds: number | null;
  active_tracks_count: number;
  confirmed_events_count: number;
  review_required_events_count: number;
  total_candidates_count: number;
  reference_seconds: number;
  start_seconds: number;
  end_seconds: number | null;
  sampling_interval_seconds: number;
  started_at: string | null;
  completed_at: string | null;
  created_at: string;
  updated_at: string;
  error_code: string | null;
  error_message: string | null;
  configuration_snapshot: Record<string, unknown>;
  evidence_generation_status: string;
  evidence_generation_error: string | null;
}

export interface ProcessingJobListResponse {
  items: ProcessingJobRecord[];
  pagination: PaginationMeta;
}

export interface JobCancelResponse {
  job_id: string;
  status: string;
  current_stage: string;
  message: string;
}

export interface JobLogEntry {
  id: number;
  job_id: string;
  level: string;
  stage: string | null;
  message: string;
  details: Record<string, unknown> | null;
  created_at: string;
}

export interface ProcessingJobLogsResponse {
  job_id: string;
  items: JobLogEntry[];
  total_items: number;
}

export interface ProcessingJobSummaryResponse {
  job_id: string;
  video_id: string;
  status: string;
  scientific_note: string;
  summary: Record<string, unknown> | null;
}

export interface EvidenceRecord {
  id: string;
  event_id: string | null;
  job_id: string;
  video_id?: string | null;
  evidence_type: string;
  original_filename: string;
  mime_type: string;
  file_size_bytes: number;
  sha256: string;
  timestamp_seconds: number | null;
  width: number | null;
  height: number | null;
  bbox_xywh: number[] | null;
  download_url: string | null;
  created_at: string;
}

export interface EvidenceListResponse {
  event_id: string;
  items: EvidenceRecord[];
  count: number;
}

export interface EvidencePaginatedResponse {
  items: EvidenceRecord[];
  pagination: PaginationMeta;
}

export interface EventRecord {
  id: string;
  job_id: string;
  video_id: string;
  track_id: number;
  event_state: string;
  first_seen_seconds: number;
  last_seen_seconds: number;
  duration_seconds: number;
  observations: number;
  missed_observations: number;
  total_missed_observations: number;
  detection_consistency: number;
  center_deviation_pixels: number;
  bbox_stability: number;
  motion_samples: number;
  motion_median_ratio: number | null;
  motion_p90_ratio: number | null;
  settling_motion_samples: number;
  settling_motion_median_ratio: number | null;
  settling_motion_p90_ratio: number | null;
  confidence: number;
  confidence_type: string;
  was_confirmed: boolean;
  confirmation_seconds: number | null;
  review_required: boolean;
  post_confirmation_degraded: boolean;
  degradation_first_seen_seconds: number | null;
  post_confirmation_degradation_reasons: string[];
  boundary_candidate: boolean;
  latest_bbox_xywh: number[] | null;
  description: string;
  created_at: string;
  updated_at: string;
}

export interface EventStateHistoryEntry {
  timestamp_seconds: number;
  state: string;
}

export interface EventTimelineEntry {
  timestamp_seconds: number;
  bbox_xywh?: number[];
  observed_in_latest_update?: boolean;
  observations?: number;
  missed_observations?: number;
  total_missed_observations?: number;
  detection_consistency?: number;
  center_deviation_pixels?: number;
  bbox_stability?: number;
  motion_ratio?: number | null;
  settling_motion_samples?: number;
  settling_motion_median_ratio?: number | null;
  settling_motion_p90_ratio?: number | null;
  confidence?: number;
  review_required?: boolean;
  post_confirmation_degraded?: boolean;
  completion?: boolean;
}

export interface EventDetailResponse extends EventRecord {
  state_history: EventStateHistoryEntry[];
  evidence_timeline: EventTimelineEntry[];
  evidence_items: EvidenceRecord[];
}

export interface EventListResponse {
  items: EventRecord[];
  pagination: PaginationMeta;
}

export interface AnalyticsOverviewResponse {
  total_uploaded_videos: number;
  total_processing_jobs: number;
  completed_jobs: number;
  failed_jobs: number;
  cancelled_jobs: number;
  queued_jobs: number;
  active_jobs: number;
  total_candidate_events: number;
  historical_confirmations: number;
  non_review_confirmations: number;
  review_required_events: number;
  post_confirmation_degraded_events: number;
  total_processing_duration_seconds: number;
  note: string;
}

export interface VideoAnalyticsItem {
  video_id: string;
  original_filename: string;
  duration_seconds: number;
  upload_status: string;
  job_count: number;
  candidate_event_count: number;
  historical_confirmed_event_count: number;
  review_required_event_count: number;
}

export interface VideoAnalyticsResponse {
  items: VideoAnalyticsItem[];
  count: number;
}

export interface DistributionBucket {
  bucket: string;
  count: number;
}

export interface EventAnalyticsResponse {
  total_candidate_events: number;
  historical_confirmations: number;
  non_review_confirmations: number;
  review_required_events: number;
  post_confirmation_degraded_events: number;
  by_state: Record<string, number>;
  confidence_distribution: DistributionBucket[];
  duration_distribution: DistributionBucket[];
}

export interface ProcessingAnalyticsResponse {
  jobs_by_status: Record<string, number>;
  completed_jobs_count: number;
  average_processing_fps: number | null;
  average_elapsed_seconds: number | null;
  total_processed_frames_completed_jobs: number;
}

export interface TimelinePoint {
  date: string;
  candidate_events: number;
  historical_confirmations: number;
  review_required_events: number;
}

export interface TimelineAnalyticsResponse {
  items: TimelinePoint[];
  count: number;
}

export interface HealthResponse {
  status: string;
  api: string;
  database: string;
  storage: string;
  worker: string;
  checkpoint_available: boolean;
  model_initialized: boolean;
  cuda_available: boolean;
  timestamp: string;
}

export interface SystemStatusResponse {
  project: string;
  version: string;
  environment: string;
  database_connected: boolean;
  storage_writable: boolean;
  worker_running: boolean;
  queued_jobs: number;
  active_jobs: number;
  checkpoint_available: boolean;
  model_initialized: boolean;
  cuda_available: boolean;
  scientific_scope: string;
}

export interface SystemResourcesResponse {
  cpu_percent: number | null;
  ram_total_bytes: number | null;
  ram_available_bytes: number | null;
  ram_used_percent: number | null;
  disk_total_bytes: number;
  disk_used_bytes: number;
  disk_free_bytes: number;
  cuda_available: boolean;
  gpu_name: string | null;
  gpu_memory_allocated_bytes: number | null;
  gpu_memory_reserved_bytes: number | null;
  gpu_memory_total_bytes: number | null;
}

export interface SystemModelResponse {
  architecture: string;
  backbone: string;
  feature_dimensions: number[];
  output_logits_dimensions: number[];
  checkpoint_exists: boolean;
  expected_sha256: string;
  checkpoint_sha256_verified: boolean | null;
  model_initialized: boolean;
  device: string | null;
  load_time_seconds: number | null;
  last_error: string | null;
}

export interface SystemSettingItem {
  key: string;
  value_json: unknown;
  description: string | null;
  updated_at: string;
}

export interface SystemSettingsResponse {
  items: SystemSettingItem[];
}
