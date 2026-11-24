import { useQuery } from "@tanstack/react-query";
import { RefreshCw } from "lucide-react";
import { trinetraApi } from "@/api/client";
import { Button } from "@/components/ui/Button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";
import { ErrorState } from "@/components/ui/StateFeedback";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { formatBytes, formatTimestampIso } from "@/lib/utils";

function DiagRow({
  label,
  value,
}: {
  label: string;
  value: React.ReactNode;
}) {
  return (
    <div className="flex items-center justify-between gap-4 border-b border-border py-2.5 last:border-b-0 text-xs">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-medium text-foreground text-right tabular-nums">
        {value}
      </span>
    </div>
  );
}

export function SystemPage() {
  const healthQ = useQuery({
    queryKey: ["health"],
    queryFn: ({ signal }) => trinetraApi.getHealth(signal),
    refetchInterval: 10_000,
  });
  const statusQ = useQuery({
    queryKey: ["system-status"],
    queryFn: ({ signal }) => trinetraApi.getSystemStatus(signal),
    refetchInterval: 10_000,
  });
  const resourcesQ = useQuery({
    queryKey: ["system-resources"],
    queryFn: ({ signal }) => trinetraApi.getSystemResources(signal),
    refetchInterval: 10_000,
  });
  const modelQ = useQuery({
    queryKey: ["system-model"],
    queryFn: ({ signal }) => trinetraApi.getSystemModel(signal),
    refetchInterval: 15_000,
  });

  const refreshAll = () => {
    void healthQ.refetch();
    void statusQ.refetch();
    void resourcesQ.refetch();
    void modelQ.refetch();
  };

  const h = healthQ.data;
  const st = statusQ.data;
  const res = resourcesQ.data;
  const mod = modelQ.data;

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="text-xs text-muted-foreground">
          Live separation of API process health, database/storage readiness, background worker, CUDA availability, and lazy model state.
        </p>
        <Button variant="outline" size="sm" onClick={refreshAll}>
          <RefreshCw className="h-3.5 w-3.5" aria-hidden="true" />
          Refresh Diagnostics
        </Button>
      </div>

      {healthQ.isError && (
        <ErrorState
          title="Backend unreachable"
          message={healthQ.error.message}
          onRetry={refreshAll}
        />
      )}

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Service &amp; Infrastructure Readiness</CardTitle>
            {h && (
              <StatusBadge
                tone={h.status === "healthy" ? "success" : "warning"}
                label={h.status.toUpperCase()}
                icon={h.status === "healthy" ? "check" : "warn"}
              />
            )}
          </CardHeader>
          <CardContent>
            {healthQ.isLoading || statusQ.isLoading ? (
              <Skeleton className="h-44 w-full" />
            ) : (
              <div>
                <DiagRow label="API Process" value={h?.api ?? "Unavailable"} />
                <DiagRow
                  label="Environment"
                  value={st?.environment ?? "Unavailable"}
                />
                <DiagRow
                  label="API Version"
                  value={st?.version ?? "Unavailable"}
                />
                <DiagRow
                  label="Database Connectivity"
                  value={
                    <StatusBadge
                      tone={st?.database_connected ? "success" : "danger"}
                      label={st?.database_connected ? "Connected" : "Unavailable"}
                      icon={st?.database_connected ? "check" : "error"}
                    />
                  }
                />
                <DiagRow
                  label="Storage Writable"
                  value={
                    <StatusBadge
                      tone={st?.storage_writable ? "success" : "danger"}
                      label={st?.storage_writable ? "Writable" : "Unavailable"}
                      icon={st?.storage_writable ? "check" : "error"}
                    />
                  }
                />
                <DiagRow
                  label="Background Worker"
                  value={
                    <StatusBadge
                      tone={st?.worker_running ? "success" : "danger"}
                      label={st?.worker_running ? "Running" : "Stopped"}
                      icon={st?.worker_running ? "activity" : "error"}
                    />
                  }
                />
                <DiagRow
                  label="Active / Queued Jobs"
                  value={`${st?.active_jobs ?? 0} active / ${st?.queued_jobs ?? 0} queued`}
                />
                <DiagRow
                  label="Last Probe"
                  value={formatTimestampIso(h?.timestamp)}
                />
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Frozen Model &amp; Integrity Status</CardTitle>
            {mod && (
              <StatusBadge
                tone={mod.model_initialized ? "info" : "neutral"}
                label={mod.model_initialized ? "Model Loaded" : "Standby (Lazy)"}
                icon={mod.model_initialized ? "check" : "clock"}
              />
            )}
          </CardHeader>
          <CardContent>
            {modelQ.isLoading ? (
              <Skeleton className="h-44 w-full" />
            ) : modelQ.isError ? (
              <ErrorState message={modelQ.error.message} />
            ) : (
              <div>
                <DiagRow label="Architecture" value={mod?.architecture ?? "—"} />
                <DiagRow label="Backbone" value={mod?.backbone ?? "—"} />
                <DiagRow
                  label="Feature / Logit Shapes"
                  value={
                    mod
                      ? `[${mod.feature_dimensions.join(", ")}] → [${mod.output_logits_dimensions.join(", ")}]`
                      : "—"
                  }
                />
                <DiagRow
                  label="Checkpoint File"
                  value={
                    <StatusBadge
                      tone={mod?.checkpoint_exists ? "success" : "danger"}
                      label={mod?.checkpoint_exists ? "Present" : "Missing"}
                      icon={mod?.checkpoint_exists ? "check" : "error"}
                    />
                  }
                />
                <DiagRow
                  label="SHA-256 Verification"
                  value={
                    mod?.checkpoint_sha256_verified === true
                      ? "Verified"
                      : mod?.checkpoint_sha256_verified === false
                        ? "Failed"
                        : "Pending first model load"
                  }
                />
                <DiagRow
                  label="Expected SHA-256"
                  value={
                    <span className="font-mono text-[11px] break-all">
                      {mod?.expected_sha256 ?? "—"}
                    </span>
                  }
                />
                <DiagRow
                  label="Target Device"
                  value={mod?.device ?? "Unavailable"}
                />
                <DiagRow
                  label="Model Load Time"
                  value={
                    mod?.load_time_seconds !== null &&
                    mod?.load_time_seconds !== undefined
                      ? `${mod.load_time_seconds.toFixed(3)}s`
                      : "Not loaded yet"
                  }
                />
              </div>
            )}
          </CardContent>
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Host Compute, Storage &amp; GPU Resources</CardTitle>
            {res && (
              <StatusBadge
                tone={res.cuda_available ? "success" : "neutral"}
                label={res.cuda_available ? "CUDA Available" : "CPU Only"}
                icon={res.cuda_available ? "check" : "neutral"}
              />
            )}
          </CardHeader>
          <CardContent>
            {resourcesQ.isLoading ? (
              <Skeleton className="h-32 w-full" />
            ) : resourcesQ.isError ? (
              <ErrorState message={resourcesQ.error.message} />
            ) : (
              <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
                <div>
                  <DiagRow
                    label="CPU Utilization"
                    value={
                      res?.cpu_percent !== null && res?.cpu_percent !== undefined
                        ? `${res.cpu_percent.toFixed(1)}%`
                        : "Unavailable"
                    }
                  />
                  <DiagRow
                    label="RAM Used / Total"
                    value={
                      res?.ram_total_bytes && res?.ram_available_bytes !== null
                        ? `${formatBytes(res.ram_total_bytes - (res.ram_available_bytes ?? 0))} / ${formatBytes(res.ram_total_bytes)} (${res.ram_used_percent?.toFixed(1)}%)`
                        : "Unavailable"
                    }
                  />
                  <DiagRow
                    label="Storage Disk Free / Total"
                    value={
                      res
                        ? `${formatBytes(res.disk_free_bytes)} free / ${formatBytes(res.disk_total_bytes)}`
                        : "Unavailable"
                    }
                  />
                </div>
                <div>
                  <DiagRow
                    label="GPU Device"
                    value={res?.gpu_name ?? "Unavailable"}
                  />
                  <DiagRow
                    label="VRAM Allocated"
                    value={
                      res?.gpu_memory_allocated_bytes !== null &&
                      res?.gpu_memory_allocated_bytes !== undefined
                        ? formatBytes(res.gpu_memory_allocated_bytes)
                        : "Unavailable"
                    }
                  />
                  <DiagRow
                    label="VRAM Reserved / Total"
                    value={
                      res?.gpu_memory_total_bytes !== null &&
                      res?.gpu_memory_total_bytes !== undefined
                        ? `${formatBytes(res.gpu_memory_reserved_bytes)} / ${formatBytes(res.gpu_memory_total_bytes)}`
                        : "Unavailable"
                    }
                  />
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
