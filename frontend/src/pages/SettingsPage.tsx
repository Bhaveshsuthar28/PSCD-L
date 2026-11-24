import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Save } from "lucide-react";
import { API_BASE_URL, trinetraApi } from "@/api/client";
import { Button } from "@/components/ui/Button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";
import { ErrorState } from "@/components/ui/StateFeedback";
import { useToast } from "@/components/ui/Toast";
import { useTheme, type ThemePreference } from "@/hooks/useTheme";

const DENSITY_STORAGE_KEY = "trinetra-table-density";

export function SettingsPage() {
  const queryClient = useQueryClient();
  const { pushToast } = useToast();
  const { preference, setPreference } = useTheme();

  const [density, setDensity] = useState<"comfortable" | "compact">(() => {
    try {
      const saved = localStorage.getItem(DENSITY_STORAGE_KEY);
      return saved === "compact" ? "compact" : "comfortable";
    } catch {
      return "comfortable";
    }
  });

  const settingsQuery = useQuery({
    queryKey: ["system-settings"],
    queryFn: ({ signal }) => trinetraApi.getSettings(signal),
  });

  const [defaultInterval, setDefaultInterval] = useState<string>("1");

  useEffect(() => {
    const items = settingsQuery.data?.items ?? [];
    const found = items.find(
      (i) => i.key === "processing.default_sampling_interval_seconds",
    );
    if (found && typeof found.value_json === "number") {
      setDefaultInterval(String(found.value_json));
    }
  }, [settingsQuery.data]);

  const updateMutation = useMutation({
    mutationFn: (payload: Record<string, unknown>) =>
      trinetraApi.updateSettings(payload),
    onSuccess: () => {
      pushToast({
        tone: "success",
        title: "Settings saved",
        description: "Persisted configuration updated in backend database.",
      });
      void queryClient.invalidateQueries({ queryKey: ["system-settings"] });
    },
    onError: (err) => {
      pushToast({
        tone: "error",
        title: "Failed to save settings",
        description: err instanceof Error ? err.message : "Save rejected.",
      });
    },
  });

  const handleDensityChange = (next: "comfortable" | "compact") => {
    setDensity(next);
    try {
      localStorage.setItem(DENSITY_STORAGE_KEY, next);
    } catch {
      // ignore
    }
    pushToast({
      tone: "info",
      title: "Display preference updated",
      description: `Table density set to ${next}.`,
    });
  };

  const handleSaveBackendSettings = () => {
    const parsed = Number(defaultInterval);
    if (!Number.isFinite(parsed) || parsed < 0.1 || parsed > 60) {
      pushToast({
        tone: "error",
        title: "Invalid sampling interval",
        description: "Default sampling interval must be between 0.1s and 60s.",
      });
      return;
    }
    updateMutation.mutate({
      "processing.default_sampling_interval_seconds": parsed,
    });
  };

  return (
    <div className="space-y-6 max-w-4xl">
      {/* Appearance & Local Preferences */}
      <Card>
        <CardHeader>
          <CardTitle>Appearance &amp; Display Preferences</CardTitle>
          <span className="text-xs text-muted-foreground">Stored locally in browser</span>
        </CardHeader>
        <CardContent className="space-y-4 text-xs">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="font-medium text-foreground">Theme Preference</p>
              <p className="text-muted-foreground">
                Choose light, dark, or follow your operating system theme.
              </p>
            </div>
            <div className="flex gap-2">
              {(["light", "dark", "system"] as ThemePreference[]).map((opt) => (
                <Button
                  key={opt}
                  size="sm"
                  variant={preference === opt ? "primary" : "outline"}
                  onClick={() => setPreference(opt)}
                >
                  {opt.charAt(0).toUpperCase() + opt.slice(1)}
                </Button>
              ))}
            </div>
          </div>

          <div className="flex flex-wrap items-center justify-between gap-3 border-t border-border pt-4">
            <div>
              <p className="font-medium text-foreground">Table Density</p>
              <p className="text-muted-foreground">
                Controls row spacing preference across explorer tables.
              </p>
            </div>
            <div className="flex gap-2">
              {(["comfortable", "compact"] as const).map((opt) => (
                <Button
                  key={opt}
                  size="sm"
                  variant={density === opt ? "primary" : "outline"}
                  onClick={() => handleDensityChange(opt)}
                >
                  {opt.charAt(0).toUpperCase() + opt.slice(1)}
                </Button>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Persisted Backend Settings */}
      <Card>
        <CardHeader>
          <CardTitle>Backend Persisted Configuration</CardTitle>
          <span className="text-xs text-muted-foreground">
            GET / PUT /api/v1/settings
          </span>
        </CardHeader>
        <CardContent className="space-y-4 text-xs">
          {settingsQuery.isLoading ? (
            <Skeleton className="h-24 w-full" />
          ) : settingsQuery.isError ? (
            <ErrorState
              message={settingsQuery.error.message}
              onRetry={() => void settingsQuery.refetch()}
            />
          ) : (
            <>
              <div className="flex flex-wrap items-end justify-between gap-3">
                <div className="space-y-1">
                  <label
                    htmlFor="default-sampling-interval"
                    className="block font-medium text-foreground"
                  >
                    Default Sampling Interval (seconds)
                  </label>
                  <p className="text-muted-foreground">
                    Applied as the baseline interval when configuring new jobs (0.1–60s).
                  </p>
                  <input
                    id="default-sampling-interval"
                    type="number"
                    min="0.1"
                    max="60"
                    step="0.1"
                    value={defaultInterval}
                    onChange={(e) => setDefaultInterval(e.target.value)}
                    className="h-8 w-36 rounded-md border border-border bg-surface px-2.5 text-xs tabular-nums"
                  />
                </div>
                <Button
                  size="sm"
                  loading={updateMutation.isPending}
                  onClick={handleSaveBackendSettings}
                >
                  <Save className="h-3.5 w-3.5" aria-hidden="true" />
                  Save Setting
                </Button>
              </div>

              <div className="border-t border-border pt-3 space-y-2">
                <p className="font-medium text-foreground">All Persisted Non-Secret Keys</p>
                {settingsQuery.data!.items.map((item) => (
                  <div
                    key={item.key}
                    className="flex flex-wrap items-center justify-between gap-2 rounded border border-border bg-surface-muted px-3 py-2"
                  >
                    <div>
                      <p className="font-mono text-[11px] text-foreground">{item.key}</p>
                      {item.description && (
                        <p className="text-[11px] text-muted-foreground">
                          {item.description}
                        </p>
                      )}
                    </div>
                    <span className="font-mono text-[11px] text-foreground">
                      {JSON.stringify(item.value_json)}
                    </span>
                  </div>
                ))}
              </div>
            </>
          )}
        </CardContent>
      </Card>

      {/* Connectivity & Security Boundary */}
      <Card>
        <CardHeader>
          <CardTitle>API Connectivity &amp; Security Boundary</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-xs text-muted-foreground">
          <p>
            <strong className="text-foreground">Configured API Base:</strong>{" "}
            <span className="font-mono">
              {API_BASE_URL || "Same-origin Vite proxy (/api/v1 → 127.0.0.1:8000)"}
            </span>
          </p>
          <p>
            <strong className="text-foreground">Frozen Model Governance:</strong>{" "}
            Checkpoint paths, SHA-256 verification, and Task 2 detection thresholds are enforced server-side and cannot be altered from the browser.
          </p>
          <p>
            <strong className="text-foreground">Local Development Boundary:</strong>{" "}
            Authentication is disabled in local development mode; deploy behind a TLS reverse proxy with access controls before external exposure.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
