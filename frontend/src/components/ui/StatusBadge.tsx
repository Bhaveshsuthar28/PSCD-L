import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Clock,
  Eye,
  HelpCircle,
  Loader2,
  XCircle,
} from "lucide-react";
import { cn } from "@/lib/utils";

export type BadgeTone = "success" | "warning" | "danger" | "info" | "neutral";

const toneClasses: Record<BadgeTone, string> = {
  success:
    "bg-status-success-bg text-status-success border-status-success-border",
  warning:
    "bg-status-warning-bg text-status-warning border-status-warning-border",
  danger: "bg-status-danger-bg text-status-danger border-status-danger-border",
  info: "bg-status-info-bg text-status-info border-status-info-border",
  neutral:
    "bg-status-neutral-bg text-status-neutral border-status-neutral-border",
};

interface StatusBadgeProps {
  tone: BadgeTone;
  label: string;
  icon?: "check" | "warn" | "error" | "spinner" | "clock" | "eye" | "activity" | "neutral";
  className?: string;
}

export function StatusBadge({
  tone,
  label,
  icon = "neutral",
  className,
}: StatusBadgeProps) {
  const Icon = {
    check: CheckCircle2,
    warn: AlertTriangle,
    error: XCircle,
    spinner: Loader2,
    clock: Clock,
    eye: Eye,
    activity: Activity,
    neutral: HelpCircle,
  }[icon];

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded border px-2 py-0.5 text-xs font-medium whitespace-nowrap",
        toneClasses[tone],
        className,
      )}
    >
      <Icon
        className={cn("h-3 w-3 shrink-0", icon === "spinner" && "animate-spin")}
        aria-hidden="true"
      />
      <span>{label}</span>
    </span>
  );
}

export function JobStatusBadge({ status }: { status: string }) {
  switch (status) {
    case "COMPLETED":
      return <StatusBadge tone="success" label="Completed" icon="check" />;
    case "FAILED":
      return <StatusBadge tone="danger" label="Failed" icon="error" />;
    case "CANCELLED":
      return <StatusBadge tone="neutral" label="Cancelled" icon="error" />;
    case "CANCEL_REQUESTED":
      return <StatusBadge tone="warning" label="Cancelling" icon="spinner" />;
    case "PROCESSING":
    case "INITIALIZING":
    case "VALIDATING":
    case "FINALIZING":
      return <StatusBadge tone="info" label={status.replace("_", " ")} icon="spinner" />;
    case "QUEUED":
      return <StatusBadge tone="neutral" label="Queued" icon="clock" />;
    default:
      return <StatusBadge tone="neutral" label={status} icon="neutral" />;
  }
}

export function EventLifecycleBadge({ state }: { state: string }) {
  switch (state) {
    case "CONFIRMED":
      return <StatusBadge tone="success" label="Confirmed" icon="check" />;
    case "COMPLETED":
      return <StatusBadge tone="info" label="Completed" icon="check" />;
    case "SETTLING":
      return <StatusBadge tone="info" label="Settling" icon="activity" />;
    case "OBSERVING":
      return <StatusBadge tone="neutral" label="Observing" icon="eye" />;
    case "CANDIDATE":
      return <StatusBadge tone="neutral" label="Candidate" icon="clock" />;
    default:
      return <StatusBadge tone="neutral" label={state} icon="neutral" />;
  }
}
