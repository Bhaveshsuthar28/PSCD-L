import { useState, type ReactNode } from "react";
import { NavLink, useLocation, Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  Cpu,
  Film,
  Images,
  LayoutDashboard,
  Menu,
  Monitor,
  Moon,
  PanelLeftClose,
  PanelLeftOpen,
  PlayCircle,
  Settings as SettingsIcon,
  Sun,
  X,
  XCircle,
} from "lucide-react";
import { trinetraApi } from "@/api/client";
import { useTheme, type ThemePreference } from "@/hooks/useTheme";
import { cn } from "@/lib/utils";

interface NavItem {
  to: string;
  label: string;
  icon: typeof LayoutDashboard;
}

const NAV_ITEMS: NavItem[] = [
  { to: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { to: "/videos", label: "Video Library", icon: Film },
  { to: "/processing", label: "Processing", icon: PlayCircle },
  { to: "/events", label: "Events", icon: Activity },
  { to: "/evidence", label: "Evidence", icon: Images },
  { to: "/analytics", label: "Analytics", icon: BarChart3 },
  { to: "/system", label: "System Status", icon: Cpu },
  { to: "/settings", label: "Settings", icon: SettingsIcon },
];

const ROUTE_TITLES: Record<string, string> = {
  "/dashboard": "Overview Dashboard",
  "/videos": "Video Library",
  "/processing": "Video Processing",
  "/events": "Events Explorer",
  "/evidence": "Evidence Viewer",
  "/analytics": "Analytics & Exports",
  "/system": "System Status",
  "/settings": "Settings",
};

export function AppShell({ children }: { children: ReactNode }) {
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const { pathname } = useLocation();
  const { preference, setPreference } = useTheme();

  const healthQuery = useQuery({
    queryKey: ["health"],
    queryFn: ({ signal }) => trinetraApi.getHealth(signal),
    refetchInterval: 15_000,
  });

  const statusQuery = useQuery({
    queryKey: ["system-status"],
    queryFn: ({ signal }) => trinetraApi.getSystemStatus(signal),
    refetchInterval: 15_000,
  });

  const pageTitle =
    Object.entries(ROUTE_TITLES).find(([prefix]) =>
      pathname.startsWith(prefix),
    )?.[1] ?? "TRINETRA";

  const health = healthQuery.data;
  const sysStatus = statusQuery.data;
  const activeJobs = sysStatus?.active_jobs ?? 0;

  const renderConnectivityPill = () => {
    if (healthQuery.isError) {
      return (
        <Link
          to="/system"
          className="inline-flex items-center gap-1.5 rounded-md border border-status-danger-border bg-status-danger-bg px-2.5 py-1 text-xs font-medium text-status-danger"
          title="Backend API unreachable"
        >
          <XCircle className="h-3.5 w-3.5" aria-hidden="true" />
          <span className="hidden sm:inline">Backend Offline</span>
        </Link>
      );
    }
    if (!health) {
      return (
        <span className="inline-flex items-center gap-1.5 rounded-md border border-border bg-surface-muted px-2.5 py-1 text-xs text-muted-foreground">
          Checking API…
        </span>
      );
    }

    const fullyReady =
      health.status === "healthy" &&
      health.database === "connected" &&
      health.worker === "running";

    return (
      <Link
        to="/system"
        className={cn(
          "inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs font-medium transition-colors",
          fullyReady
            ? "border-status-success-border bg-status-success-bg text-status-success"
            : "border-status-warning-border bg-status-warning-bg text-status-warning",
        )}
        title={`API: ${health.api} | DB: ${health.database} | Worker: ${health.worker} | Model: ${health.model_initialized ? "Loaded" : "Standby"}`}
      >
        {fullyReady ? (
          <CheckCircle2 className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
        ) : (
          <AlertTriangle className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
        )}
        <span className="hidden sm:inline">
          {fullyReady ? "Backend Ready" : "Backend Degraded"}
        </span>
        <span className="hidden md:inline text-[11px] opacity-80">
          · {health.model_initialized ? "Model Loaded" : "Model Standby"}
        </span>
      </Link>
    );
  };

  const themeOptions: { value: ThemePreference; label: string; icon: typeof Sun }[] = [
    { value: "light", label: "Light", icon: Sun },
    { value: "dark", label: "Dark", icon: Moon },
    { value: "system", label: "System", icon: Monitor },
  ];

  const SidebarContent = ({ isMobile = false }: { isMobile?: boolean }) => (
    <div className="flex h-full flex-col bg-surface border-r border-border">
      <div className="flex h-14 items-center justify-between border-b border-border px-3.5">
        <Link
          to="/dashboard"
          onClick={() => isMobile && setMobileOpen(false)}
          className="flex min-w-0 flex-col"
        >
          <span className="text-sm font-semibold tracking-tight text-foreground truncate">
            {collapsed && !isMobile ? "TR" : "TRINETRA"}
          </span>
          {(!collapsed || isMobile) && (
            <span className="text-[11px] text-muted-foreground truncate">
              PSCD&amp;L Console
            </span>
          )}
        </Link>
        {isMobile ? (
          <button
            type="button"
            onClick={() => setMobileOpen(false)}
            aria-label="Close navigation menu"
            className="rounded p-1 text-muted-foreground hover:bg-surface-muted hover:text-foreground"
          >
            <X className="h-4 w-4" />
          </button>
        ) : (
          <button
            type="button"
            onClick={() => setCollapsed((c) => !c)}
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            className="rounded p-1 text-muted-foreground hover:bg-surface-muted hover:text-foreground"
          >
            {collapsed ? (
              <PanelLeftOpen className="h-4 w-4" />
            ) : (
              <PanelLeftClose className="h-4 w-4" />
            )}
          </button>
        )}
      </div>

      <nav className="flex-1 space-y-1 p-2 overflow-y-auto" aria-label="Main navigation">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.to}
              to={item.to}
              onClick={() => isMobile && setMobileOpen(false)}
              title={collapsed && !isMobile ? item.label : undefined}
              className={({ isActive }) =>
                cn(
                  "flex items-center gap-2.5 rounded-md px-2.5 py-2 text-sm font-medium transition-colors focus-visible:outline-2 focus-visible:outline-ring",
                  collapsed && !isMobile && "justify-center px-2",
                  isActive
                    ? "bg-primary-soft text-primary"
                    : "text-muted-foreground hover:bg-surface-muted hover:text-foreground",
                )
              }
            >
              <Icon className="h-4 w-4 shrink-0" aria-hidden="true" />
              {(!collapsed || isMobile) && (
                <span className="truncate">{item.label}</span>
              )}
            </NavLink>
          );
        })}
      </nav>

      {(!collapsed || isMobile) && (
        <div className="border-t border-border p-3 text-[11px] leading-relaxed text-muted-foreground">
          Detects persistent visual scene changes. Does not independently establish
          object identity or abandonment.
        </div>
      )}
    </div>
  );

  return (
    <div className="min-h-screen flex bg-background text-foreground overflow-x-hidden">
      {/* Desktop Sidebar */}
      <aside
        className={cn(
          "hidden lg:block shrink-0 transition-[width] duration-150",
          collapsed ? "w-16" : "w-60",
        )}
      >
        <div className="sticky top-0 h-screen">
          <SidebarContent />
        </div>
      </aside>

      {/* Mobile Drawer */}
      {mobileOpen && (
        <div
          className="fixed inset-0 z-50 flex lg:hidden bg-black/50"
          onMouseDown={(e) => {
            if (e.target === e.currentTarget) setMobileOpen(false);
          }}
        >
          <aside className="w-64 max-w-[80vw] h-full">
            <SidebarContent isMobile />
          </aside>
        </div>
      )}

      {/* Main Content */}
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-30 flex h-14 items-center justify-between gap-3 border-b border-border bg-surface px-4 lg:px-6">
          <div className="flex items-center gap-3 min-w-0">
            <button
              type="button"
              onClick={() => setMobileOpen(true)}
              aria-label="Open navigation menu"
              className="rounded-md p-1.5 text-muted-foreground hover:bg-surface-muted hover:text-foreground lg:hidden"
            >
              <Menu className="h-5 w-5" />
            </button>
            <div className="min-w-0">
              <h1 className="text-sm sm:text-base font-semibold text-foreground truncate">
                {pageTitle}
              </h1>
            </div>
          </div>

          <div className="flex items-center gap-2 sm:gap-3 shrink-0">
            {activeJobs > 0 && (
              <Link
                to="/processing"
                className="inline-flex items-center gap-1.5 rounded-md border border-status-info-border bg-status-info-bg px-2.5 py-1 text-xs font-medium text-status-info"
              >
                <Activity className="h-3.5 w-3.5 animate-pulse" aria-hidden="true" />
                <span className="tabular-nums">{activeJobs} Active</span>
              </Link>
            )}

            {renderConnectivityPill()}

            <div
              role="group"
              aria-label="Theme preference"
              className="inline-flex items-center rounded-md border border-border bg-surface-muted p-0.5"
            >
              {themeOptions.map((opt) => {
                const Icon = opt.icon;
                const active = preference === opt.value;
                return (
                  <button
                    key={opt.value}
                    type="button"
                    onClick={() => setPreference(opt.value)}
                    aria-label={`${opt.label} theme`}
                    aria-pressed={active}
                    title={`${opt.label} theme`}
                    className={cn(
                      "rounded px-1.5 py-1 text-xs transition-colors focus-visible:outline-2 focus-visible:outline-ring",
                      active
                        ? "bg-surface text-foreground shadow-xs"
                        : "text-muted-foreground hover:text-foreground",
                    )}
                  >
                    <Icon className="h-3.5 w-3.5" aria-hidden="true" />
                  </button>
                );
              })}
            </div>
          </div>
        </header>

        <main className="flex-1 p-4 lg:p-6 max-w-[1600px] w-full mx-auto min-w-0">
          {children}
        </main>
      </div>
    </div>
  );
}
