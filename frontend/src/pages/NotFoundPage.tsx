import { Link } from "react-router-dom";
import { Button } from "@/components/ui/Button";
import { Card, CardContent } from "@/components/ui/Card";

export function NotFoundPage() {
  return (
    <Card className="mx-auto max-w-md mt-12">
      <CardContent className="p-6 text-center space-y-3">
        <p className="text-xs font-mono text-muted-foreground">404</p>
        <h2 className="text-lg font-semibold text-foreground">Page not found</h2>
        <p className="text-xs text-muted-foreground">
          The requested TRINETRA route does not exist.
        </p>
        <div className="pt-2">
          <Link to="/dashboard">
            <Button size="sm">Return to Dashboard</Button>
          </Link>
        </div>
      </CardContent>
    </Card>
  );
}
