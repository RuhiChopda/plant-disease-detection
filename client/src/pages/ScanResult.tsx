import { useRoute, useLocation } from "wouter";
import { Header } from "@/components/Header";
import { useScan, useDeleteScan } from "@/hooks/use-scans";
import { Loader2, ArrowLeft, Trash2, CheckCircle, AlertTriangle, Activity, ShieldCheck, Thermometer } from "lucide-react";
import { Link } from "wouter";
import { cn } from "@/lib/utils";
import { format } from "date-fns";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";

export default function ScanResult() {
  const [, params] = useRoute("/scan/:id");
  const [, setLocation] = useLocation();
  const id = Number(params?.id);
  
  const { data: scan, isLoading, isError } = useScan(id);
  const { mutate: deleteScan, isPending: isDeleting } = useDeleteScan();

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background flex flex-col">
        <Header />
        <div className="flex-1 flex flex-col items-center justify-center gap-4">
          <Loader2 className="w-12 h-12 text-primary animate-spin" />
          <p className="text-muted-foreground font-medium">Loading details...</p>
        </div>
      </div>
    );
  }

  if (isError || !scan) {
    return (
      <div className="min-h-screen bg-background flex flex-col">
        <Header />
        <div className="flex-1 flex flex-col items-center justify-center gap-4 px-4 text-center">
          <AlertTriangle className="w-16 h-16 text-destructive/50" />
          <h2 className="text-2xl font-bold font-display">Scan not found</h2>
          <p className="text-muted-foreground">The scan you are looking for does not exist or has been deleted.</p>
          <Link href="/">
            <Button variant="outline" className="mt-4">Back to Home</Button>
          </Link>
        </div>
      </div>
    );
  }

  const isHealthy = scan.diseaseName?.toLowerCase().includes("healthy");
  const confidence = scan.confidence || 0;

  const handleDelete = () => {
    deleteScan(id, {
      onSuccess: () => setLocation("/history"),
    });
  };

  return (
    <div className="min-h-screen bg-background leaf-pattern font-sans pb-20">
      <Header />
      
      <main className="container max-w-5xl mx-auto px-4 py-8">
        <div className="mb-8 flex items-center justify-between">
          <Link href="/" className="inline-flex items-center gap-2 text-sm font-medium text-muted-foreground hover:text-foreground transition-colors group">
            <ArrowLeft className="w-4 h-4 group-hover:-translate-x-1 transition-transform" />
            Back to Dashboard
          </Link>

          <AlertDialog>
            <AlertDialogTrigger asChild>
              <button 
                className="text-muted-foreground hover:text-destructive transition-colors p-2 rounded-full hover:bg-destructive/10"
                disabled={isDeleting}
              >
                {isDeleting ? <Loader2 className="w-5 h-5 animate-spin" /> : <Trash2 className="w-5 h-5" />}
              </button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>Are you absolutely sure?</AlertDialogTitle>
                <AlertDialogDescription>
                  This action cannot be undone. This will permanently delete this scan record from our servers.
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Cancel</AlertDialogCancel>
                <AlertDialogAction onClick={handleDelete} className="bg-destructive text-destructive-foreground hover:bg-destructive/90">
                  Delete Scan
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-12">
          {/* Left Column: Image & Quick Stats */}
          <div className="space-y-6 animate-in slide-in-from-left-8 fade-in duration-700">
            <div className="relative rounded-3xl overflow-hidden border border-border shadow-2xl shadow-black/5 aspect-[4/3] bg-muted group">
              <img 
                src={scan.imageUrl} 
                alt="Analyzed plant" 
                className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-105"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent opacity-60" />
              <div className="absolute bottom-6 left-6 text-white">
                 <p className="text-sm font-medium opacity-90 mb-1">Scanned on</p>
                 <p className="text-lg font-bold">{scan.createdAt ? format(new Date(scan.createdAt), "MMMM d, yyyy 'at' h:mm a") : "Unknown Date"}</p>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className={cn(
                "p-6 rounded-2xl border flex flex-col items-center text-center gap-3 shadow-sm",
                isHealthy ? "bg-green-50 border-green-100 text-green-900" : "bg-amber-50 border-amber-100 text-amber-900"
              )}>
                <div className={cn(
                  "w-12 h-12 rounded-full flex items-center justify-center",
                  isHealthy ? "bg-green-100 text-green-600" : "bg-amber-100 text-amber-600"
                )}>
                  {isHealthy ? <CheckCircle className="w-6 h-6" /> : <AlertTriangle className="w-6 h-6" />}
                </div>
                <div>
                  <p className="text-sm font-medium opacity-80 uppercase tracking-wider text-xs">Diagnosis</p>
                  <p className="font-display font-bold text-lg leading-tight mt-1">{scan.diseaseName || "Unknown"}</p>
                </div>
              </div>

              <div className="p-6 rounded-2xl border border-border bg-card flex flex-col items-center text-center gap-3 shadow-sm">
                <div className="w-12 h-12 rounded-full bg-primary/10 text-primary flex items-center justify-center">
                  <Activity className="w-6 h-6" />
                </div>
                <div>
                  <p className="text-sm font-medium text-muted-foreground uppercase tracking-wider text-xs">Confidence</p>
                  <p className="font-display font-bold text-lg leading-tight mt-1">{confidence}%</p>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column: Analysis Details */}
          <div className="space-y-8 animate-in slide-in-from-right-8 fade-in duration-700 delay-150">
            <div>
              <div className="flex items-center gap-3 mb-4">
                <div className="p-2 bg-blue-100 text-blue-600 rounded-lg">
                  <Thermometer className="w-5 h-5" />
                </div>
                <h2 className="text-2xl font-display font-bold text-foreground">Detailed Analysis</h2>
              </div>
              <div className="glass-card p-6 rounded-3xl text-muted-foreground leading-relaxed">
                {scan.analysis ? (
                  <p className="whitespace-pre-wrap">{scan.analysis}</p>
                ) : (
                  <p className="italic text-muted-foreground/50">No detailed analysis available.</p>
                )}
              </div>
            </div>

            <div>
              <div className="flex items-center gap-3 mb-4">
                <div className="p-2 bg-purple-100 text-purple-600 rounded-lg">
                  <ShieldCheck className="w-5 h-5" />
                </div>
                <h2 className="text-2xl font-display font-bold text-foreground">Prevention & Treatment</h2>
              </div>
              <div className="glass-card p-6 rounded-3xl text-muted-foreground leading-relaxed border-l-4 border-l-primary">
                {scan.prevention ? (
                  <p className="whitespace-pre-wrap">{scan.prevention}</p>
                ) : (
                  <p className="italic text-muted-foreground/50">No prevention tips available.</p>
                )}
              </div>
            </div>

            <div className="bg-muted/30 p-6 rounded-2xl border border-dashed border-border">
              <h4 className="font-semibold text-foreground mb-2">Disclaimer</h4>
              <p className="text-sm text-muted-foreground">
                This AI-generated diagnosis is for informational purposes only. Always consult with a professional botanist or local agricultural extension for definitive advice.
              </p>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
