import { useLocation } from "wouter";
import { Header } from "@/components/Header";
import { UploadZone } from "@/components/UploadZone";
import { useCreateScan, useScans } from "@/hooks/use-scans";
import { ScanCard } from "@/components/ScanCard";
import { Loader2, ArrowRight } from "lucide-react";
import { Link } from "wouter";

export default function Home() {
  const [, setLocation] = useLocation();
  const { mutate: createScan, isPending } = useCreateScan();
  const { data: recentScans, isLoading: isLoadingHistory } = useScans();

  const handleUpload = (base64: string) => {
    createScan(
      { image: base64 },
      {
        onSuccess: (data) => {
          setLocation(`/scan/${data.id}`);
        },
      }
    );
  };

  return (
    <div className="min-h-screen bg-background leaf-pattern font-sans">
      <Header />
      
      <main className="container max-w-7xl mx-auto px-4 py-12 md:py-20">
        {/* Hero Section */}
        <div className="text-center max-w-3xl mx-auto mb-16 space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-700">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-accent/10 text-accent-foreground text-sm font-semibold mb-2 border border-accent/20">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-accent opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-accent"></span>
            </span>
            AI-Powered Disease Detection
          </div>
          
          <h1 className="text-4xl md:text-6xl lg:text-7xl font-display font-bold text-foreground leading-tight tracking-tight text-balance">
            Heal your plants with <span className="text-primary italic relative">
              AI precision
              <svg className="absolute w-full h-3 -bottom-1 left-0 text-primary/20 -z-10" viewBox="0 0 100 10" preserveAspectRatio="none">
                <path d="M0 5 Q 50 10 100 5" stroke="currentColor" strokeWidth="8" fill="none" />
              </svg>
            </span>
          </h1>
          
          <p className="text-lg md:text-xl text-muted-foreground text-balance max-w-2xl mx-auto">
            Upload a photo of your plant. Our advanced AI instantly identifies diseases and provides expert treatment advice to keep your garden thriving.
          </p>
        </div>

        {/* Upload Section */}
        <div className="mb-24 animate-in fade-in slide-in-from-bottom-8 duration-700 delay-150">
          <UploadZone onUpload={handleUpload} isAnalyzing={isPending} />
        </div>

        {/* Recent Scans Section */}
        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-8 duration-700 delay-300">
          <div className="flex items-center justify-between border-b border-border pb-4">
            <h2 className="text-2xl font-display font-bold text-foreground">Recent Scans</h2>
            <Link href="/history" className="text-primary hover:text-primary/80 font-medium flex items-center gap-1 transition-colors group">
              View All History <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </Link>
          </div>

          {isLoadingHistory ? (
            <div className="flex justify-center py-12">
              <Loader2 className="w-8 h-8 text-primary animate-spin" />
            </div>
          ) : recentScans && recentScans.length > 0 ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
              {recentScans.slice(0, 3).map((scan) => (
                <ScanCard key={scan.id} scan={scan} />
              ))}
            </div>
          ) : (
            <div className="text-center py-12 bg-muted/30 rounded-3xl border border-dashed border-border">
              <p className="text-muted-foreground">No previous scans found. Start by uploading a photo above!</p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
