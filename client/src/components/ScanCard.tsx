import { Link } from "wouter";
import { ArrowRight, Calendar, AlertTriangle, CheckCircle, Leaf } from "lucide-react";
import { format } from "date-fns";
import { type Scan } from "@shared/schema";
import { cn } from "@/lib/utils";

interface ScanCardProps {
  scan: Scan;
}

export function ScanCard({ scan }: ScanCardProps) {
  const confidence = scan.confidence || 0;
  const isHealthy = scan.diseaseName?.toLowerCase().includes("healthy");

  return (
    <div className="group relative bg-card hover:bg-card/50 border border-border/50 hover:border-primary/30 rounded-2xl overflow-hidden transition-all duration-300 hover:shadow-lg hover:-translate-y-1">
      <Link href={`/scan/${scan.id}`} className="absolute inset-0 z-10">
        <span className="sr-only">View scan details</span>
      </Link>
      
      <div className="aspect-[4/3] relative overflow-hidden bg-muted">
        <img 
          src={scan.imageUrl} 
          alt={scan.diseaseName || "Scan"} 
          className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
        />
        <div className="absolute top-3 right-3">
          <span className={cn(
            "px-3 py-1 rounded-full text-xs font-semibold backdrop-blur-md shadow-sm border border-white/10 flex items-center gap-1.5",
            isHealthy 
              ? "bg-green-500/90 text-white" 
              : "bg-amber-500/90 text-white"
          )}>
            {isHealthy ? <CheckCircle className="w-3 h-3" /> : <AlertTriangle className="w-3 h-3" />}
            {confidence}% Confidence
          </span>
        </div>
      </div>
      
      <div className="p-5">
        <div className="flex justify-between items-start mb-2">
          <h3 className="font-display font-semibold text-lg text-foreground line-clamp-1 group-hover:text-primary transition-colors">
            {scan.diseaseName || "Unknown Issue"}
          </h3>
        </div>
        
        <p className="text-muted-foreground text-sm line-clamp-2 mb-4 h-10">
          {scan.analysis || "No analysis details available."}
        </p>

        <div className="flex items-center justify-between text-xs text-muted-foreground mt-auto pt-4 border-t border-border/50">
          <div className="flex items-center gap-1.5">
            <Calendar className="w-3.5 h-3.5" />
            {scan.createdAt ? format(new Date(scan.createdAt), 'MMM d, yyyy') : 'Unknown Date'}
          </div>
          <div className="flex items-center gap-1 text-primary font-medium opacity-0 group-hover:opacity-100 transition-opacity -translate-x-2 group-hover:translate-x-0 duration-300">
            View Details <ArrowRight className="w-3.5 h-3.5" />
          </div>
        </div>
      </div>
    </div>
  );
}
