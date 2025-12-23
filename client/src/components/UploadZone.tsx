import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, Loader2, ImagePlus } from "lucide-react";
import { cn } from "@/lib/utils";

interface UploadZoneProps {
  onUpload: (base64: string) => void;
  isAnalyzing: boolean;
}

export function UploadZone({ onUpload, isAnalyzing }: UploadZoneProps) {
  const [preview, setPreview] = useState<string | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = () => {
        const result = reader.result as string;
        setPreview(result);
        onUpload(result);
      };
      reader.readAsDataURL(file);
    }
  }, [onUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.webp']
    },
    maxFiles: 1,
    disabled: isAnalyzing
  });

  return (
    <div className="w-full max-w-xl mx-auto">
      <div
        {...getRootProps()}
        className={cn(
          "relative overflow-hidden rounded-3xl border-2 border-dashed transition-all duration-300 cursor-pointer group",
          "h-80 flex flex-col items-center justify-center text-center p-8",
          isDragActive 
            ? "border-primary bg-primary/5 scale-[1.02]" 
            : "border-border hover:border-primary/50 hover:bg-muted/30",
          isAnalyzing && "pointer-events-none opacity-80"
        )}
      >
        <input {...getInputProps()} />

        {isAnalyzing ? (
          <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-background/80 backdrop-blur-sm">
            <Loader2 className="w-12 h-12 text-primary animate-spin mb-4" />
            <p className="text-lg font-medium text-foreground">Analyzing Plant Health...</p>
            <p className="text-sm text-muted-foreground mt-2">Our AI is diagnosing potential issues</p>
          </div>
        ) : preview ? (
          <div className="absolute inset-0 z-10 w-full h-full bg-black/5 p-2">
            <img 
              src={preview} 
              alt="Preview" 
              className="w-full h-full object-contain rounded-2xl shadow-inner" 
            />
            <div className="absolute inset-0 flex items-center justify-center bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity">
              <p className="text-white font-medium flex items-center gap-2">
                <ImagePlus className="w-5 h-5" />
                Upload different photo
              </p>
            </div>
          </div>
        ) : (
          <div className="z-10 flex flex-col items-center gap-4">
            <div className="w-20 h-20 rounded-full bg-primary/10 flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
              <Upload className="w-10 h-10 text-primary" />
            </div>
            <div className="space-y-2">
              <h3 className="text-xl font-display font-semibold text-foreground">
                {isDragActive ? "Drop it here!" : "Upload Plant Photo"}
              </h3>
              <p className="text-muted-foreground text-sm max-w-xs mx-auto">
                Drag and drop your image here, or click to browse files.
                Supports JPG, PNG, WEBP.
              </p>
            </div>
          </div>
        )}
        
        {/* Decorative background blobs */}
        <div className="absolute -top-10 -right-10 w-32 h-32 bg-primary/5 rounded-full blur-2xl pointer-events-none" />
        <div className="absolute -bottom-10 -left-10 w-32 h-32 bg-accent/10 rounded-full blur-2xl pointer-events-none" />
      </div>
    </div>
  );
}
