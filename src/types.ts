export interface AnalysisResult {
  isAIGenerated: boolean;
  confidenceScore: number;
  metadata: {
    resolution: string;
    format: string;
    size: string;
    duration?: string; // Optional, for videos only
    frameRate?: string; // Optional, for videos only
  };
  detectionMethods: {
    name: string;
    score: number;
    details: {
      technique: string;
      explanation: string;
    }[];
  }[];
}
