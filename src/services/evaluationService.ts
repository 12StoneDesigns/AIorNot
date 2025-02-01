import { analysisService } from './analysisService';
import fs from 'fs';
import path from 'path';

interface TestCase {
  filePath: string;
  isAIGenerated: boolean;
}

interface EvaluationResult {
  truePositives: number;
  trueNegatives: number;
  falsePositives: number;
  falseNegatives: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
}

class EvaluationService {
  private testCases: TestCase[] = [];
  private results: EvaluationResult = {
    truePositives: 0,
    trueNegatives: 0,
    falsePositives: 0,
    falseNegatives: 0,
    accuracy: 0,
    precision: 0,
    recall: 0,
    f1Score: 0
  };

  async loadTestCases(datasetPath: string): Promise<void> {
    const files = fs.readdirSync(datasetPath);
    this.testCases = files.map(file => ({
      filePath: path.join(datasetPath, file),
      isAIGenerated: file.startsWith('ai_') // Assuming AI-generated files are prefixed with 'ai_'
    }));
  }

  async evaluate(): Promise<EvaluationResult> {
    for (const testCase of this.testCases) {
      const file = fs.readFileSync(testCase.filePath);
      const blob = new Blob([file]);
      const fileObj = new File([blob], path.basename(testCase.filePath));

      let result;
      if (testCase.filePath.endsWith('.mp4') || testCase.filePath.endsWith('.mov')) {
        result = await analysisService.analyzeVideo(fileObj);
      } else {
        result = await analysisService.analyzeImage(fileObj);
      }

      const prediction = result.isAIGenerated;
      
      if (prediction && testCase.isAIGenerated) {
        this.results.truePositives++;
      } else if (!prediction && !testCase.isAIGenerated) {
        this.results.trueNegatives++;
      } else if (prediction && !testCase.isAIGenerated) {
        this.results.falsePositives++;
      } else {
        this.results.falseNegatives++;
      }
    }

    this.calculateMetrics();
    return this.results;
  }

  private calculateMetrics(): void {
    const total = this.testCases.length;
    const { truePositives, trueNegatives, falsePositives, falseNegatives } = this.results;

    this.results.accuracy = (truePositives + trueNegatives) / total;
    this.results.precision = truePositives / (truePositives + falsePositives);
    this.results.recall = truePositives / (truePositives + falseNegatives);
    this.results.f1Score = 2 * (this.results.precision * this.results.recall) / 
      (this.results.precision + this.results.recall);
  }

  generateReport(): string {
    return `
Evaluation Report:
=================
True Positives: ${this.results.truePositives}
True Negatives: ${this.results.trueNegatives}
False Positives: ${this.results.falsePositives}
False Negatives: ${this.results.falseNegatives}

Metrics:
--------
Accuracy: ${(this.results.accuracy * 100).toFixed(2)}%
Precision: ${(this.results.precision * 100).toFixed(2)}%
Recall: ${(this.results.recall * 100).toFixed(2)}%
F1 Score: ${(this.results.f1Score * 100).toFixed(2)}%
`;
  }
}

export const evaluationService = new EvaluationService();
