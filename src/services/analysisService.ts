/**
 * Analysis Service - Handles AI content analysis and detection
 * 
 * @author T. Landon Love <12stonedesigns@gmail.com>
 * @copyright Copyright (c) 2024 T. Landon Love
 * @license MIT
 */

import * as tf from '@tensorflow/tfjs';
import { loadGraphModel } from '@tensorflow/tfjs-converter';
import { FFmpeg } from '@ffmpeg/ffmpeg';
import { fetchFile } from '@ffmpeg/util';
import * as exifr from 'exifr';

// Initialize FFmpeg for video processing
const ffmpeg = new FFmpeg();
let ffmpegLoaded = false;

// Load FFmpeg
const loadFFmpeg = async () => {
  if (!ffmpegLoaded) {
    await ffmpeg.load();
    ffmpegLoaded = true;
  }
};

// Models for different aspects of detection
interface Models {
  patternDetector: tf.GraphModel;
  artifactAnalyzer: tf.GraphModel;
  frequencyAnalyzer: tf.GraphModel;
}

let models: Models | null = null;

// Function to load all models
async function loadModels(): Promise<void> {
  if (models) return; // Only load once
  
  try {
    const patternModel = await loadGraphModel('/models/pattern_detector/model.json');
    const artifactModel = await loadGraphModel('/models/artifact_analyzer/model.json');
    const frequencyModel = await loadGraphModel('/models/frequency_analyzer/model.json');

    models = {
      patternDetector: patternModel,
      artifactAnalyzer: artifactModel,
      frequencyAnalyzer: frequencyModel
    };
  } catch (error) {
    console.error('Error loading models:', error);
    throw error;
  }
}

// Evaluation metrics tracking
interface EvaluationMetrics {
  truePositives: number;
  falsePositives: number;
  trueNegatives: number;
  falseNegatives: number;
}

let evaluationMetrics: EvaluationMetrics = {
  truePositives: 0,
  falsePositives: 0,
  trueNegatives: 0,
  falseNegatives: 0
};

// Function to update evaluation metrics
function updateEvaluationMetrics(prediction: boolean, actual: boolean): void {
  if (prediction && actual) evaluationMetrics.truePositives++;
  else if (prediction && !actual) evaluationMetrics.falsePositives++;
  else if (!prediction && !actual) evaluationMetrics.trueNegatives++;
  else evaluationMetrics.falseNegatives++;
}

// Convert image to frequency domain for analysis
async function getFrequencyDomain(tensor: tf.Tensor3D): Promise<tf.Tensor> {
  // Convert to grayscale if not already
  const grayscale = tensor.mean(2, true);
  
  // Perform FFT using real FFT
  const realFFT = tf.spectral.rfft(grayscale.squeeze(), grayscale.shape[0]);
  
  // Get magnitude spectrum
  const magnitudeSpectrum = tf.abs(realFFT);
  
  // Normalize and return
  return tf.div(magnitudeSpectrum, tf.max(magnitudeSpectrum));
}

// Analysis functions with null checks and proper tensor data extraction
async function analyzePatterns(tensor: tf.Tensor): Promise<number[]> {
  if (!models?.patternDetector) throw new Error('Pattern detector model not loaded');
  const prediction = models.patternDetector.predict(tensor) as tf.Tensor;
  const data = await prediction.data();
  prediction.dispose();
  return Array.from(data);
}

async function analyzeArtifacts(tensor: tf.Tensor): Promise<number[]> {
  if (!models?.artifactAnalyzer) throw new Error('Artifact analyzer model not loaded');
  const prediction = models.artifactAnalyzer.predict(tensor) as tf.Tensor;
  const data = await prediction.data();
  prediction.dispose();
  return Array.from(data);
}

async function analyzeFrequencyDomain(tensor: tf.Tensor): Promise<number[]> {
  if (!models?.frequencyAnalyzer) throw new Error('Frequency analyzer model not loaded');
  const freqDomain = await getFrequencyDomain(tensor as tf.Tensor3D);
  const prediction = models.frequencyAnalyzer.predict(freqDomain) as tf.Tensor;
  const data = await prediction.data();
  prediction.dispose();
  freqDomain.dispose();
  return Array.from(data);
}

const extractMetadata = async (file: File) => {
  try {
    // Extract EXIF metadata
    const metadata = await exifr.parse(file);
    
    // Check for common AI tool signatures
    const aiToolSignatures = [
      'DALL-E', 'Midjourney', 'Stable Diffusion',
      'GAN', 'StyleGAN', 'Adobe Firefly'
    ];
    
    const hasAISignature = metadata?.Software && 
      aiToolSignatures.some(sig => 
        metadata.Software.toLowerCase().includes(sig.toLowerCase())
      );

    return {
      hasAISignature,
      metadata
    };
  } catch (error) {
    console.error('Error extracting metadata:', error);
    return { hasAISignature: false, metadata: null };
  }
};

const preprocessImage = async (imageData: ImageData | HTMLImageElement): Promise<tf.Tensor> => {
  // Convert image to tensor
  const tensor = tf.browser.fromPixels(imageData);
  
  // Create multiple resolutions for analysis
  const resolutions = [
    tf.image.resizeBilinear(tensor, [224, 224]),
    tf.image.resizeBilinear(tensor, [512, 512]),
    tf.image.resizeBilinear(tensor, [1024, 1024])
  ];
  
  // Normalize pixel values
  const normalized = resolutions.map(t => 
    t.toFloat().div(tf.scalar(127.5)).sub(tf.scalar(1))
  );
  
  // Stack tensors for batch processing
  return tf.stack(normalized);
};

const createImageElement = async (file: File): Promise<HTMLImageElement> => {
  const img = new Image();
  const imageUrl = URL.createObjectURL(file);
  await new Promise((resolve) => {
    img.onload = resolve;
    img.src = imageUrl;
  });
  return img;
};

const analyzeImage = async (file: File, groundTruth?: boolean) => {
  try {
    await loadModels(); // Ensure models are loaded

    // Create image element
    const img = await createImageElement(file);
    
    // Convert to tensor
    const tensor = await preprocessImage(img);
    
    // Get predictions from each model
    const [patternScores, artifactScores, frequencyScores] = await Promise.all([
      analyzePatterns(tensor),
      analyzeArtifacts(tensor),
      analyzeFrequencyDomain(tensor)
    ]);

    // Extract metadata
    const metadata = await extractMetadata(file);
    const hasAISignature = metadata?.hasAISignature || false;

    // Calculate weighted ensemble score
    const ensembleScore = 
      patternScores[0] * 0.3 +
      artifactScores[0] * 0.25 +
      (hasAISignature ? 1 : 0) * 0.2 +
      frequencyScores[0] * 0.25;

    // Final decision based on ensemble score
    const isAIGenerated = ensembleScore > 0.85;

    // Update evaluation metrics if ground truth is provided
    if (groundTruth !== undefined) {
      updateEvaluationMetrics(isAIGenerated, groundTruth);
    }

    // Cleanup tensors
    tensor.dispose();

    return {
      isAIGenerated,
      confidence: ensembleScore,
      details: {
        patternScore: patternScores[0],
        artifactScore: artifactScores[0],
        metadataScore: hasAISignature ? 1 : 0,
        frequencyScore: frequencyScores[0],
        confidenceLevels: {
          pattern: patternScores[0] > 0.85,
          artifact: artifactScores[0] > 0.80,
          metadata: hasAISignature,
          frequency: frequencyScores[0] > 0.85
        }
      }
    };

  } catch (error) {
    console.error('Error analyzing image:', error);
    throw error;
  }
};

const analyzeVideo = async (videoFile: File, groundTruth?: boolean) => {
  try {
    await loadFFmpeg();
    
    const inputFileName = 'input.mp4';
    const outputFileName = 'frame_%d.jpg';
    
    // Write video file to FFmpeg's virtual file system
    await ffmpeg.writeFile(inputFileName, await fetchFile(videoFile));
    
    // Extract frames
    await ffmpeg.exec([
      '-i', inputFileName,
      '-vf', 'fps=1',
      '-frame_pts', '1',
      outputFileName
    ]);
    
    // Read frames and analyze
    const frames = await ffmpeg.listDir('.');
    const frameFiles = frames.filter(f => f.name.startsWith('frame_'));
    
    // Process each frame...
    const frameAnalyses = await Promise.all(frameFiles.map(async (frameFile) => {
      const frameData = await ffmpeg.readFile(frameFile.name);
      const blob = new Blob([frameData], { type: 'image/jpeg' });
      const file = new File([blob], 'frame.jpg', { type: 'image/jpeg' });
      return analyzeImage(file);
    }));

    // Calculate temporal consistency
    const temporalScores = frameAnalyses.map(analysis => analysis.confidence);
    const temporalConsistency = calculateTemporalConsistency(temporalScores);

    // Calculate weighted ensemble score for video
    const averageEnsembleScore = temporalScores.reduce((a, b) => a + b, 0) / temporalScores.length;
    const finalScore = averageEnsembleScore * 0.7 + temporalConsistency * 0.3;

    // Get video metadata
    const video = document.createElement('video');
    const duration = await new Promise<number>((resolve) => {
      video.onloadedmetadata = () => resolve(video.duration);
      video.src = URL.createObjectURL(videoFile);
    });

    const isAIGenerated = finalScore > 0.85;

    // Update evaluation metrics if ground truth is provided
    if (groundTruth !== undefined) {
      updateEvaluationMetrics(isAIGenerated, groundTruth);
    }

    return {
      isAIGenerated,
      confidence: finalScore,
      metadata: {
        format: videoFile.type.split('/')[1].toUpperCase(),
        size: `${(videoFile.size / (1024 * 1024)).toFixed(2)} MB`,
        duration: `${duration.toFixed(1)} seconds`,
        frameCount: frames.length,
        temporalConsistency
      },
      detectionMethods: [
        {
          name: 'Temporal Consistency Analysis',
          score: temporalConsistency,
          details: [{
            technique: 'Frame Sequence Analysis',
            explanation: `Temporal consistency score: ${(temporalConsistency * 100).toFixed(1)}%`
          }]
        },
        {
          name: 'Frame-by-Frame Analysis',
          score: averageEnsembleScore,
          details: [{
            technique: 'Multi-Frame Ensemble',
            explanation: `Average frame analysis confidence: ${(averageEnsembleScore * 100).toFixed(1)}%`
          }]
        }
      ]
    };
  } catch (error) {
    console.error('Error analyzing video:', error);
    throw error;
  }
};

// Helper function to calculate temporal consistency
const calculateTemporalConsistency = (scores: number[]): number => {
  if (scores.length < 2) return 1;
  
  let consistency = 0;
  for (let i = 1; i < scores.length; i++) {
    const diff = Math.abs(scores[i] - scores[i-1]);
    consistency += 1 - diff;
  }
  
  return consistency / (scores.length - 1);
};

const analysisService = {
  analyzeImage,
  analyzeVideo
};

export { analysisService };
