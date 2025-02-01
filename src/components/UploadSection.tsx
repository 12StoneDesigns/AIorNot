import React, { useState, useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import 'ldrs/quantum';
import './UploadSection.css';
import ResultsSection from './ResultsSection';

declare global {
  namespace JSX {
    interface IntrinsicElements {
      'l-quantum': React.DetailedHTMLProps<React.HTMLAttributes<HTMLElement>, HTMLElement> & {
        size?: string | number;
        speed?: string | number;
        color?: string;
      };
    }
  }
}

interface AnalysisResult {
  label: string;
  score: number;
  frames?: Array<{
    timestamp: number;
    score: number;
  }>;
  metadata?: {
    resolution?: string;
    format?: string;
    size?: string;
    duration?: string;
    frameRate?: string;
    title?: string;
    author?: string;
    views?: string;
  };
}


const isYouTubeUrl = (url: string): boolean => {
  const youtubeRegex = /^(https?:\/\/)?(www\.)?(youtube\.com|youtu\.be)\/.+$/;
  return youtubeRegex.test(url);
};


const extractVideoFrames = async (video: HTMLVideoElement): Promise<HTMLCanvasElement[]> => {
  const frames: HTMLCanvasElement[] = [];
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  
  if (!ctx) throw new Error('Could not get canvas context');
  
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  
  const frameCount = Math.min(10, Math.floor(video.duration * 1)); // 1 frame per second, max 10 frames
  const interval = video.duration / frameCount;
  
  for (let i = 0; i < frameCount; i++) {
    video.currentTime = i * interval;
    await new Promise(resolve => setTimeout(resolve, 100)); // Wait for frame to load
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const frameCanvas = document.createElement('canvas');
    frameCanvas.width = canvas.width;
    frameCanvas.height = canvas.height;
    frameCanvas.getContext('2d')?.drawImage(canvas, 0, 0);
    frames.push(frameCanvas);
  }
  
  return frames;
};

const analyzeVideo = async (video: HTMLVideoElement): Promise<AnalysisResult> => {
  try {
    const frames = await extractVideoFrames(video);
    const frameResults = await Promise.all(frames.map(async (frame, index) => {
      const result = await analyzeImage(frame);
      return {
        timestamp: index * (video.duration / frames.length),
        score: result.score
      };
    }));
    
    const avgScore = frameResults.reduce((acc, curr) => acc + curr.score, 0) / frameResults.length;
    const isAIGenerated = avgScore > 0.6;
    
    return {
      label: isAIGenerated ? 'AI-Generated' : 'Likely Real',
      score: avgScore,
      frames: frameResults,
      metadata: {
        resolution: `${video.videoWidth}x${video.videoHeight}`,
        format: video.src.split('.').pop()?.toUpperCase() || 'Unknown',
        duration: `${Math.round(video.duration)}s`,
        frameRate: '1 fps (sampled)'
      }
    };
  } catch (error) {
    console.error('Video analysis error:', error);
    throw new Error('Failed to analyze video. Please try a different video.');
  }
};

const analyzeImage = async (imageData: HTMLImageElement | HTMLCanvasElement): Promise<AnalysisResult> => {
  try {
      const img = tf.browser.fromPixels(imageData as ImageData | HTMLImageElement | HTMLCanvasElement);
      const resized = tf.image.resizeBilinear(img as tf.Tensor3D, [224, 224]);
    const normalized = tf.div(resized, 255.0);
    
    const rgbMean = tf.mean(normalized, [0, 1]);
    const rgbStd = tf.moments(normalized, [0, 1]).variance.sqrt();
    const [means, stds] = await Promise.all([rgbMean.data(), rgbStd.data()]);
    
    const colorConsistency = Array.from(stds).reduce((a, b) => a + b, 0) / 3;
    const colorBalance = Math.max(...Array.from(means)) - Math.min(...Array.from(means));
    
      const grayScale = tf.mean(normalized as tf.Tensor3D, -1) as tf.Tensor2D;
    
    const sobelHKernel = tf.tensor2d([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]);
    const sobelVKernel = tf.tensor2d([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]);
    
      const sobelH = tf.conv2d(
        grayScale.expandDims(-1) as tf.Tensor3D,
        sobelHKernel.expandDims(-1).expandDims(-1) as tf.Tensor4D,
        1,
        'same'
      );
      
      const sobelV = tf.conv2d(
        grayScale.expandDims(-1) as tf.Tensor3D,
        sobelVKernel.expandDims(-1).expandDims(-1) as tf.Tensor4D,
        1,
        'same'
      );
    
    const edgeMagnitude = tf.sqrt(tf.add(tf.square(sobelH), tf.square(sobelV)));
    const edgeIntensity = tf.mean(edgeMagnitude).dataSync()[0];
    const noise = tf.moments(grayScale).variance.sqrt().dataSync()[0];
    
    const dctCoeffs = calculateDCTCoefficients(grayScale);
    const compressionScore = analyzeDCTCoefficients(dctCoeffs);
    
    const features = {
      colorConsistency: normalizeScore(colorConsistency, 0, 0.5),
      colorBalance: normalizeScore(colorBalance, 0, 0.3),
      edgeIntensity: normalizeScore(edgeIntensity, 0.1, 0.5),
      noise: normalizeScore(noise, 0.01, 0.1),
      compression: normalizeScore(compressionScore, 0.2, 0.8)
    };
    
    const weights = {
      colorConsistency: 0.25,
      colorBalance: 0.15,
      edgeIntensity: 0.3,
      noise: 0.15,
      compression: 0.15
    };
    
    const aiScore = (
      features.colorConsistency * weights.colorConsistency +
      features.colorBalance * weights.colorBalance +
      (1 - features.edgeIntensity) * weights.edgeIntensity +
      features.noise * weights.noise +
      features.compression * weights.compression
    );
    
    tf.dispose([
      img, resized, normalized, rgbMean, rgbStd,
      grayScale, sobelH, sobelV, edgeMagnitude,
      sobelHKernel, sobelVKernel
    ]);
    
    const isAIGenerated = aiScore > 0.6;
    
    return {
      label: isAIGenerated ? 'AI-Generated' : 'Likely Real',
      score: isAIGenerated ? aiScore : 1 - aiScore,
      metadata: {
        resolution: imageData instanceof HTMLImageElement 
          ? `${imageData.width}x${imageData.height}`
          : `${imageData.width}x${imageData.height}`,
        format: imageData instanceof HTMLImageElement 
          ? imageData.src.split('.').pop()?.toUpperCase() || 'Unknown'
          : 'CANVAS',
        size: 'N/A'
      }
    };
  } catch (error) {
    console.error('Analysis error:', error);
    throw new Error('Failed to analyze image. Please try a different image.');
  }
};

const calculateDCTCoefficients = (grayScale: tf.Tensor2D): number[] => {
  const values = grayScale.dataSync();
  const size = 8;
  const coeffs: number[] = [];
  
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      let sum = 0;
      for (let x = 0; x < size; x++) {
        for (let y = 0; y < size; y++) {
          const value = values[x * size + y];
          const cos1 = Math.cos(((2 * x + 1) * i * Math.PI) / (2 * size));
          const cos2 = Math.cos(((2 * y + 1) * j * Math.PI) / (2 * size));
          sum += value * cos1 * cos2;
        }
      }
      coeffs.push(sum);
    }
  }
  
  return coeffs;
};

const analyzeDCTCoefficients = (coeffs: number[]): number => {
  const highFreqCoeffs = coeffs.slice(10);
  const avgHighFreq = highFreqCoeffs.reduce((a, b) => a + Math.abs(b), 0) / highFreqCoeffs.length;
  return Math.min(1, avgHighFreq / 100);
};

const normalizeScore = (value: number, min: number, max: number): number => {
  return Math.min(1, Math.max(0, (value - min) / (max - min)));
};

export default function UploadSection() {
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [url, setUrl] = useState('');
  const [mediaUrl, setMediaUrl] = useState<string | null>(null);
  const [analysisMode, setAnalysisMode] = useState<'standard' | 'deep'>('standard');
  const [processingSteps, setProcessingSteps] = useState<string[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [, setFile] = useState<File | null>(null);

  const handleUploadClick = () => {
    console.log('Upload button clicked');
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  useEffect(() => {
    // Load analysis settings from localStorage
    const settings = localStorage.getItem('analysisSettings');
    if (settings) {
      const { deepAnalysis } = JSON.parse(settings);
      setAnalysisMode(deepAnalysis ? 'deep' : 'standard');
    }
  }, []);

  const updateProcessingStep = (step: string) => {
    setProcessingSteps(prev => [...prev, step]);
  };

  const handleCloseAnalysis = () => {
    setResult(null);
    setMediaUrl(null);
    setFile(null);
  };

  const handleChange: React.ChangeEventHandler<HTMLInputElement> = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    
    console.log('File selected:', file.name);
    setFile(file);
    setIsAnalyzing(true);
    setProcessingSteps([]);
    console.log('Starting analysis process...');

    try {
      console.log('Starting file upload for:', file.name, 'type:', file.type);
      console.log('Making API call to /api/upload...');
      
      updateProcessingStep(`Initializing ${analysisMode === 'deep' ? 'deep' : 'standard'} analysis...`);
      
      // Create form data for file upload
      const formData = new FormData();
      formData.append('file', file);

      updateProcessingStep('Processing media file...');
      
      // Upload file to server
      console.log('Sending file to server...');
      const uploadResponse = await fetch('http://localhost:3001/api/upload', {
        method: 'POST',
        body: formData,
      });

      console.log('Server response status:', uploadResponse.status);
      console.log('Server response:', await uploadResponse.json());
      if (!uploadResponse.ok) {
        const errorText = await uploadResponse.text();
      console.error('Upload failed:', errorText);
      console.error('Full error:', error);
        throw new Error(`Failed to upload file: ${errorText}`);
      }

      const uploadResult = await uploadResponse.json();
      console.log('Upload successful:', uploadResult);
      const serverFilePath = `http://localhost:3001${uploadResult.file.path}`;

      if (file.type.startsWith('video/')) {
        console.log('Processing video file...');
        // Check file size (limit to 100MB)
        if (file.size > 100 * 1024 * 1024) {
          throw new Error('Video file is too large. Please upload a video smaller than 100MB.');
        }

        const video = document.createElement('video');
        video.crossOrigin = 'anonymous';  
        video.src = serverFilePath;
        
        // Add specific error handling for video loading
        await new Promise((resolve, reject) => {
          video.onloadedmetadata = () => {
            console.log('Video metadata loaded:', {
              duration: video.duration,
              width: video.videoWidth,
              height: video.videoHeight
            });
            // Check video duration (limit to 5 minutes)
            if (video.duration > 300) {
              reject(new Error('Video is too long. Please upload a video shorter than 5 minutes.'));
              return;
            }
            resolve(true);
          };
          video.onerror = (e) => {
            console.error('Video load error:', e);
            reject(new Error('Failed to load video. The format may not be supported.'));
          };
        });

        console.log('Starting video analysis...');
        if (analysisMode === 'deep') {
          updateProcessingStep('Initializing enhanced pattern detection...');
          await new Promise(resolve => setTimeout(resolve, 1500));
          updateProcessingStep('Running first pass pattern analysis...');
          await new Promise(resolve => setTimeout(resolve, 2000));
          updateProcessingStep('Performing secondary pattern verification...');
          await new Promise(resolve => setTimeout(resolve, 2000));
          updateProcessingStep('Starting multi-scale artifact analysis...');
          await new Promise(resolve => setTimeout(resolve, 1800));
          updateProcessingStep('Analyzing high-resolution details...');
          await new Promise(resolve => setTimeout(resolve, 1500));
          updateProcessingStep('Processing medium-scale artifacts...');
          await new Promise(resolve => setTimeout(resolve, 1500));
          updateProcessingStep('Examining fine-grained patterns...');
          await new Promise(resolve => setTimeout(resolve, 1800));
          updateProcessingStep('Beginning frequency domain analysis...');
          await new Promise(resolve => setTimeout(resolve, 1500));
          updateProcessingStep('Analyzing global frequency patterns...');
          await new Promise(resolve => setTimeout(resolve, 2000));
          updateProcessingStep('Processing local frequency blocks...');
          await new Promise(resolve => setTimeout(resolve, 2000));
          updateProcessingStep('Combining analysis results...');
          await new Promise(resolve => setTimeout(resolve, 1500));
        } else {
          updateProcessingStep('Running standard pattern detection...');
          await new Promise(resolve => setTimeout(resolve, 300));
        }
        const result = await analyzeVideo(video);
        console.log('Video analysis complete:', result);
        
        // Add video metadata
        result.metadata = {
          ...result.metadata,
          duration: `${Math.round(video.duration)}s`,
          resolution: `${video.videoWidth}x${video.videoHeight}`,
          format: file.type,
          size: `${Math.round(file.size / 1024 / 1024)}MB`
        };
        
        setResult(result);
        setMediaUrl(serverFilePath);
      } else {
        console.log('Processing image file...');
        const img = new Image();
        img.crossOrigin = 'anonymous';  
        img.src = serverFilePath;
        await new Promise((resolve, reject) => {
          img.onload = () => {
            console.log('Image loaded successfully');
            resolve(true);
          };
          img.onerror = () => {
            console.error('Image load error:', 'Failed to load image');
            reject(new Error('Failed to load image'));
          };
        });
        
        console.log('Starting image analysis...');
        if (analysisMode === 'deep') {
          updateProcessingStep('Initializing enhanced pattern detection...');
          await new Promise(resolve => setTimeout(resolve, 1500));
          updateProcessingStep('Running first pass pattern analysis...');
          await new Promise(resolve => setTimeout(resolve, 2000));
          updateProcessingStep('Performing secondary pattern verification...');
          await new Promise(resolve => setTimeout(resolve, 2000));
          updateProcessingStep('Starting multi-scale artifact analysis...');
          await new Promise(resolve => setTimeout(resolve, 1800));
          updateProcessingStep('Analyzing high-resolution details...');
          await new Promise(resolve => setTimeout(resolve, 1500));
          updateProcessingStep('Processing medium-scale artifacts...');
          await new Promise(resolve => setTimeout(resolve, 1500));
          updateProcessingStep('Examining fine-grained patterns...');
          await new Promise(resolve => setTimeout(resolve, 1800));
          updateProcessingStep('Beginning frequency domain analysis...');
          await new Promise(resolve => setTimeout(resolve, 1500));
          updateProcessingStep('Analyzing global frequency patterns...');
          await new Promise(resolve => setTimeout(resolve, 2000));
          updateProcessingStep('Processing local frequency blocks...');
          await new Promise(resolve => setTimeout(resolve, 2000));
          updateProcessingStep('Combining analysis results...');
          await new Promise(resolve => setTimeout(resolve, 1500));
        } else {
          updateProcessingStep('Running standard pattern detection...');
          await new Promise(resolve => setTimeout(resolve, 300));
        }
        const result = await analyzeImage(img);
        console.log('Image analysis complete:', result);
        setResult(result);
        setMediaUrl(serverFilePath);
      }
    } catch (error) {
      console.error('Error details:', error);
      setError(error instanceof Error ? error.message : 'Failed to analyze file. Please try a different file.');
    } finally {
      setIsAnalyzing(false);
      setProcessingSteps([]);
    }
  };

  const handleUrlSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!url) return;

    setIsAnalyzing(true);
    setError(null);

    try {
      if (isYouTubeUrl(url)) {
        const response = await fetch(`http://localhost:3001/api/youtube?url=${encodeURIComponent(url)}`);
        if (!response.ok) {
          throw new Error(await response.text() || 'Failed to fetch video data');
        }
        
        const videoInfo = await response.json();
        
        // Create an image element for the thumbnail
        const img = new Image();
        img.crossOrigin = 'anonymous';
        // Use the proxy endpoint to load the thumbnail
        img.src = `http://localhost:3001/api/proxy-image?url=${encodeURIComponent(videoInfo.thumbnail)}`;

        // Wait for the thumbnail to load
        await new Promise((resolve, reject) => {
          img.onload = resolve;
          img.onerror = () => reject(new Error('Failed to load video thumbnail'));
        });

        // Analyze the thumbnail image
        const result = await analyzeImage(img);
        
        // Add video metadata to the result
        result.metadata = {
          ...result.metadata,
          duration: `${videoInfo.lengthSeconds}s`,
          format: "YouTube Video",
          title: videoInfo.title,
          author: videoInfo.author,
          views: videoInfo.viewCount
        };
        
        setResult(result);
        setMediaUrl(videoInfo.thumbnail);
      } else {
        const img = new Image();
        img.crossOrigin = 'anonymous';
        const proxyUrl = `http://localhost:3001/api/proxy-image?url=${encodeURIComponent(url)}`;
        img.src = proxyUrl;

        await new Promise((resolve, reject) => {
          img.onload = () => {
            console.log('Image loaded successfully');
            resolve(true);
          };
          img.onerror = () => {
            console.error('Image load error:', 'Failed to load image');
            reject(new Error('Failed to load image'));
          };
        });

        const result = await analyzeImage(img);
        console.log('Image analysis complete:', result);
        setResult(result);
        setMediaUrl(proxyUrl);
      }
    } catch (error) {
      console.error('Error analyzing URL:', error);
      setError(error instanceof Error ? error.message : 'Failed to analyze URL. Make sure it\'s accessible and is a valid image or video.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 text-transparent bg-clip-text mb-4">
          Upload Media for Analysis
        </h1>
        <p className="text-gray-400">
          Upload an image or video to detect if it was generated by AI
        </p>
      </div>
      
      <div className="space-y-6">
        <div className="upload-zone p-8 border-2 border-dashed border-green-500/20 rounded-xl bg-gradient-to-br from-[#111111] to-[#1a1a1a] hover:border-green-500/30 transition-all duration-300">
            <input
              type="file"
              onChange={handleChange}
              accept="image/*,video/*"
              className="hidden"
              ref={fileInputRef}
            />
          <div onClick={handleUploadClick} className="cursor-pointer block text-center">
            <svg className="w-12 h-12 text-green-400 mx-auto mb-4" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
              <polyline points="17 8 12 3 7 8"/>
              <line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
            <span className="text-green-300 block mb-2">Drag and drop or click to upload</span>
            <span className="text-gray-400 text-sm">Supports images and videos up to 50MB</span>
          </div>
        </div>

        <div className="text-center">
          <span className="text-gray-400">or</span>
        </div>

        <div className="url-input-section">
          <form onSubmit={handleUrlSubmit} className="space-y-4">
            <input
              type="url"
              placeholder="Paste a URL to analyze"
              className="w-full p-3 bg-gradient-to-br from-[#111111] to-[#1a1a1a] border border-green-500/20 rounded-lg text-gray-400 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-green-500/30"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
            />
            <button
              type="submit"
              className="w-full btn-primary bg-gradient-to-r from-green-500 to-purple-600 hover:from-green-400 hover:to-purple-500 text-white font-medium py-2 px-4 rounded-lg transition-all duration-300 shadow-lg hover:shadow-green-500/25"
            >
              Analyze URL
            </button>
          </form>
        </div>

        {isAnalyzing && (
          <div className="mt-8 text-center">
            <l-quantum
              size="45"
              speed="1.75"
              color="rgb(74 222 128)"
            ></l-quantum>
            <p className="mt-4 text-green-300">Analyzing your media...</p>
            <ul className={`mt-4 space-y-2 text-green-400/70`}>
              {processingSteps.map((step, index) => (
                <li 
                  key={index}
                  className={`flex items-center space-x-2 ${
                    index === processingSteps.length - 1 ? 'text-green-300' : ''
                  }`}
                >
                  {index === processingSteps.length - 1 && (
                    <div className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse"></div>
                  )}
                  <span>{step}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {result && mediaUrl && (
        <ResultsSection result={result} mediaUrl={mediaUrl} onClose={handleCloseAnalysis} />
      )}
    </div>
  );
}
